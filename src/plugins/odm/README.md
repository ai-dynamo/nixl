# ODM NIXL plugin (ODM controller + dma-buf)

`ODM` moves data between GPU VRAM and Marvell Iliad/Structera device memory
using the **ODM DMA controller + a dma-buf export of GPU VRAM**, for **both
directions**:

| Transfer pair               | Engine                       | Direction & rate (H200 + Marvell CXL) |
|-----------------------------|------------------------------|----------------------------------------|
| `VRAM_SEG <-> ODM_MEM_SEG`  | ODM DMA controller (dma-buf) | Iliad → VRAM (READ): ODM writes GPU VRAM, **~50.9 GB/s**; VRAM → Iliad (WRITE): ODM reads GPU VRAM, **~20 GB/s** |

The path exports GPU VRAM as a **dma-buf** and hands the fd to the Marvell ODM
kernel driver, which imports it (`dma_buf_dynamic_attach` with `allow_peer2peer`)
and drives the ODM DMA engine.

## Engine selection

`ODM` accepts only `VRAM_SEG <-> ODM_MEM_SEG` (see `prepXfer`). The destination
side decides the kernel command:

- data into VRAM (`Iliad -> VRAM`, READ) -> `MRVL_CXL_DMA_READ_COMMAND_FD`
  (`ODM_XTYPE_OUTBOUND`) — the ODM engine writes into GPU VRAM.
- data into ODM  (`VRAM -> Iliad`, WRITE) -> `MRVL_CXL_DMA_WRITE_COMMAND_FD`
  (`ODM_XTYPE_INBOUND`) — the ODM engine reads GPU VRAM.

Register the Iliad region as `ODM_MEM_SEG` (its device DPA range) for both
directions.

### Why WRITE is slower

VRAM → Iliad makes the ODM engine *read* GPU VRAM over PCIe. A third-party PCIe
device reading GPU memory is bounded by the GPU's read-completion path, which is
slower than the rate at which it can *write into* VRAM. Measured on H200 NVL +
Marvell CXL the WRITE direction does **not** improve with more ODM queues
(8 → 16), smaller per-descriptor size, larger PCIe `MaxReadReq` (512 → 4096),
10-bit tags, Relaxed-Ordering/IDO, or more caller threads. The kernel WRITE path
**pipelines** submissions (many outstanding PCIe reads). The READ direction (a
PCIe write into VRAM) saturates at ~50.9 GB/s.

**PCIe ACS peer-to-peer routing matters.** The GPU and the ODM device sit under
the same PCIe switch. By default the switch downstream ports have ACS
`ReqRedir+ CmpltRedir+`, which bounces GPU↔ODM peer traffic up to the root
complex instead of routing it switch-internally. That extra round-trip latency
caps large-block WRITE at **~20 GB/s**. Disabling ACS redirect on those two
downstream ports lifts large-block WRITE to **~28 GB/s** (~42%) with READ
unaffected and full data-integrity verified. Apply it with the reversible helper
(needs root; IOMMU must be off, as it is here):

```bash
sudo tests/acs_p2p.sh off       # enable direct GPU<->ODM P2P (the optimization)
sudo tests/acs_p2p.sh status    # show current ACS Control for the switch ports
sudo tests/acs_p2p.sh restore   # revert to the saved prior state
```

This is a switch-port (not ODM-device) setting, so it is applied as an operator
step rather than from the ODM driver. WARNING: disabling ACS removes PCIe
isolation between these peers; only safe when the IOMMU is off and the devices
already share data by design (not for VFIO passthrough to untrusted VMs).

The remaining ~28 → ~50 GB/s gap (vs. READ) is the GPU's intrinsic
read-completion ceiling.

## Requirements & failure policy

`ODM` has **no fallback path**. If a required capability is missing it prints a
clear error and fails instead of silently degrading:

- **No CUDA / no GPU** → the backend refuses to initialize
  (`createBackend` fails): *"ODM: CUDA with GPUDirect is NOT available …"*.
- **dma-buf export unsupported** (`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED)`
  is false) → the backend refuses to initialize: *"ODM: GPU dma-buf export is
  NOT supported on this platform …"*.

Requirements:

- CUDA (driver + runtime). The dma-buf export uses
  `cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)` and
  is gated on `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED)`.
  (CUDA is needed only to export VRAM as a dma-buf, not for any copy path.)
- **GPU VRAM must be allocated so it is dma-buf exportable** (CUDA VMM:
  `cuMemCreate` / `cuMemMap` / `cuMemSetAccess`, or `cudaMalloc` on platforms
  that support exporting it). `ODM` sets `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS` on
  the buffer.
- The Marvell ODM kernel driver built with the dma-buf import path
  (`MRVL_CXL_DMA_{WRITE,READ}_COMMAND_FD`) and `/dev/odm0` present.
- A BAR2 devdax device (e.g. `/dev/dax0.0`) is needed **only** for
  `--check_consistency` (seed/read-back via the BAR2 alias), not the data path.

## Backend options (plugin)

| Param          | Default       | Meaning                                                  |
|----------------|---------------|----------------------------------------------------------|
| `dmadev_param` | `odm0`        | ODM char device. A bare name resolves under `/dev`; an absolute path (starting with `/`) is used as-is. |
| `odm_qid`      | `0`           | ODM DMA queue id (seed for the queue range)               |
| `odm_qid_start` / `odm_qid_end` | same as `odm_qid` | ODM DMA queue range (inclusive); nixlbench defaults to `0..7` |
| `dmabuf_cache_max` | `512`     | Max dma-buf exports kept open (LRU cap); bounds open-fd growth |

nixlbench-only options such as `--dax_device` (consistency checks) are handled in
the benchmark, not in the plugin.

## Build

```
meson setup build -Denable_plugins=ODM
ninja -C build
```

ODM is built by default; the `-Denable_plugins=ODM` flag is only needed if you
previously configured the build with a different plugin set.

## Full setup: kernel module → NIXL → nixlbench

The ODM backend needs the Marvell ODM kernel driver loaded (for `/dev/odm0` and
the BAR2 devdax `/dev/dax0.0`) and NIXL built with the `ODM` plugin. Do this once
before running any benchmark. Paths below assume this repository layout.

### A. Build and load the Marvell ODM kernel driver

```bash
cd structera-cxl-host/odm/host_cxl_drivers/mrvl_cxl

# Build the PCIe driver (kernel headers only; dma-buf path uses in-kernel APIs).
make -C mrvl_cxl_pcie

# nvidia.ko is needed at runtime for CUDA/GPU dma-buf export by userspace.
sudo modprobe nvidia

# Load the driver with BAR2 exposed as device-dax (creates /dev/odm0 + /dev/dax0.0).
sudo modprobe device_dax
sudo insmod mrvl_cxl_pcie/mrvl_cxl_pcie.ko enable_dax=1

# On newer kernels device_dax may not auto-bind; create the dax char device:
sudo daxctl reconfigure-device dax0.0 --mode=devdax --force   # only if /dev/dax0.0 is missing

# Verify the driver bound and both devices exist.
lspci -d 177d: -k                         # "Kernel driver in use: mrvl_cxl_pcie"
ls -l /dev/odm0 /dev/dax0.0
cat /sys/bus/dax/devices/dax0.0/size      # must be >= --total_buffer_size you plan to use
cd ../..
```

### B. Build NIXL (with the ODM plugin) and nixlbench

```bash
cd nixl-odm

# Build NIXL with ODM into a home prefix (no sudo).
meson setup build -Dbuildtype=debug \
    -Denable_plugins=UCX,ODM \
    -Ducx_path=/opt/ucx \
    -Dprefix=$HOME/nixl-install
ninja -C build
ninja -C build install

# Build nixlbench against that NIXL.
cd benchmark/nixlbench
meson setup build -Dnixl_path=$HOME/nixl-install -Dprefix=$HOME/nixl-install
ninja -C build                            # -> benchmark/nixlbench/build/nixlbench

# Confirm the plugin and benchmark exist.
ls $HOME/nixl-install/lib/x86_64-linux-gnu/plugins/libplugin_ODM.so
cd ../..
```

### C. Point the runtime at the installed plugins/libs

```bash
export NIXL_PLUGIN_DIR=$HOME/nixl-install/lib/x86_64-linux-gnu/plugins
export LD_LIBRARY_PATH=$HOME/nixl-install/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
B=nixl-odm/benchmark/nixlbench/build/nixlbench   # nixlbench binary
```

You are now ready to run the READ / WRITE / round-trip commands below.

## Read vs write (both directions, one engine)

Both directions use the ODM DMA controller + dma-buf; only the kernel command
and DMA direction differ:

| nixlbench op | NIXL op | ODM path | Mechanism | Rate |
|--------------|---------|----------|-----------|------|
| `--op_type READ`  | `NIXL_READ`  | `VRAM_SEG <-> ODM_MEM_SEG` (`ODM_DIR_TO_GPU`, `MRVL_CXL_DMA_READ_COMMAND_FD`, `ODM_XTYPE_OUTBOUND`) | ODM *writes into* GPU VRAM (PCIe write to GPU) | ~50.9 GB/s |
| `--op_type WRITE` | `NIXL_WRITE` | `VRAM_SEG <-> ODM_MEM_SEG` (`ODM_DIR_FROM_GPU`, `MRVL_CXL_DMA_WRITE_COMMAND_FD`, `ODM_XTYPE_INBOUND`) | ODM *reads* GPU VRAM (PCIe read from GPU), pipelined | ~20 GB/s |

The WRITE direction is a PCIe read of GPU VRAM and is hardware-bounded by the
GPU's read-completion path (see "Why WRITE is slower" above). The kernel WRITE
path pipelines submissions (`odm_dma_transfer_submit_nowait` + bulk
`odm_wait_batch_completions`, up to `ODM_BATCH_POLL_DEPTH` outstanding) to reach
that ceiling; the READ path keeps the proven synchronous per-batch submit.

## How to run (read and write)

All runs use the `nixlbench` benchmark with `--backend ODM`. The initiator is
always GPU VRAM (`--initiator_seg_type VRAM`); `--op_type` selects the direction
(both use the ODM/dma-buf engine).

These assume you have completed "Full setup" above (driver loaded, NIXL +
nixlbench built, `NIXL_PLUGIN_DIR` / `LD_LIBRARY_PATH` / `B` exported).

### Automatic ODM base address and queues

Two things that previously had to be passed on the command line are now handled
automatically for the `ODM` backend:

- **ODM DMA target (IOVA).** nixlbench allocates a mailbox IOVA via
  `MRVL_CXL_GET_IOVA_COMMAND` on `/dev/odm0` and uses that as
  `target_iova_addr` for DMA. CXL **IDENTIFY** still reports the volatile DPA
  base (e.g. 32 GiB → `0x800000000`) for the BAR2/DAX host alias only — it is
  **not** a valid DMA target on `/dev/odm0`. Override with the `ODM_ADDR` env
  var if needed. The old `--odm_addr` flag has been **removed**.
- **ODM queue range.** Both directions are hardcoded to spray across queues
  **0..7**, so `--odm_qid_start/--odm_qid_end` are not needed.

The startup banner shows what was detected, e.g.:

```
ODM backend: dma_device=odm0 qid=0 qid_range=0..7 ... addr=auto(GET_IOVA)
ODM: allocated IOVA 0xfffe0000 size 4096 via GET_IOVA on /dev/odm0
     (IDENTIFY DPA base 0x800000000 used for DAX alias only)
```

The kernel driver also logs the same value at load (`dmesg | grep odm_addr`):

```
ODM: odm_addr (device DPA base) = 0x800000000  (CXL total_capacity = 32 GiB)
ODM: BAR2 (DAX host window) base = 0x7d000000000, size = 0x1000000000
```

### 1. READ — Iliad → VRAM (ODM DMA controller + dma-buf)

The ODM engine *writes into* GPU VRAM (the fast PCIe-write direction). The ODM
base address and the multi-queue fan-out are now both automatic (see "Automatic
ODM base address and queues" below), so the command is just:

```bash
$B --backend ODM --initiator_seg_type VRAM --op_type READ \
   --start_block_size $((64*1024*1024)) --max_block_size $((64*1024*1024)) \
   --start_batch_size 1 --max_batch_size 1 \
   --total_buffer_size $((1024*1024*1024)) \
   --num_iter 100 --warmup_iter 20 --num_threads 1
```

- The ODM DMA IOVA is allocated automatically via `GET_IOVA` (see above); there
  is no `--odm_addr` flag.
- Both directions automatically spray across ODM queues **0..7** (the
  plugin splits each transfer across all of them), so a single caller thread
  saturates PCIe (for READ; WRITE is GPU-read-completion bound). There is no need
  to pass `--odm_qid_start/--odm_qid_end`.
- For small blocks, sweep `--start_batch_size/--max_batch_size` (e.g. 1, 8, 16,
  32) to amortize the per-submit latency.
- `--device_list odm0` selects the ODM char device (default `odm0`).

### 2. WRITE — VRAM → Iliad (ODM DMA controller + dma-buf)

The ODM engine *reads* GPU VRAM over PCIe (pipelined). Same single engine, no
extra flags:

```bash
$B --backend ODM --initiator_seg_type VRAM --op_type WRITE \
   --start_block_size $((64*1024*1024)) --max_block_size $((64*1024*1024)) \
   --start_batch_size 1 --max_batch_size 1 \
   --total_buffer_size $((1024*1024*1024)) \
   --num_iter 100 --warmup_iter 20 --num_threads 1
```

- This direction is GPU-read-completion bound (~20 GB/s); see "Why WRITE is
  slower". Batching and the 8-queue spray are applied automatically.
- The kernel pipelines the WRITE submission ring (many outstanding PCIe reads)
  to reach that ceiling.

### 3. Round trip (write then read back, with verification)

Write `0xbb` to Iliad via ODM/dma-buf, then read the same physical memory back
via ODM/dma-buf and byte-verify it. An explicit `--check_value 0xbb` makes the
READ path **skip its default re-seed** (which would otherwise overwrite Iliad
DRAM with `0xaa`), so LEG 2 reads back exactly what LEG 1 wrote.

Here a 256 MiB region is moved as 16 MiB blocks; the batch tiles the whole region
(`batch = 256MiB / 16MiB = 16`) so every byte is written and verified.

```bash
# LEG 1: WRITE VRAM -> Iliad (ODM/dma-buf), writes pattern 0xbb.
$B --backend ODM --initiator_seg_type VRAM --op_type WRITE \
   --start_block_size $((16*1024*1024)) --max_block_size $((16*1024*1024)) \
   --start_batch_size 16 --max_batch_size 16 \
   --total_buffer_size $((256*1024*1024)) \
   --num_iter 4 --warmup_iter 1 --num_threads 1

# LEG 2: READ Iliad -> VRAM (ODM/dma-buf) of the same memory, verify 0xbb.
sudo env NIXL_PLUGIN_DIR=$NIXL_PLUGIN_DIR LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $B --backend ODM --initiator_seg_type VRAM --op_type READ \
   --start_block_size $((16*1024*1024)) --max_block_size $((16*1024*1024)) \
   --start_batch_size 16 --max_batch_size 16 \
   --total_buffer_size $((256*1024*1024)) \
   --num_iter 4 --warmup_iter 1 --num_threads 1 \
   --dax_device /dev/dax0.0 --check_consistency 1 --check_value 0xbb
```

Ready-made scripts live in `tests/`: `tests/roundtrip.sh` (write-then-read
integrity, both legs ODM/dma-buf), `tests/01_smallblock_sweep.sh` (small-block
sweep over both directions and batch sizes). They source `tests/common.sh` for
the environment and device defaults (`DAX`, `ODM`, `ODM_ADDR`).

#### Single-command consistency check (`--check_consistency`)

`--check_consistency 1` byte-verifies a single direction on its own (no manual
round trip needed). Because the ODM/Iliad memory is a *device* address space,
ODM reaches it for verification via the BAR2 DAX alias (`/dev/dax0.0`):

- **READ** expects the destination VRAM to read back `0xaa`
  (`XFERBENCH_TARGET_BUFFER_ELEMENT`). The worker **seeds** the ODM source DRAM
  with `0xaa` through the BAR2 DAX window before the transfer
  (`seedOdmDramForRead`), then verifies the read-back. (Skipped when an explicit
  `--check_value` is given — see the round trip above.)
- **WRITE** writes `0xbb` (`XFERBENCH_INITIATOR_BUFFER_ELEMENT`) into Iliad via
  the ODM engine; the target `ODM_MEM_SEG` (DPA) is read back via the BAR2 DAX
  alias rather than dereferenced as a host pointer.

Both paths mmap the root-only `/dev/dax0.0`, so **consistency runs need `sudo`**
(plain bandwidth runs do not). `--check_value 0xNN` overrides the expected byte.

```bash
# READ consistency (auto-seeds Iliad with 0xaa, then verifies):
sudo env NIXL_PLUGIN_DIR=$NIXL_PLUGIN_DIR LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $B --backend ODM --initiator_seg_type VRAM --op_type READ \
     --total_buffer_size $((256*1024*1024)) --dax_device /dev/dax0.0 \
     --check_consistency 1

# WRITE consistency (writes 0xbb, reads back via the BAR2 DAX alias):
sudo env NIXL_PLUGIN_DIR=$NIXL_PLUGIN_DIR LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $B --backend ODM --initiator_seg_type VRAM --op_type WRITE \
     --total_buffer_size $((256*1024*1024)) --dax_device /dev/dax0.0 \
     --check_consistency 1
```

### 4. Best-performance commands and measured results

Highest-bandwidth invocation for each direction on an H200 NVL + Marvell CXL
card. Both use **batch size 64** to amortize per-transfer latency on
small/medium blocks; both fan across **8 ODM queues** automatically.

```bash
# READ (Iliad -> VRAM): ODM dma-buf, 8 queues + batch 64  -> ~50.9 GB/s
$B --backend ODM --initiator_seg_type VRAM --op_type READ \
   --start_batch_size 64 --max_batch_size 64

# WRITE (VRAM -> Iliad): ODM dma-buf (pipelined) + batch 64  -> ~20 GB/s
$B --backend ODM --initiator_seg_type VRAM --op_type WRITE \
   --start_batch_size 64 --max_batch_size 64
```

Measured bandwidth (block-size sweep, `--total_buffer_size` 8 GiB,
`--num_threads 1`, batch 64), both directions over the ODM controller + dma-buf:

| Block size | READ GB/s (Iliad→VRAM) | WRITE GB/s (VRAM→Iliad) |
|-----------:|-----------------------:|------------------------:|
| 4 KB       | ~1.0                   | 0.75                    |
| 64 KB      | ~8                     | 11.64                   |
| 128 KB     | ~28                    | 18.21                   |
| 256 KB     | 43.50                  | 18.87                   |
| 1 MB       | 48.93                  | 19.65                   |
| 4 MB       | 50.33                  | 19.91                   |
| 16 MB      | 50.80                  | 19.98                   |
| 64 MB      | **50.87**              | **20.01**               |

- READ peaks **~50.9 GB/s** (≥256 KB) — the ODM engine writes into GPU VRAM (a
  PCIe write); the multi-queue spray saturates PCIe.
- WRITE peaks **~20 GB/s** (≥256 KB) — the ODM engine reads GPU VRAM (a PCIe
  read), bounded by the GPU's read-completion path. This is a **hardware
  ceiling**: it does not move with more ODM queues (8→16), smaller
  per-descriptor size, larger PCIe `MaxReadReq` (512→4096), or more caller
  threads. Reaching ~38 GB/s VRAM→Iliad would require the transfer to be a PCIe
  *write* (the removed GPU copy-engine path).
- Batching (64) is what lifts small/medium blocks; below ~64 KB both directions
  are latency-bound.

### 4a. Complete sweep (both directions, copy-paste)

A single nixlbench invocation already sweeps every power-of-two block size
between `--start_block_size` and `--max_block_size`, so a full block-size sweep
for one direction is one command. The block below runs the whole matrix — both
directions, a batch-size sweep, and prints a compact `dir block batch GB/s`
table. It needs the runtime env from `tests/common.sh` (or set `NIXL_PLUGIN_DIR`
/ `LD_LIBRARY_PATH` / `$B` yourself).

```bash
# Runtime env + nixlbench path (or export these yourself).
source tests/common.sh           # sets $B, NIXL_PLUGIN_DIR, LD_LIBRARY_PATH, ODM_ADDR

TBUF=$((8*1024*1024*1024))       # 8 GiB registered working set
SBLK=4096                        # 4 KiB ...
MBLK=$((64*1024*1024))           # ... 64 MiB (nixlbench doubles in between)

printf '%-6s %-12s %-6s %-12s\n' dir block batch GB/s
for op in READ WRITE; do
  for batch in 1 8 64; do
    log=$(mktemp)
    "$B" --backend ODM --initiator_seg_type VRAM --op_type "$op" \
         --start_block_size "$SBLK" --max_block_size "$MBLK" \
         --start_batch_size "$batch" --max_batch_size "$batch" \
         --total_buffer_size "$TBUF" \
         --num_iter 50 --warmup_iter 10 --num_threads 1 >"$log" 2>&1
    # nixlbench prints one data row per block size (lines starting with a digit).
    awk -v d="$op" -v b="$batch" '/^[0-9]/ {printf "%-6s %-12s %-6s %-12s\n", d, $1, b, $3}' "$log"
    rm -f "$log"
  done
done
```

Notes:
- Drop the batch loop (use a single `batch=64`) for the headline large-block
  numbers; keep `1 8 64` to also see the latency-bound small-block behavior.
- Add `--check_consistency 1 --dax_device /dev/dax0.0` (and run under `sudo -E`)
  to byte-verify each transfer during the sweep; consistency reads the BAR2 DAX
  alias and therefore needs root.
- WRITE bandwidth depends on PCIe ACS routing (see "Why WRITE is slower"). For
  the higher WRITE numbers, enable direct GPU<->ODM P2P first and revert after:

```bash
sudo tests/acs_p2p.sh off       # large-block WRITE ~20 -> ~28 GB/s
# ... run the sweep above ...
sudo tests/acs_p2p.sh restore   # put ACS back to its prior state
```

- Ready-made wrappers: [`tests/microlever_sweep.sh`](../../../../tests/microlever_sweep.sh)
  runs the WRITE block-size sweep and prints a `block GB/s` table (env overrides
  `BLOCKS`, `BATCH`, `ITER`, `WARMUP`, `TBUF`); `tests/01_smallblock_sweep.sh`
  sweeps small blocks across both directions and batch sizes;
  `tests/roundtrip.sh` does a write-then-read integrity round trip.

Measured large-block WRITE with ACS redirect disabled (same sweep, batch 64):
~28 GB/s vs. ~20 GB/s with the default ACS redirect; READ is unchanged at
~50.9 GB/s.

### 5. nixlbench parameter reference (ODM) and typical use

#### ODM / ODM-specific parameters

| Parameter | Default | What it does | Typical use |
|-----------|---------|--------------|-------------|
| `--backend ODM` | — | Select the ODM dma-buf backend. | Always, for this plugin. |
| `--initiator_seg_type VRAM` | `DRAM` | The initiator buffer is GPU VRAM (required for dma-buf export). | Always `VRAM` for ODM. |
| `--op_type [READ,WRITE]` | `READ` | `READ` = Iliad → VRAM (ODM writes VRAM, ~50.9 GB/s); `WRITE` = VRAM → Iliad (ODM reads VRAM, ~20 GB/s). Both over ODM/dma-buf. | Pick the direction to measure. |
| `--dax_device /dev/daxX.Y` | `/dev/dax0.0` | BAR2 devdax window, used **only** by `--check_consistency` (seed/read-back of Iliad DRAM via the alias). | Consistency runs only. |
| `--device_list odm0` | `all`→`odm0` | ODM char device (bare name resolves under `/dev`, or absolute path). | Non-default ODM device node. |
| `ODM_ADDR=0x...` (env) | auto (`GET_IOVA`) | Override the mailbox-allocated DMA IOVA (skips `GET_IOVA`). | Only for fixed-address testing. |

> **Removed from nixlbench CLI:** `--odm_addr`. The DMA IOVA comes from
> `GET_IOVA` on `/dev/odm0` and the spray is fixed at queues **0..7**.

#### Kernel module parameters (dma-buf FD path)

| Param | Default | What it does |
|-------|---------|--------------|
| `dmabuf_trace` | `1` (on) | Log the dma-buf importer API lifecycle (attach/map/unmap/detach) and the per-transfer engine/direction marker. |
| `odm_fd_inst_size` | `0` (= HW max) | Experimental per-descriptor byte cap for the dma-buf FD path. Lowering it adds outstanding PCIe reads per queue but gave **no** WRITE speedup (GPU-read-completion bound) and can regress READ, so the default is the HW max. |
| `odm_dmabuf_cache` / `odm_dmabuf_cache_max` | on / `1024` | Cache the `{dma_buf -> attach + sg_table}` so attach/map happen once per exported buffer. |

#### Common sizing / iteration parameters

| Parameter | Default | What it does | Typical use |
|-----------|---------|--------------|-------------|
| `--start_block_size N` / `--max_block_size N` | `4096` / `67108864` | Per-transfer block size; nixlbench sweeps powers of two from start to max. | Set both equal to measure one size; leave default for a full sweep. |
| `--start_batch_size N` / `--max_batch_size N` | `1` / `1` | Number of blocks submitted per transfer. **Key for small blocks** — batching amortizes per-submit latency (e.g. 256 KB READ: ~25 → ~45 GB/s at batch 64). | Sweep `1,8,16,32,64` for small blocks; tile a whole region with `batch = region / block`. |
| `--total_buffer_size N` | `8589934592` (8 GiB) | Size of the registered working set (VRAM + ODM/BAR2). Does **not** by itself set how much is moved per transfer. | Match to the region under test (e.g. `64 GiB` for a full-BAR2 sweep); must be ≤ BAR2 size. |
| `--num_iter N` / `--warmup_iter N` | `1008` / `112` | Measured / warmup iterations (auto-reduced for large blocks via `--large_blk_iter_ftr`). | Lower (e.g. `--num_iter 4`) for quick correctness runs; default for stable bandwidth. |
| `--num_threads N` | `1` | Caller threads. ODM already saturates PCIe from **one** thread via the internal 8-queue pool, so this is usually `1`. | Leave `1`; raise only to test multi-thread submission. |
| `--check_consistency [0,1]` | `0` | Byte-verify the transfer. With `GET_IOVA`, READ seeds via host WRITE ioctl and WRITE verifies via host READ ioctl; without GET_IOVA, BAR2 DAX is used instead (**needs `sudo`** for `/dev/dax0.0`). | Data-correctness tests (see "Single-command consistency check"). |
| `--check_value BYTE` | `0` (use default) | Override the expected byte; also makes the READ path **skip its re-seed** so it reads back pre-existing data. | Round trips, e.g. `--check_value 0xbb` to read back data a prior ODM write left in Iliad. |

`NIXL_LOG_LEVEL=DEBUG` (env) prints per-transfer routing and dma-buf export;
`ERROR` quiets nixlbench to just the results table.

> Note on sizes: each transfer moves `block_size * batch_size` bytes;
> `--total_buffer_size` only sizes the registered working set. To move/verify a
> whole region, tile it with the batch size (`batch = region / block`).

Set `NIXL_LOG_LEVEL=DEBUG` to see the routing and dma-buf export per transfer,
e.g.:

```
ODM prepXfer: VRAM<->ODM op=WRITE -> WRITE_FD (VRAM->Iliad) segments=1
ODM: exported VRAM 0x.. len=0x.. as dma-buf fd=65
ODM ioctl WRITE_FD(VRAM->Iliad) fd=65 iova=0x800000000 size=.. qid=0
```

## Verifying dma-buf is used

After an `Iliad -> VRAM` (READ) or `VRAM -> Iliad` (WRITE) transfer over the ODM
path, the kernel log shows the dma-buf sharing API lifecycle. The driver traces
each importer API call plus a per-transfer engine/direction marker (gated by the
`dmabuf_trace` module param, default on):

```
sudo dmesg | grep 'dma-buf API'
# odm dma-buf API: engine=ODM dma-buf WRITE(VRAM->Iliad) qid=0 size=.. (pipelined submit)
# odm dma-buf API: dma_buf_dynamic_attach(dmabuf=..) ok attach=.. (allow_peer2peer)
# odm dma-buf API: dma_buf_map_attachment(attach=.., dir=1) ok nents=64   # dir=1=DMA_TO_DEVICE (WRITE); dir=2=DMA_FROM_DEVICE (READ)
cat /sys/kernel/debug/dma_buf/bufinfo   # exported buffer attached to the ODM PCIe dev
```

Set `dmabuf_trace=0` (e.g. `insmod ... dmabuf_trace=0`, or write the sysfs
param) to silence these. The dma-buf attachment cache (`odm_dmabuf_cache`,
default on) means `attach`/`map` happen once per exported buffer and the
`unmap`/`detach`/`put` only on cache eviction/flush.

## Contiguous-descriptor coalescing (small/medium-block speedup)

A batched transfer (e.g. `--batch_size 64`) arrives at `prepXfer` as N separate
descriptors. When the blocks tile the registered buffers they are contiguous in
**both** the GPU VA space and the ODM device-address space, so `prepXfer` merges
adjacent `[addr, addr+len)` runs into one larger segment before the multi-queue
split. This collapses many tiny DMAs into a few large ones — one dma-buf export,
ioctl, doorbell, and completion wait per merged run instead of per block — which
is the dominant cost for small batched blocks. It only joins truly adjacent
ranges, so it is always correct and a no-op for non-contiguous batches.

Measured impact (batch 64, single caller thread), small/medium blocks improve up
to ~12x; large blocks are unchanged (already at their ceilings):

| Block | READ before → after | WRITE before → after |
|------:|--------------------:|---------------------:|
| 4 KB  | ~1.0 → **11.8** | 0.75 → **9.1** |
| 32 KB | ~6.8 → **27.3** | 5.7  → **19.6** |
| 64 KB | ~8   → **30.3** | 11.6 → **18.5** |
| 128 KB| ~28  → **42.6** | 18.2 → **18.8** |

Other control-plane optimizations were already in place or not applicable: the
host→device doorbell is already a posted MMIO write (the device never reads a
host tail pointer); an ODM instruction is already batched to the HW limit of 4
src/4 dst SGEs with up to `odm_batch_poll_depth` (module param, default 64,
clamp 1–128) instructions pipelined; the
payload is GPU VRAM so it cannot be inlined into a descriptor; and the completion
poll uses the `ODM_VDMA_CNT` register, which aggregates all completions into a
single read.

## Register-time VRAM export (WRITE latency-neutral)

- **`registerMem(VRAM_SEG)`** pre-exports the registered range in dma-buf-sized
  chunks (`vram_preexport_chunks`) and pins them until `deregisterMem`, so the
  first `postXfer` avoids a cold `cuMemGetHandleForAddressRange`. Steady-state
  transfers reuse these via the LRU `dmabuf_cache_`.
- **Worker threads** call `ensureCudaContext` once per dequeued `OdmWork` in
  `odmWorkerLoop`, not on every `odmDoWork` sub-chunk.

## Write-path tuning (VRAM -> Iliad) and its ceiling

The WRITE direction makes the ODM engine *read* GPU VRAM over PCIe, which is
latency-bound. The kernel FD path (`odm_dma_fd_xfer`) therefore **pipelines** the
WRITE submission: it issues all descriptor batches with
`odm_dma_transfer_submit_nowait` (keeping up to **`odm_batch_poll_depth`**
(default 64) instructions outstanding on the ring) and reaps completions in bulk with one
`odm_wait_batch_completions`, instead of waiting after each 4-descriptor batch.
Adjacent split SGEs are **merged** in the kernel (`odm_fd_merge_contiguous_sg`)
up to `MAX_ODM_SIZE_PER_INST` so fewer ODM instructions run. The per-transfer SGE
arrays are allocated with `kvmalloc_array` (freed with `kvfree`). The READ
direction keeps the proven synchronous per-batch submit (pipelining it regressed
it with deep completion-poll timeouts).

**Kernel sysfs:** `/sys/module/mrvl_cxl_pcie/parameters/odm_batch_poll_depth` (and
`odm_dmabuf_cache_max`, `odm_fd_inst_size`).

This pipelining brings WRITE to its **hardware ceiling of ~20 GB/s** (≥256 KB
blocks). That ceiling is the GPU's PCIe read-completion bandwidth and does **not**
move with: queue fan-out (8→16), per-descriptor size (`odm_fd_inst_size`), PCIe
`MaxReadReq` (512→4096), or caller threads (1/2/4). Reaching ~38 GB/s VRAM→Iliad
would require the transfer to be a PCIe *write* into Iliad, which only the removed
GPU copy-engine path could do.

## Read-path tuning (Iliad -> VRAM)

The ODM/dma-buf read path is optimized with two techniques:

- **Internal multi-queue pool (queues 0..7, hardcoded for ODM) — implemented + measured.**
  A single ODM hardware queue caps the dma-buf read path (~31 GB/s) and cannot be
  driven concurrently. To reach full bandwidth **from a single caller thread**
  (e.g. `nixlbench --num_threads 1`), the engine starts one persistent worker per
  queue (each owning its own work queue, so a given ODM queue is only ever touched
  by one thread — no unsafe same-queue concurrency) and **splits each transfer
  across all queues in parallel** (`postOdmDmabuf` → per-queue `OdmWork` items;
  workers run `odmDoWork`). A transfer is only fanned across more queues while each
  piece stays >= the per-direction min split (`ODM_QSPLIT_MIN_PIECE_READ` 4 MiB /
  `ODM_QSPLIT_MIN_PIECE_WRITE` 4 MiB), **except** WRITE segments **below 8 MiB**
  stay on **one queue** to avoid multiplying per-ioctl dma-buf attach/map cost.
  Small transfers stay on one
  queue (run inline, no pool round-trip) and large ones use the full range. Worker
  threads call `ensureCudaContext` once per work item in `odmWorkerLoop` before
  `odmDoWork`. Multiple caller threads can submit
  concurrently; each submission is tracked by its own `OdmBatch` completion
  counter.

  Measured on an H200 + Marvell CXL card, **`--num_threads 1`** with 4 internal
  queues (an earlier configuration; ODM now always uses queues 0..7):

  | block size | READ B/W (1 caller thread) |
  |------------|----------------------------|
  | 1-4 MB     | ~26-31 GB/s (single queue, inline) |
  | 8 MB       | ~39 GB/s |
  | 16 MB      | ~42 GB/s |
  | 32 MB      | ~46 GB/s |
  | 64 MB      | ~48.6 GB/s |

  Larger blocks reach the ~48-50 GB/s PCIe ceiling because each split piece stays
  large enough to amortize the kernel's per-ioctl dma-buf attach/map overhead.
  (External `--num_threads N` still works and composes with the pool.)

- **dma-buf export cache (plugin) — implemented (bounded LRU).** `exportVramDmabuf`
  caches the exported fd keyed by `(gpu_va, len)` (`dmabuf_cache_`), so a repeated
  transfer over the same VRAM region reuses one export instead of re-issuing
  `cuMemGetHandleForAddressRange` each time. The cache owns the fds and closes
  them itself; `postOdmDmabuf` no longer closes the fd per transfer. To avoid
  leaking one open fd for every distinct `(gpu_va, len)` ever exported — which
  over a long run (e.g. a nixlbench block-size sweep over a single registered
  buffer) accumulates until the process hits `RLIMIT_NOFILE` and
  `cuMemGetHandleForAddressRange(DMA_BUF_FD)` fails (`NVRM: failed to get dma-buf
  file descriptor`) — the cache is a **bounded LRU**: it keeps at most
  `dmabuf_cache_max` (default 512) exports and closes the least-recently-used
  ones once over the cap. Entries handed to an in-flight ioctl are pinned and
  released only after the kernel has taken its own `dma_buf_get` reference, so an
  fd is never closed out from under a worker; eviction only touches unpinned
  entries (the hot working set of one block-size run stays resident, stale
  entries from earlier block sizes are reclaimed).

- **Kernel dma-buf attachment cache (`odm_dmabuf_cache`) — implemented + measured.**
  Previously `odm_dma_fd_xfer` (`mrvl_cxl_core_drv.c`) ran the full
  `dma_buf_get/dynamic_attach/map/unmap/detach/put` lifecycle on every ioctl,
  which dominates small transfers. The driver now caches `{dma_buf -> attach +
  sg_table}` (keyed by dma_buf + direction) and reuses it across transfers of the
  same exported buffer; per-ioctl `dev_info` logging was also dropped to
  `dev_dbg`. Toggle with the `odm_dmabuf_cache` module param (default on).
  Assumes resident buffers (pinned VRAM + SYNC_MEMOPS); `move_notify` warns if a
  cached buffer is ever moved. This lifted large-block throughput (64 MB:
  48.6 -> 50.0 GB/s) and, combined with multi-queue + batching, made small
  transfers scale:

  | 256K config (`--num_threads 1`, 8 queues) | READ B/W |
  |-------------------------------------------|----------|
  | batch 1  | ~17-19 GB/s |
  | batch 8  | ~30 GB/s |
  | batch 16 | ~37 GB/s |
  | batch 32 | ~42 GB/s |

  A single 256K block (batch 1) is bounded by the synchronous ODM DMA
  submit+completion-poll latency (~11us), which the cache cannot remove; batching
  multiple blocks across queues amortizes it.

Remaining (not yet implemented):

- **io_uring async submission.** The dma-buf FD path uses synchronous ioctl
  submission from userspace. The kernel pipelines the WRITE submission ring
  internally (see "Write-path tuning"); a userspace FD uring op would mainly help
  *single small* blocks (batch 1) get past the per-ioctl submit+poll latency.
