# MARVELL_ODM Backend for nixlbench

This directory contains MARVELL_ODM-specific configuration examples and
documentation for using the Marvell ODM DMA controller backend with nixlbench.

## Contents

| File | Description |
|------|-------------|
| `README.md` | This file - overview and configuration guide |
| `marvell_odm_example.conf` | Example ODM knobs (documentation; maps to CLI/env) |
| `nixlbench_marvell_odm.toml` | Complete nixlbench `--config_file` example |

## Configuration Method

MARVELL_ODM uses **nixlbench CLI flags** and a few **environment variables**.
There is no plugin config-file parser. Use `marvell_odm_example.conf` as a
reference for which CLI flag or env var each knob maps to.

## Quick Start

### Step 1: Build NIXL with the MARVELL_ODM plugin

```bash
meson setup build -Denable_plugins=MARVELL_ODM
ninja -C build
```

### Step 2: Set environment

Run from the NIXL repository root (paths stay valid after changing into the
nixlbench build directory):

```bash
export LD_LIBRARY_PATH="$(pwd)/build/src/core:$(pwd)/build/src/infra:$(pwd)/build/src/utils/stream:$(pwd)/build/src/plugins/marvell_odm"
export NIXL_PLUGIN_DIR="$(pwd)/build/src/plugins/marvell_odm"
```

### Step 3: Run nixlbench

```bash
# From the nixlbench build directory
cd benchmark/nixlbench/build
./nixlbench --config_file ../marvell_odm/nixlbench_marvell_odm.toml

# Or command-line only
./nixlbench \
  --backend MARVELL_ODM \
  --device_list odm0 \
  --initiator_seg_type VRAM \
  --target_seg_type DRAM \
  --op_type WRITE \
  --total_buffer_size 8589934592 \
  --start_block_size 4096 \
  --max_block_size 33554432 \
  --start_batch_size 64 \
  --max_batch_size 64 \
  --num_iter 112 \
  --warmup_iter 16
```

## Configuration Parameters

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ODM_USE_IO_URING` | Enable io_uring submit path (fallback if `--odm_use_io_uring` unset) | `1` |
| `NIXL_PLUGIN_DIR` | Directory containing `libplugin_MARVELL_ODM.so` | `build/src/plugins/marvell_odm` |

### nixlbench CLI Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--backend` | Must be `MARVELL_ODM` | `MARVELL_ODM` |
| `--device_list` | ODM character device | `odm0` |
| `--initiator_seg_type` | Source memory type | `VRAM` |
| `--target_seg_type` | Destination memory type | `DRAM` |
| `--op_type` | Transfer direction | `WRITE` or `READ` |
| `--check_consistency` | Verify data after WRITE | `0` or `1` |
| `--total_buffer_size` | Registered buffer size (bytes) | `8589934592` |
| `--odm_use_io_uring` | Enable io_uring `uring_cmd` submit path | off |
| `--odm_qid_start` | First ODM queue ID | `0` |
| `--odm_qid_end` | Last ODM queue ID | `15` |
| `--num_threads` | Benchmark worker threads | `1` |

### Plugin Parameters (via backend init)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `odm_qid_start` | First ODM queue ID | `0` (from `--odm_qid_start`) |
| `odm_qid_end` | Last ODM queue ID | `15` (from `--odm_qid_end`) |
| `odm_use_io_uring` | Use io_uring `uring_cmd` submit | off (from `--odm_use_io_uring`) |

## Consistency Checking

When `--check_consistency=1` is set on a WRITE benchmark, nixlbench reads device
memory back through a NIXL READ transfer and compares against the initiator
buffer. No ETCD is required for single-instance runs.

```bash
./nixlbench \
  --backend MARVELL_ODM \
  --device_list odm0 \
  --initiator_seg_type VRAM \
  --target_seg_type DRAM \
  --op_type WRITE \
  --check_consistency=1 \
  --total_buffer_size 67108864 \
  --start_block_size 4096 \
  --max_block_size 4096 \
  --start_batch_size 64 \
  --max_batch_size 64 \
  --num_iter 16
```

## How It Works

1. **nixlbench** selects backend `MARVELL_ODM` and passes device/queue params.
2. **Plugin** allocates device IOVA via `GET_IOVA` when remote `DRAM_SEG` is registered with `addr=0`.
3. **Plugin** exports GPU VRAM as dma-buf and submits ODM DMA via ioctl or io_uring.
4. **Consistency** (optional) pulls device data via NIXL READ for verification.

## See Also

- Example configs in `benchmark/nixlbench/marvell_odm/`
- Plugin README: `src/plugins/marvell_odm/README.md`
- Main nixlbench README: `benchmark/nixlbench/README.md`
