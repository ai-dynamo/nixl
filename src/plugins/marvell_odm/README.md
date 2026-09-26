# MARVELL_ODM NIXL Plugin

Marvell ODM moves data between host/GPU memory and Structera device memory
using the ODM DMA controller. Two transfer modes are supported:

| Mode | Segment pair | Kernel ioctl | GPU required |
|------|-------------|--------------|--------------|
| dma-buf | `VRAM_SEG` <-> `DRAM_SEG` | `READ/WRITE_COMMAND_FD` | Yes |
| host-VA | `DRAM_SEG` <-> `DRAM_SEG` | `READ/WRITE_COMMAND` | No |

For the host-VA path, **local** `DRAM_SEG` is host DRAM (userspace virtual
address) and **remote** `DRAM_SEG` is Structera device DRAM. Register remote
`DRAM_SEG` with `addr=0` to auto-allocate device IOVA inside the plugin; any
other address is treated as caller-supplied host VA or pre-allocated IOVA:

- **WRITE**: host DRAM -> device DRAM
- **READ**: device DRAM -> host DRAM

NOTE: io_uring spray and pre-pinned pages are not implemented for the host-VA
path. Those optimizations would require kernel module changes.

## Build

```bash
meson setup build -Denable_plugins=MARVELL_ODM
ninja -C build
```

Disable with `-Ddisable_odm_backend=true`.

## Requirements

- ODM kernel character device (for example `/dev/odm0`)
- Marvell ODM kernel module loaded (provides `/dev/odm0`)
- For the VRAM dma-buf path only:
  - CUDA with GPUDirect dma-buf export (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED`)
  - VRAM allocated with dma-buf-exportable memory (CUDA VMM or `cudaMalloc`)

## Addressing

Device DRAM registrations use mailbox-allocated IOVA from `GET_IOVA` on
`/dev/odm0` when registered with `addr=0`. The plugin frees IOVA on
`deregisterMem`. Register with a non-zero address only when supplying a
caller-owned host VA or pre-allocated device IOVA.

## io_uring / SQE128

The plugin supports an optional io_uring `uring_cmd` submit path when the ODM
kernel module is loaded with `odm_use_io_uring=1`:

```bash
export ODM_USE_IO_URING=1
```

SQE128 rings are used when supported by the kernel and liburing.

## Plugin Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dmadev` | ODM device name or path | `odm0` |
| `odm_qid` | Primary queue ID | `0` |
| `odm_qid_start` | First queue in range | `0` |
| `odm_qid_end` | Last queue in range | `0` (plugin default; nixlbench passes `15` unless overridden) |
| `odm_use_io_uring` | Enable io_uring submit | off |
| `num_threads` | Worker thread count | `1` |

## nixlbench

VRAM -> device DRAM (dma-buf path):

```bash
export LD_LIBRARY_PATH=build/src/core:build/src/plugins/marvell_odm
export NIXL_PLUGIN_DIR=build/src/plugins/marvell_odm

./nixlbench --backend MARVELL_ODM \
  --initiator_seg_type VRAM --target_seg_type DRAM \
  --device_list odm0 --op_type WRITE
```

Host DRAM -> device DRAM (host-VA path): register local buffers as host
`DRAM_SEG` and remote buffers as device `DRAM_SEG` (IOVA). WRITE moves host ->
device; READ moves device -> host. No GPU is required for this path.

See `benchmark/nixlbench/marvell_odm/README.md` for sweep examples and TOML config.

## Tests

```bash
ninja -C build test/unit/plugins/marvell_odm/marvell_odm_nixl_test

# Hardware round-trip (requires /dev/odm0 and CUDA GPU):
./build/test/unit/plugins/marvell_odm/marvell_odm_nixl_test --device odm0

# With io_uring:
./build/test/unit/plugins/marvell_odm/marvell_odm_nixl_test \
  --device odm0 --odm_use_io_uring \
  --odm_qid_start 0 --odm_qid_end 15
```

Register in meson test suite with `-Denable_odm_hw_tests=true` when hardware is
present.

## See Also

- nixlbench examples: `benchmark/nixlbench/marvell_odm/`
- Main nixlbench README: `benchmark/nixlbench/README.md`
