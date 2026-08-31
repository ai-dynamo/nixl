# ODM NIXL plugin

Marvell ODM moves data between GPU VRAM and Iliad/Structera device memory using
the ODM DMA controller with GPU VRAM exported as a dma-buf (`VRAM_SEG <->
`ODM_MEM_SEG`).

## Build

```bash
meson setup build -Denable_plugins=ODM
ninja -C build
```

## Requirements

- CUDA with GPUDirect dma-buf export (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED`)
- ODM kernel character device (for example `/dev/odm0`)
- VRAM allocated with dma-buf-exportable memory (CUDA VMM or `cudaMalloc`)

## Addressing

DMA targets use mailbox-allocated IOVA from `GET_IOVA` on `/dev/odm0`, not the
CXL IDENTIFY DPA. Override with `$ODM_ADDR` if needed. The BAR2 DAX window
(`--dax_device` in nixlbench) aliases the IDENTIFY DPA for optional consistency
checks when GET_IOVA is not used.

## nixlbench

```bash
./nixlbench --backend ODM \
  --initiator_seg_type VRAM --target_seg_type VRAM \
  --device_list odm0 --op_type WRITE
```

See `benchmark/nixlbench/README.md` for consistency-check options.

## Tests

```bash
ninja -C build test/unit/plugins/odm/odm_nixl_test
# Requires hardware: /dev/odm0 and a CUDA GPU with dma-buf export.
```
