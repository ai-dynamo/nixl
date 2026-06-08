# Building NIXL for AMD ROCm

Supported on AMD Instinct accelerators (MI300X, MI350X, MI355X).

## Build

```bash
./contrib/build-container.sh --rocm
```

Produces a `nixl-rocm:<version>` image containing the NIXL libraries,
the meta-loader `nixl` wheel, and the backend `nixl_rocm` wheel under
`/workspace/nixl/dist/`.

## Run

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --network=host --ipc=host --cap-add=IPC_LOCK \
  nixl-rocm:<version>
```

## Runtime knobs for AMD Pensando AINIC (ionic) NICs

The ionic kernel driver requires UCX's dmabuf-based VRAM registration
path. Without it, UCX falls back to `ibv_reg_mr` on raw VRAM pointers,
which the driver rejects with `EINVAL`. To enable the dmabuf path:

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --network=host --ipc=host --cap-add=IPC_LOCK \
  -v /boot:/boot:ro \
  -e UCX_ROCM_COPY_DMABUF=yes \
  -e UCX_ROCM_IPC_MIN_ZCOPY=0 \
  nixl-rocm:<version>
```

- `/boot:/boot:ro` lets UCX's `uct_rocm_base_is_dmabuf_supported()`
  read `/boot/config-$(uname -r)` to verify `CONFIG_PCI_P2PDMA=y` and
  `CONFIG_DMABUF_MOVE_NOTIFY=y`.
- `UCX_ROCM_COPY_DMABUF=yes` opts into the dmabuf path in the
  `rocm_copy` memory domain (default is `no` upstream).
- `UCX_ROCM_IPC_MIN_ZCOPY=0` engages the rocm_ipc zero-copy path.

These knobs are not needed for non-ionic RDMA NICs (e.g. Mellanox,
Broadcom Thor).

## nixlbench

The `nixlbench` benchmark binary is not built by the container by
default. To build it inside the running container:

```bash
cd benchmark/nixlbench
meson setup build -Duse_rocm=true -Drocm_path=$ROCM_PATH -Dnixl_path=$NIXL_PREFIX
ninja -C build
./build/nixlbench --help
```
