# ROCm Based NIXL CI Setup and Build

The AMD/ROCm build and CI path is maintained separately from the NVIDIA one.
ROCm images come from [`.ci/dockerfiles/Dockerfile.rocm`](.ci/dockerfiles/Dockerfile.rocm)
and [`.gitlab/build-rocm.sh`](.gitlab/build-rocm.sh); the NVIDIA
`Dockerfile.base` / `.gitlab/build.sh` flow is not used here and needs no
ROCm-specific changes.

## Build Process

The ROCm image is built directly on a public ROCm base and is self-contained —
it provides HIP, hipFile (`libhipfile.so`), the `ais-stats` CLI, and a UCX built
with `--with-rocm`. No CUDA stack is involved.

```bash
DOCKER_BUILDKIT=1 docker build -f .ci/dockerfiles/Dockerfile.rocm \
    --build-arg ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full \
    -t nixl-rocm .
```

`Dockerfile.rocm` is a BuildKit-parallel refactor of the sequential
`build-rocm.sh` flow: each source dependency builds in its own stage so the DAG
scheduler runs them concurrently. `build-rocm.sh` remains the reference for the
sequential build and for configuring NIXL/nixlbench
(`-Dnixlbench_gpu=rocm -Drocm_path=...`).
