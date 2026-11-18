# rocm-nixl

NIXL (NVIDIA Inference Xfer Library) Python bindings for AMD ROCm.

`rocm-nixl` is one wheel per Python version that holds the NIXL backends for
every supported ROCm version, currently 7.14 and 10.1. `import nixl` selects
the backend matching the ROCm version installed on the host at import time.
Linux x86_64 only.

```bash
pip install rocm-nixl
```

## With PyTorch

`rocm-nixl` does not depend on PyTorch. To install it together with AMD's
official ROCm build of PyTorch, name the extra for your ROCm version and add
AMD's package index for that ROCm version:

```bash
# ROCm 10.1
pip install "rocm-nixl[rocm10.1]" --extra-index-url https://stable.repo.amd.com/rocm/whl-next/

# ROCm 7.14
pip install "rocm-nixl[rocm7.14]" --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/
```

Each extra pins the ROCm build of `torch` exactly, so it can only be
installed from AMD's index. The `torch` package on PyPI is the CUDA build and
does not work with `rocm-nixl`.

The extras need pip 23.3 or newer (`python -m pip install -U pip`); older pip
ignores them with a warning and installs no PyTorch.

AMD's PyTorch depends on AMD's pip ROCm packages (`rocm`, `rocm-sdk-core`,
`rocm-sdk-libraries`), so the extra installs those too. The host ROCm
installation described below is still required.

## Requirements

A ROCm installation (7.14 or 10.1) is required on the host, with its
libraries resolvable by the dynamic loader (e.g. `/opt/rocm/lib` in
`LD_LIBRARY_PATH`). ROCm is not bundled in the wheel.

The ROCm version is read from an installed ROCm build of PyTorch, or from
`$ROCM_PATH` (default `/opt/rocm`): `.info/version` or `core/.info/version`.
If ROCm is not installed under `/opt/rocm`, e.g. only `/opt/rocm/core-<X.Y>`
exists, set `ROCM_PATH` to that prefix (`export ROCM_PATH=/opt/rocm/core-7.14`)
and put its `lib` directory on the loader path. If no ROCm version can be
detected, the backend for the newest ROCm version is used.

## Do not install together with `nixl`

`rocm-nixl` and `nixl` (the NVIDIA CUDA distribution) both provide the `nixl`
import package and must not be installed in the same environment.
