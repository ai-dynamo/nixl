## Build options

### Build & install in a conda environment
Refer to [README.md](../README.md) for instructions on how to install prerequisites.
```
$ conda activate <ENV_NAME>
$ export PKG_CONFIG_PATH=<PATH_TO_UCX>/lib/pkgconfig:$PKG_CONFIG_PATH

$ meson setup nixl-build -Ducx_path=<PATH_TO_UCX> --prefix=<PATH_TO_CONDA_ENV>
$ cd nixl-build
$ ninja
$ ninja install

$ cd ..
$ pip install . --config-settings=setup-args="-Ducx_path=<PATH_TO_UCX>"
$ pip list | grep nixl
$ python -c "from nixl._api import nixl_agent; print('Success')"
```

### Build wheels with OpenSSL 4

`Dockerfile.manylinux` defaults to OpenSSL 3.0.16. To use OpenSSL 4, pass its
version and release SHA-256 checksum:

```bash
docker build -f contrib/Dockerfile.manylinux \
    --build-arg OPENSSL_VERSION=4.0.3 \
    --build-arg OPENSSL_SHA256=325b5c806167c13b40b1ffeadfe0248197c00eccc4cf123ec1e28d2d2fd216d9 \
    --build-arg WHL_PYTHON_VERSIONS=3.13 \
    --build-arg BUILD_NIXL_EP=false \
    -t nixl-wheel:openssl4 .
```

Run from the repository root. Wheels are written to `/workspace/nixl/dist` in
the image. They require OpenSSL 4 runtime libraries (`libssl.so.4` and
`libcrypto.so.4`), which are not bundled in the wheel.

### Build ROCm wheels

`rocm-nixl` is one wheel per Python version holding the NIXL backends of
every supported ROCm version (7.14 and 10.1, x86_64 only); `import nixl`
selects the backend matching the installed ROCm. It is built in three steps,
all run from the repository root.

1. Build the backend wheels, one image per ROCm version. `Dockerfile.manylinux`
   builds CUDA wheels by default; `--gpu-stack rocm` builds against a ROCm
   release instead:

   ```bash
   for rocm in 10.1.0 7.14.1; do
       ./contrib/build-container.sh \
           --dockerfile contrib/Dockerfile.manylinux \
           --gpu-stack rocm --rocm-version "$rocm" \
           --python-versions 3.10,3.11,3.12,3.13,3.14 \
           --tag "nixl-wheel:rocm$rocm"
       cid=$(docker create "nixl-wheel:rocm$rocm")
       docker cp "$cid:/workspace/nixl/dist" "./dist-rocm$rocm"
       docker rm "$cid"
   done
   ```

   Behind a TLS-intercepting proxy, add `--proxy-ca-cert FILE` to pass the
   proxy's root CA certificate to the build as a BuildKit secret.

   Each `dist-rocm<version>` then holds the intermediate backend wheels
   `nixl_rocm<X>_<Y>-<version>-cp3*-cp3*-manylinux_2_28_x86_64.whl`
   (distribution `nixl-rocm<X>.<Y>`, import package `nixl_rocm<X>_<Y>`), one
   per Python version. They are not published. The platform tag is
   `manylinux_2_28_x86_64` unless `--wheel-base` is given.

2. Merge the backends of each Python version into one `rocm-nixl` wheel:

   ```bash
   for py in cp310 cp311 cp312 cp313 cp314; do
       ./contrib/wheel_merge_rocm.py --output-dir dist-rocm \
           dist-rocm*/nixl_rocm*-"$py"-"$py"-*.whl
   done
   ```

   `wheel_merge_rocm.py` copies every backend, adds the `nixl` dispatcher and
   `nixl_meta_utils.py` from `--dispatcher-dir` (default
   `src/bindings/python/nixl-meta`) and writes the `rocm-nixl` metadata. It
   refuses backends that differ in NIXL version, Python/ABI tag or platform
   tag, two backends for the same ROCm version, and any file present in more
   than one input. The backends' `.data` and `.scripts` directories are left
   out, so `rocm-nixl` does not install the example programs
   (`nixl_example`, `nixl_etcd_example`, `telemetry_reader`) or the asio
   headers that the backend wheels carry.

3. The result is in `dist-rocm`:
   `rocm_nixl-<version>-cp3*-cp3*-manylinux_2_28_x86_64.whl`, one per Python
   version.

`rocm-nixl` depends on `numpy` only. It declares one extra per ROCm version
(`rocm7.14`, `rocm10.1`), each pinning AMD's ROCm build of PyTorch exactly, so
it can only be installed from AMD's index and never from PyPI, whose `torch`
is the CUDA build:

```bash
pip install "rocm-nixl[rocm10.1]" --extra-index-url https://stable.repo.amd.com/rocm/whl-next/
pip install "rocm-nixl[rocm7.14]" --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/
```

The extras need pip 23.3 or newer; older pip ignores them with a warning.
AMD's PyTorch depends on AMD's pip ROCm packages (`rocm`, `rocm-sdk-core`,
`rocm-sdk-libraries`), so the extra installs those as well.

The pins and index URLs are the `ROCM_PYTORCH` table in
`wheel_merge_rocm.py`; a ROCm version without an entry cannot be merged.

The ROCm runtime is not bundled: it must be installed on the host with its
libraries resolvable by the dynamic loader. `rocm-nixl` and `nixl` both
provide the `nixl` import package: do not install them in the same
environment.

`--cuda-version`, `--torch-cuda-index`, `--torch-versions` and
`--build-nixl-ep` are rejected with `--gpu-stack rocm` (`nixl_ep` is
CUDA-only).

Inside an environment that already has ROCm, UCX and NIXL installed the way
`Dockerfile.manylinux` sets them up, `build-wheel.sh` builds one backend wheel
into `--output-dir` (default `dist`):

```bash
./contrib/build-wheel.sh --gpu-stack rocm --rocm-path /opt/rocm --python-version 3.12
```

The wheel is tagged for `--platform` (default `manylinux_2_39_x86_64`); pass
the tag matching the glibc of the build environment, e.g.
`--platform manylinux_2_28_x86_64` in the manylinux image.
`--rocm-version X.Y[.Z]` overrides the version read from
`<rocm-path>/.info/version` (or `<rocm-path>/core/.info/version`). The script
rewrites `pyproject.toml` (name `nixl-rocm<X>.<Y>`, `torch` removed from the
dependencies), keeps the ROCm runtime out of the repaired wheel and fails if a
ROCm library was vendored or the wheel contains the `nixl` dispatcher or
`nixl_ep`.
