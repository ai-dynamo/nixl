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
