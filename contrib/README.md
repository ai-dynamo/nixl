## Build options

### WQS SDK

`contrib/Dockerfile` can optionally build the WQS plugin using a platform-matched
SDK supplied in the repository's Docker build context. Stage the public headers
in `wqs-sdk/include/` and `libwqs_store.so` with its SPDK/DPDK runtime dependencies
in `wqs-sdk/lib/`, then build from the repository root:

```bash
docker build -f contrib/Dockerfile \
  --build-arg WQS_SDK_DIR=wqs-sdk \
  -t nixl-wqs .
```

The SDK is installed in `/opt/wqs`, its library directory is added to the loader
configuration, and its include/library paths are passed to Meson. Leaving
`WQS_SDK_DIR` empty keeps the existing container build behavior. No WQS source
checkout or SDK download is performed. See the
[WQS backend README](../src/plugins/wqs/README.md) for runtime prerequisites and
backend configuration.

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
