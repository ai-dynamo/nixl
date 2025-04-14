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
