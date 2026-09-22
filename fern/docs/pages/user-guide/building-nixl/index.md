---
title: Building NIXL from Source
description: Build NIXL from source -- C++ library, Python bindings, Rust bindings, or Docker container.
---

Build NIXL from source using one of the following methods:

- **[Docker](/nixl/developer-guide/building-nixl-from-source/docker)** -- Build and run NIXL in a container
- **[NIXL C++ (Meson)](/nixl/developer-guide/building-nixl-from-source/nixl-c-meson)** -- Build the core C++ library
- **[Python Bindings](/nixl/developer-guide/building-nixl-from-source/python-bindings)** -- Build and install the Python package from source
- **[Rust Bindings](/nixl/developer-guide/building-nixl-from-source/rust-bindings)** -- Build the Rust bindings

## Build Options

### Common Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `build_docs` | `false` | Build Doxygen documentation |
| `build_nixl_ep` | `false` | Build NIXL-EP, Expert-parallel communication |
| `ucx_path` | `""` | Path to UCX installation |
| `libfabric_path` | `""` | Path to Libfabric installation |
| `gds_path` | `/usr/local/cuda/` | Path to GDS CuFile installation |
| `install_headers` | `true` | Install development headers |
| `disable_gds_backend` | `false` | Disable GDS backend |
| `disable_mooncake_backend` | `false` | Disable Mooncake backend |
| `cudapath_inc` | auto-detected | Custom CUDA include path |
| `cudapath_lib` | auto-detected | Custom CUDA library path |
| `cudapath_stub` | `""` | Custom CUDA stub library path |
| `etcd_inc_path` | `""` | Path to etcd headers |
| `etcd_lib_path` | `""` | Path to etcd libraries |
| `static_plugins` | `""` | Comma-separated plug-ins to build statically |
| `enable_plugins` | all | Comma-separated plug-ins to build (cannot combine with `disable_plugins`) |
| `disable_plugins` | `""` | Comma-separated plug-ins to exclude (cannot combine with `enable_plugins`) |
| `rust` | `false` | Build Rust bindings |
| `log_level` | `auto` | Log level: trace, debug, info, warning, error, fatal, auto |

Example with custom options:

```bash
meson setup build \
    -Ducx_path=/opt/ucx \
    -Denable_plugins=UCX,POSIX \
    -Drust=true
```

## Backend Specific Instructions

<Note>
Some backends have additional build requirements beyond the core prerequisites. Refer to the plug-in-specific documentation under [NIXL Backends](/nixl/user-guide/backend-selection) for details.
</Note>
