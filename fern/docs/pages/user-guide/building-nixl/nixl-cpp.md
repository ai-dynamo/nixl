---
title: "NIXL C++ (Meson)"
description: Build the NIXL C++ library from source using the Meson build system.
---

## Prerequisites

Install the required system packages for your distribution:

**Ubuntu:**

```bash
sudo apt install build-essential cmake pkg-config
```

**Fedora:**

```bash
sudo dnf install gcc-c++ cmake pkg-config
```

**Python dependencies:**

```bash
pip3 install meson ninja pybind11 tomlkit
```

## Build Steps

```bash
meson setup <name_of_build_dir>
cd <name_of_build_dir>
ninja
ninja install
```

<Tip>
See [Build Options](/nixl/developer-guide/building-nixl-from-source#build-options) for the full list of Meson configuration options (e.g., `ucx_path`, `enable_plugins`, `rust`).
</Tip>

<Note>
For backend-specific build instructions, see [NIXL Backends](/nixl/user-guide/backend-selection).
</Note>
