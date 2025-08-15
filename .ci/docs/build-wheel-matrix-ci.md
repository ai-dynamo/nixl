# Build Wheel Matrix CI Job Documentation

## Overview

The Build Wheel Matrix CI job is a comprehensive continuous integration pipeline that builds NixL Python wheels for multiple Python versions and architectures. This document explains how the CI job works with `contrib/Dockerfile.manylinux` and `contrib/build-wheel.sh` to create distributable Python packages.

## Architecture

The CI pipeline consists of three main components:

1. **Jenkins Matrix Job** (`.ci/jenkins/lib/build-wheel-matrix.yaml`)
2. **Docker Build Environment** (`contrib/Dockerfile.manylinux`)
3. **Wheel Building Script** (`contrib/build-wheel.sh`)

## Workflow Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Jenkins       │    │   Docker         │    │   Build Wheel   │
│   Matrix Job    │───▶│   Container      │───▶│   Script        │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 1. Jenkins Matrix Configuration

### Job Structure
- **Job Name**: `nixl-ci-build-wheel`
- **Timeout**: 240 minutes
- **Failure Behavior**: Continue on failure (`failFast: false`)
- **Resources**: 10Gi memory, 10 CPU cores

### Matrix Axes
The job builds wheels for multiple combinations:

**Python Versions:**
- 3.9
- 3.10
- 3.11
- 3.12

**Architectures:**
- x86_64
- aarch64

### Docker Image Configuration
```yaml
runs_on_dockers:
  - {
      file: 'contrib/Dockerfile.manylinux',
      name: 'manylinux_2_28',
      uri: 'ci/$arch/$name_base',
      tag: '20250701',
      build_args: '--no-cache --target base --build-arg NPROC=10 --build-arg ARCH=$arch --build-arg BASE_IMAGE=harbor.mellanox.com/ucx/$arch/cuda --build-arg BASE_IMAGE_TAG=12.8-devel-manylinux--25.03'
    }
```

## 2. Docker Build Environment (`contrib/Dockerfile.manylinux`)

### Multi-Stage Build Architecture

The Dockerfile uses a multi-stage build approach with two main stages:

```dockerfile
# Stage 1: Base stage with all dependencies and build environment
ARG BASE_IMAGE
ARG BASE_IMAGE_TAG
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as base

# ... (dependency installation and build setup)

# Stage 2: Default stage that builds and generates wheels
FROM base
# ... (NixL build and wheel generation)
```

**Base Image**: `harbor.mellanox.com/ucx/$arch/cuda:12.8-devel-manylinux--25.03`

### Stage Usage Patterns

#### CI Pipeline Usage (Base Stage Only)
In the CI pipeline, we only use the `base` stage to avoid redundant wheel generation:

```yaml
build_args: '--no-cache --target base --build-arg NPROC=10 --build-arg ARCH=$arch --build-arg BASE_IMAGE=harbor.mellanox.com/ucx/$arch/cuda --build-arg BASE_IMAGE_TAG=12.8-devel-manylinux--25.03'
```

**Why Base Stage Only in CI:**
- CI builds wheels for multiple Python versions and architectures
- Each matrix combination needs a clean build environment
- Wheel generation happens in CI steps, not in the container
- Avoids storing wheels in the Docker image (which would be architecture/version specific)

#### User Usage (Default Stage)
Users can build the complete image without specifying a target:

```bash
docker build -f contrib/Dockerfile.manylinux .
```

This will:
- Use the `base` stage as foundation
- Build NixL from source
- Generate wheels for all configured Python versions
- Install the wheel for testing

**User Benefits:**
- Self-contained build environment
- Pre-built wheels ready for distribution
- Complete testing environment
- No need to run separate build steps

### Key Dependencies Installed

#### System Packages
- Development tools (gcc, g++, cmake, ninja)
- RDMA libraries (libibverbs, rdma-core)
- Networking libraries (protobuf, gRPC)
- Build tools (meson, pybind11, patchelf)

#### OpenSSL 3.x
Custom OpenSSL 3.0.16 build with proper library paths:
```dockerfile
ENV PKG_CONFIG_PATH="/usr/local/openssl3/lib64/pkgconfig:/usr/local/openssl3/lib/pkgconfig:$PKG_CONFIG_PATH"
ENV LD_LIBRARY_PATH="/usr/local/openssl3/lib64:/usr/local/openssl3/lib:$LD_LIBRARY_PATH"
```

#### gRPC and Dependencies
- gRPC v1.73.0 with SSL support
- Microsoft cpprestsdk
- etcd-cpp-apiv3

#### Rust Toolchain
- Rust 1.86.0 for native dependencies
- Architecture-specific toolchain setup

#### UCX (Unified Communication X)
- Custom UCX build with CUDA, verbs, and gdrcopy support
- Optimized for high-performance networking

### NixL Build Process

#### Environment Setup
```dockerfile
ENV VIRTUAL_ENV=/workspace/nixl/.venv
RUN uv venv $VIRTUAL_ENV --python $DEFAULT_PYTHON_VERSION && \
    uv pip install --upgrade meson pybind11 patchelf
```

#### NixL Compilation
```dockerfile
RUN rm -rf build && \
    mkdir build && \
    uv run meson setup build/ --prefix=/usr/local/nixl --buildtype=release \
    -Dcudapath_lib="/usr/local/cuda/lib64" \
    -Dcudapath_inc="/usr/local/cuda/include" && \
    cd build && \
    ninja && \
    ninja install
```

#### Library Configuration
```dockerfile
ENV LD_LIBRARY_PATH=/usr/local/nixl/lib64/:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/nixl/lib64/plugins:$LD_LIBRARY_PATH
ENV NIXL_PLUGIN_DIR=/usr/local/nixl/lib64/plugins
```

### Default Stage Wheel Generation

When building the complete image (default stage), the Dockerfile automatically generates wheels:

```dockerfile
# Create the wheel
# No need to specifically add path to libcuda.so here, meson finds the stubs and links them
ARG WHL_PYTHON_VERSIONS="3.9,3.10,3.11,3.12"
ARG WHL_PLATFORM="manylinux_2_28_$ARCH"
RUN IFS=',' read -ra PYTHON_VERSIONS <<< "$WHL_PYTHON_VERSIONS" && \
    for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do \
        ./contrib/build-wheel.sh \
            --python-version $PYTHON_VERSION \
            --platform $WHL_PLATFORM \
            --ucx-plugins-dir /usr/lib64/ucx \
            --nixl-plugins-dir $NIXL_PLUGIN_DIR \
            --output-dir dist ; \
    done

RUN uv pip install dist/nixl-*cp${DEFAULT_PYTHON_VERSION//./}*.whl
```

**Default Stage Features:**
- Builds wheels for all Python versions (3.9, 3.10, 3.11, 3.12)
- Uses manylinux_2_28 platform tags
- Installs the default Python version wheel for testing
- Wheels are stored in `/workspace/nixl/dist/` directory

## 3. Wheel Building Script (`contrib/build-wheel.sh`)

### Purpose
The script creates Python wheels that bundle all necessary native libraries and dependencies for distribution.

### Key Features

#### Argument Parsing
```bash
--python-version: Python version to build for (default: 3.12)
--platform: Platform tag (default: manylinux_2_39_$ARCH)
--output-dir: Output directory (default: dist)
--ucx-plugins-dir: UCX plugins directory
--nixl-plugins-dir: NixL plugins directory
```

#### Wheel Building Process

1. **UV Build**
   ```bash
   uv build --wheel --out-dir $TMP_DIR --python $PYTHON_VERSION
   ```

2. **Auditwheel Repair**
   ```bash
   uv run auditwheel repair --exclude libcuda.so.1 --exclude 'libssl*' --exclude 'libcrypto*' $TMP_DIR/nixl-*.whl --plat $WHL_PLATFORM --wheel-dir $OUTPUT_DIR
   ```
   - Excludes CUDA and SSL libraries (handled by system)
   - Repairs wheel for manylinux compatibility

3. **UCX Plugin Integration**
   ```bash
   uv run ./contrib/wheel_add_ucx_plugins.py --ucx-plugins-dir $UCX_PLUGINS_DIR --nixl-plugins-dir $NIXL_PLUGINS_DIR $OUTPUT_DIR/*.whl
   ```
   - Adds UCX and NixL plugins to the wheel

## 4. CI Job Steps

### Step 1: Prepare Environment
```yaml
- name: Prepare
  run: |
    uv venv $VIRTUAL_ENV --python $python_version
    uv pip install --upgrade meson pybind11 patchelf
```

### Step 2: Build NixL
```yaml
- name: Build Nixl
  run: |
    mkdir build
    uv run meson setup build/ --prefix=/usr/local/nixl --buildtype=release \
      -Dcudapath_lib="/usr/local/cuda/lib64" -Dcudapath_inc="/usr/local/cuda/include"
    ninja -C build
    ninja -C build install
    echo "/usr/local/nixl/lib/$arch-linux-gnu" > /etc/ld.so.conf.d/nixl.conf
    echo "/usr/local/nixl/lib/$arch-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf
    ldconfig
```

### Step 3: Build Wheel
```yaml
- name: Build Wheel
  run: |
    export LD_LIBRARY_PATH=/usr/local/nixl/lib64/:/usr/local/nixl/lib64/plugins:$LD_LIBRARY_PATH
    export NIXL_PLUGIN_DIR=/usr/local/nixl/lib64/plugins
    ./contrib/build-wheel.sh \
      --python-version $python_version \
      --platform "${name}_${arch}" \
      --ucx-plugins-dir /usr/lib64/ucx \
      --nixl-plugins-dir $NIXL_PLUGIN_DIR \
      --output-dir dist
```

### Step 4: Test Wheel Installation
```yaml
- name: Test Wheel Install
  run: |
    uv pip install dist/nixl-*cp"${python_version//./}"*.whl
```

## 5. Output and Artifacts

### Generated Wheels
The CI job produces wheels with naming convention:
```
nixl-{version}-cp{python_version_no_dots}-cp{python_version_no_dots}-{platform_tag}.whl
```

Example: `nixl-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl`

### Wheel Contents
- NixL Python bindings
- Native libraries (compiled with meson)
- UCX plugins for high-performance networking
- NixL plugins for extended functionality
- All dependencies bundled for manylinux compatibility

## 6. Matrix Job Execution

### Parallel Execution
- Each matrix combination runs in parallel
- Task naming: `${name}/${arch}/${python_version}/${axis_index}`
- Total combinations: 8 (4 Python versions × 2 architectures)

### Resource Allocation
- Each job gets 10Gi memory and 10 CPU cores
- Kubernetes namespace: `swx-media`
- Cloud provider: `il-ipp-blossom-prod`

## 7. Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Docker image build logs
   - Verify CUDA paths and library availability
   - Ensure all dependencies are properly installed

2. **Wheel Creation Issues**
   - Verify Python version compatibility
   - Check auditwheel repair logs
   - Ensure UCX plugins are accessible

3. **Installation Test Failures**
   - Check wheel compatibility with target platform
   - Verify library dependencies are properly bundled
   - Test wheel installation in clean environment

### Debugging Commands

```bash
# Check wheel contents
unzip -l dist/nixl-*.whl

# Verify library dependencies
ldd /path/to/nixl/library.so

# Test wheel installation
uv pip install --force-reinstall dist/nixl-*.whl
```

## 8. Maintenance

### Updating Dependencies
- Modify `contrib/Dockerfile.manylinux` for system package updates
- Update Python versions in matrix configuration
- Test new dependencies in isolated environment

### Adding New Architectures
1. Update matrix axes in YAML configuration
2. Ensure base Docker images support new architecture
3. Test build process on target architecture
4. Update wheel platform tags if needed

### Performance Optimization
- Adjust `NPROC` build argument for parallel compilation
- Monitor resource usage and adjust Kubernetes limits
- Consider caching strategies for faster builds

## 9. Related Files

- `.ci/jenkins/lib/build-wheel-matrix.yaml` - Main CI configuration
- `contrib/Dockerfile.manylinux` - Docker build environment
- `contrib/build-wheel.sh` - Wheel building script
- `contrib/wheel_add_ucx_plugins.py` - UCX plugin integration
- `pyproject.toml` - Python package configuration
- `meson.build` - Native build configuration

## 10. References

- [ManyLinux Documentation](https://github.com/pypa/manylinux)
- [Auditwheel Documentation](https://github.com/pypa/auditwheel)
- [UV Package Manager](https://docs.astral.sh/uv/)
- [Meson Build System](https://mesonbuild.com/)
- [UCX Documentation](https://openucx.org/)
