# Build Wheel Matrix CI Job Documentation

## Overview

The Build Wheel Matrix CI job is a comprehensive continuous integration pipeline that builds NIXL Python wheels for multiple Python versions and architectures. This document explains how the CI job works with `contrib/Dockerfile.manylinux` and `contrib/build-wheel.sh` to create distributable Python packages.

## Architecture

The CI pipeline consists of four main components:

1. **Jenkins Matrix Job** (`.ci/jenkins/lib/build-wheel-matrix.yaml`)
2. **Container Build Script** (`contrib/build-container.sh`)
3. **Docker Build Environment** (`contrib/Dockerfile.manylinux`)
4. **Wheel Building Script** (`contrib/build-wheel.sh`)

## Workflow Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Jenkins       │    │  build-          │    │   Dockerfile     │    │   build-        │
│   Matrix Job    │───▶│  container.sh    │───▶│   .manylinux     │───▶│   wheel.sh      │
│  (podman runner)│    │  --wheel-base-   │    │   wheel stage    │    │  (wheel pkg)    │
│                 │    │  image <url>     │    │   only (~16 steps│    │                 │
└─────────────────┘    └──────────────────┘    └──────────────────┘    └─────────────────┘
                                                        ▲
                                               pulls pre-built image
                                               from Artifactory
                                          ┌──────────────────┐
                                          │  wheel_base stage│
                                          │  (UCX, gRPC,     │
                                          │  Rust, DOCA, …)  │
                                          │  built & pushed  │
                                          │  by CI-demo when │
                                          │  Dockerfile changes│
                                          └──────────────────┘
```

## 1. Jenkins Matrix Configuration

### Job Structure
- **Job Name**: `nixl-ci-build-wheel`
- **Timeout**: 240 minutes
- **Failure Behavior**: Continue on failure (`failFast: false`)
- **Resources**: 24Gi memory, 16 CPU cores

### Matrix Axes
The job builds wheels for multiple combinations:

**Python Versions:**
- 3.12

**Architectures:**
- x86_64
- aarch64

**Manylinux Versions:**
- 2_28

### Docker Image Configuration
```yaml
runs_on_dockers:
  - { name: "manylinux", url: "quay.io/podman/stable:v5.7.1", privileged: true }
  - { file: 'contrib/Dockerfile.manylinux', name: "nixl-wheel-base-cu13-manylinux_2_28",
      tag: "${CI_IMAGE_TAG}", build_args: '... --target wheel_base' }
```

The CI runner uses podman-in-container. `build-container.sh` is called inside this container and executes `docker build` (symlinked to podman). The CUDA-specific `nixl-wheel-base-cuXX-manylinux_2_28` entry is never used as a runner — CI-demo builds and pushes it to Artifactory whenever `CI_IMAGE_TAG` changes, so PR, nightly, and release builds can pull the pre-built `wheel_base` layer instead of compiling all dependencies from scratch.

## 2. Docker Build Environment (`contrib/Dockerfile.manylinux`)

### Multi-Stage Build Architecture

The Dockerfile has three logical stages:

- **`cuda`** — sources the versioned CUDA toolkit and cuObject pkg-config metadata from the pinned NGC UBI8 devel image.
- **`wheel_base`** — starts from the pinned public PyPA manylinux image, copies CUDA from `cuda`, and builds all slow dependencies (hwloc, OpenSSL, Abseil, gRPC, etcd, curl, AWS SDK, libxml2, Azure SDK, Rust, DOCA, gdrcopy, libfabric, Python/uv, UCX). Built and pushed to Artifactory by CI-demo when `CI_IMAGE_TAG` changes.
- **`wheel`** — `FROM $wheel_base AS wheel` — builds NIXL itself and runs `build-wheel.sh`. The `wheel_base` ARG is overridden to the Artifactory URL so this stage pulls the pre-built image as its base; only ~16 steps run per PR.

```dockerfile
ARG ARCH="x86_64"
ARG CUDA_IMAGE="nvcr.io/nvidia/cuda"
ARG CUDA_IMAGE_TAG="13.2.1-devel-ubi8"
ARG CUDA_VERSION="13.2"
ARG wheel_base=wheel_base   # overridden to Artifactory URL in CI

FROM ${CUDA_IMAGE}:${CUDA_IMAGE_TAG} AS cuda
FROM quay.io/pypa/manylinux_2_28_${ARCH}:2026.06.06-1 AS wheel_base
# ... installs UCX, gRPC, Rust, DOCA, gdrcopy, etc.

FROM $wheel_base AS wheel
ARG ARCH="x86_64"
ARG BUILD_TYPE="release"
ARG BUILD_NIXL_EP="true"
ARG UCX_SONAME_SUFFIX=""
# ... builds NIXL and generates wheels
```

**Build images**: `quay.io/pypa/manylinux_2_28_<arch>:2026.06.06-1` plus `nvcr.io/nvidia/cuda:13.2.1-devel-ubi8` for CUDA 13. CUDA 12 uses `nvcr.io/nvidia/cuda:12.9.1-devel-ubi8`.

### Stage Usage Patterns

#### CI Pipeline Usage (via build-container.sh)
In the PR CI pipeline, `build-container.sh` is called with `--wheel-base-image` pointing to the pre-built `wheel_base` image in Artifactory:

```bash
./contrib/build-container.sh \
  --cuda-image 'nvcr.io/nvidia/cuda' \
  --cuda-image-tag '13.2.1-devel-ubi8' \
  --cuda-version '13.2' \
  --wheel-base "manylinux_2_28" \
  --python-versions "${python_version}" \
  --arch ${arch} \
  --torch-versions "2.13" \
  --dockerfile contrib/Dockerfile.manylinux \
  --wheel-base-image "${registry_host}${registry_path}/${arch}/${WHEEL_BASE_IMAGE_NAME}:${CI_IMAGE_TAG}"
```

`--wheel-base-image` causes the script to pass `--build-arg wheel_base=<url> --target wheel`, so only the `wheel` stage runs and the pre-built deps are pulled from Artifactory rather than recompiled.

`--torch-versions "2.13"` limits PR builds to the latest torch version. Nightly and release builds request PyTorch 2.12 and 2.13.

#### User Usage (Full Build)
Users can build the complete image locally without specifying a target:

```bash
./contrib/build-container.sh --dockerfile contrib/Dockerfile.manylinux
```

This builds both stages from scratch without the `--wheel-base-image` override.

#### Nightly Build
The nightly/release job (`nixl-ci-build-wheel-nightly`) requests PyTorch 2.12 and 2.13 and reuses the CUDA-specific cached `wheel_base`, matching the PR compiler environment.

#### Optional: UCX spcx external plugin
`build-container.sh --build-ucx-spcx-plugin` opt-in flag fetches the internal `ucx-spcx-plugin` source on the host into the build context, compiles it against the just-built UCX inside the Dockerfile, and installs it into the UCX plugins dir where `wheel_add_ucx_plugins.py` bundles it like any other UCX module. It requires `--dockerfile contrib/Dockerfile.manylinux` and two environment variables — neither is hardcoded so the repo location and token stay out of the source and image layers:

| Env var | Purpose | Jenkins credential | Credential type |
|---|---|---|---|
| `NIXL_SPCX_PLUGIN_REPO_URL` | Git URL of the plugin repo | `ucx-plugin-gitlab-url` | Secret text |
| `NIXL_GITLAB_TOKEN` | GitLab token (`read_repository` scope) used via a git credential helper so it never appears in URLs, argv, or git error output | `svc-nixl-gitlab-token` | Username/Password (token in password field) |

The plugin ref defaults to `main` and is selectable with `--ucx-spcx-plugin-ref`. This flag is not enabled in the PR CI pipeline; it is consumed by the QA/release invocations of `build-container.sh`, which bind both credentials into the environment.

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

### NIXL Build Process

#### Environment Setup
```dockerfile
ENV VIRTUAL_ENV=/workspace/nixl/.venv
RUN uv venv $VIRTUAL_ENV --python $DEFAULT_PYTHON_VERSION && \
    uv pip install --upgrade meson pybind11 patchelf
```

#### NIXL Compilation
```dockerfile
RUN rm -rf build && \
    mkdir build && \
    meson setup build/ --prefix=/usr/local/nixl --buildtype=release \
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
The script creates Python wheels that bundle all necessary native libraries and dependencies for distribution, with optional build ID versioning.

### Key Features

#### Argument Parsing
```bash
--python-version: Python version to build for (default: 3.12)
--platform: Platform tag (default: manylinux_2_39_$ARCH)
--output-dir: Output directory (default: dist)
--ucx-plugins-dir: UCX plugins directory
--nixl-plugins-dir: NIXL plugins directory
```

#### Environment Variables
```bash
CI_BUILD_NUMBER: Optional build ID to append to version (e.g., "68")
NPROC: Number of parallel ninja jobs (default: 4)
```

#### Wheel Building Process

1. **Version Modification with Build ID**
   - Modifies `pyproject.toml` to include build number
   - Example: `0.7.1` → `0.7.1+build.68`

2. **UV Build with Ninja Control**
   - Uses meson-python build backend
   - Limits parallel compilation jobs
   - Requires `ninja` in `pyproject.toml` build requirements

3. **Auditwheel Repair**
   - Excludes system libraries (CUDA, SSL, networking)
   - Repairs wheel for manylinux compatibility
   - Bundles necessary dependencies

4. **UCX Plugin Integration**
   - Adds UCX and NIXL plugins to the wheel
   - Ensures high-performance networking capabilities

## 4. CI Job Steps

The CI pipeline consists of two sequential steps for each matrix combination:

### Step 1: Prepare

Sets up the podman container runtime and authenticates with Artifactory:

- Removes conflicting container storage config files
- Resets the podman system state (`podman system reset -f`)
- Symlinks podman binary to `/usr/bin/docker` for compatibility with build scripts
- Logs into Artifactory container registry using Jenkins credentials

### Step 2: Build Wheel

Builds only the `wheel` stage of `Dockerfile.manylinux` by pulling the pre-built `wheel_base` from Artifactory:

- Calls `contrib/build-container.sh` with `--wheel-base-image <artifactory-url>` and `--torch-versions "2.13"`
- The script invokes `docker build --target wheel --build-arg wheel_base=<url>` (podman), which:
  - Pulls the pre-built `nixl-wheel-base-cu13-manylinux_2_28:${CI_IMAGE_TAG}` image (all deps already compiled)
  - Builds NIXL from source with meson/ninja
  - Runs `build-wheel.sh` to produce the Python wheel with auditwheel repair
- Only ~16 Docker steps run per PR build (instead of the full ~53-step build)
- Wheels are written to the `dist/` directory inside the container image


## 5. Output and Artifacts

### Generated Wheels
The CI job produces wheels with naming convention including build ID:
```
nixl-cu{cuda_version}-{version}+build.{build_number}-cp{python_version_no_dots}-cp{python_version_no_dots}-{platform_tag}.whl
```

Examples:
- `nixl-cu12-0.7.1+build.68-cp39-cp39-manylinux_2_28_x86_64.whl`
- `nixl-cu12-0.7.1+build.68-cp312-cp312-manylinux_2_28_x86_64.whl`

### Build Versioning
The build system automatically appends the Jenkins build number to the package version as a PEP 440 compliant local version identifier:
- Base version: `0.7.1`
- With build ID: `0.7.1+build.68`

This is handled by `contrib/tomlutil.py` which modifies `pyproject.toml` during the build:
```python
./contrib/tomlutil.py --wheel-name nixl-cu12 --build-id 68 pyproject.toml
```

**Benefits:**
- Each CI build produces uniquely versioned wheels
- Multiple builds can coexist in Artifactory
- Easy to trace which CI build produced which wheel
- PEP 440 compliant for proper pip handling

### Wheel Contents
- NIXL Python bindings
- Native libraries (compiled with meson)
- UCX plugins for high-performance networking
- NIXL plugins for extended functionality
- Most non-system dependencies are bundled for manylinux compatibility. CUDA, cuFile, cuObject Client, and the other libraries named in `AUDITWHEEL_EXCLUDES` remain external runtime dependencies.

## 6. Matrix Job Execution

### Parallel Execution
- Each matrix combination runs in parallel
- Task naming: `${name}_${manylinux}/${arch}/python_${python_version}`
- Total combinations: 2 (1 Python version × 2 architectures)

### Resource Allocation
- Each job gets 16Gi memory and 16 CPU cores
- Kubernetes namespace: `nbu-swx-nixl`
- Cloud provider: `il-ipp-blossom-prod`

## 7. Artifactory Integration

### PyPI Repository Configuration
```yaml
credentials:
  - credentialsId: 'svc-nixl-new-artifactory-token'
    usernameVariable: 'ARTIFACTORY_USER'
    passwordVariable: 'ARTIFACTORY_TOKEN'

env:
  ARTIFACTORY_PYPI_URL: https://artifactory.nvidia.com/artifactory/api/pypi/sw-nbu-sxw-nixl-pypi-local
```

### Wheel Upload Process
Each matrix job uploads its wheels to Artifactory using `twine`:

```bash
twine upload \
  --repository-url ${ARTIFACTORY_PYPI_URL} \
  --username ${ARTIFACTORY_USER} \
  --password ${ARTIFACTORY_TOKEN} \
  --verbose \
  $DIST_DIR/nixl*cp${python_version//./}*.whl
```

**Key Points:**
- Each matrix job uploads only its specific wheels (filtered by Python version)
- Wheels are uniquely named with build ID, preventing conflicts
- Parallel uploads are safe due to unique wheel names
- Credentials are managed via Jenkins credentials store

### Installing from Artifactory

Users can install wheels from Artifactory:

```bash
# Using pip with extra index
pip install nixl-cu12 \
  --extra-index-url https://artifactory.nvidia.com/artifactory/api/pypi/sw-nbu-sxw-nixl-pypi-local

# Or configure in pip.conf
[global]
extra-index-url = https://artifactory.nvidia.com/artifactory/api/pypi/sw-nbu-sxw-nixl-pypi-local
```

## 8. Troubleshooting

### Common Issues

1. **Out of Memory (OOM) Errors**
   - **Symptom**: `c++: fatal error: Killed signal terminated program cc1plus`
   - **Cause**: Too many parallel compilation jobs exceeding available memory
   - **Solution**: Reduce `NPROC` or increase memory allocation
   - **Prevention**: Current config uses 24GB memory with `-j16` ninja jobs

2. **Ninja Not Found**
   - **Symptom**: `meson-python: error: Could not find ninja version 1.8.2 or newer`
   - **Cause**: `ninja` not in isolated build environment
   - **Solution**: Ensure `ninja` is in `pyproject.toml` `[build-system] requires`
   - **Note**: Both Python `ninja` package and system `ninja-build` should be available

3. **Meson Not Found**
   - **Symptom**: `meson: command not found`
   - **Cause**: Virtual environment not activated or PATH not set
   - **Solution**: Ensure `export PATH="$VIRTUAL_ENV/bin:$PATH"` in all build steps

4. **Build Failures**
   - Check Docker image build logs
   - Verify CUDA paths and library availability
   - Ensure all dependencies are properly installed

5. **Wheel Creation Issues**
   - Verify Python version compatibility
   - Check auditwheel repair logs
   - Ensure UCX plugins are accessible
   - Verify `pyproject.toml` has correct build requirements

6. **Installation Test Failures**
   - Check wheel compatibility with target platform
   - Verify library dependencies are properly bundled
   - Test wheel installation in clean environment

7. **Artifactory Upload Failures**
   - Verify credentials are correctly configured
   - Check repository URL is accessible
   - Ensure wheel naming follows PEP 440
   - Review twine verbose output for detailed errors

### Debugging Commands

```bash
# Check wheel contents
unzip -l dist/nixl-*.whl

# Verify library dependencies
ldd /path/to/nixl/library.so

# Test wheel installation
uv pip install --force-reinstall dist/nixl-*.whl
```

## 9. Maintenance

### Updating Dependencies
- Modify `contrib/Dockerfile.manylinux` for system package updates
- Keep the `CI_MANAGED` placeholder in all seven matrix YAMLs. `cidemo-init.sh` derives and injects the tag automatically when `Dockerfile.manylinux` or another CI source changes.
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
- Dependencies are cached in the `wheel_base` image; rebuilding it only requires bumping `CI_IMAGE_TAG`

## 10. Build Configuration Tools

### tomlutil.py
A utility script for modifying `pyproject.toml` during the build process:

```bash
# Set wheel name
./contrib/tomlutil.py --wheel-name nixl-cu12 pyproject.toml

# Add build ID to version
./contrib/tomlutil.py --wheel-name nixl-cu12 --build-id 68 pyproject.toml

# Both together
./contrib/tomlutil.py --wheel-name nixl-cu12 --build-id 68 pyproject.toml
```

**Features:**
- `--wheel-name`: Sets the package name (for CUDA variant naming)
- `--build-id`: Appends build ID as PEP 440 local version identifier
- Handles existing local version identifiers gracefully
- Used automatically by `build-wheel.sh` when `CI_BUILD_NUMBER` is set

### pyproject.toml Build Requirements

Critical dependencies in `[build-system] requires`:
```toml
[build-system]
requires = ["meson-python", "ninja", "pybind11", "patchelf", "pyyaml", "types-PyYAML", "pytest", "build", "setuptools"]
build-backend = "mesonpy"
```

**Important:** The `ninja` package must be included for isolated build environments (used by `uv build`).


## 11. Related Files

- `.ci/jenkins/lib/build-wheel-matrix.yaml` - Main CI configuration
- `contrib/Dockerfile.manylinux` - Docker build environment
- `contrib/build-wheel.sh` - Wheel building script
- `contrib/tomlutil.py` - Build configuration utility (supports --build-id)
- `contrib/wheel_add_ucx_plugins.py` - UCX plugin integration
- `pyproject.toml` - Python package configuration (must include `ninja` in build requirements)
- `meson.build` - Native build configuration

## 12. References

- [ManyLinux Documentation](https://github.com/pypa/manylinux)
- [Auditwheel Documentation](https://github.com/pypa/auditwheel)
- [UV Package Manager](https://docs.astral.sh/uv/)
- [Meson Build System](https://mesonbuild.com/)
- [Meson Python](https://meson-python.readthedocs.io/)
- [UCX Documentation](https://openucx.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PEP 440 - Version Identification](https://peps.python.org/pep-0440/)
- [JFrog Artifactory PyPI Repositories](https://www.jfrog.com/confluence/display/JFROG/PyPI+Repositories)
