# Build Wheel Matrix CI Job Documentation

## Overview

The Build Wheel Matrix CI job is a comprehensive continuous integration pipeline that builds NIXL Python wheels for multiple Python versions and architectures. This document explains how the CI job works with `contrib/Dockerfile.manylinux` and `contrib/build-wheel.sh` to create distributable Python packages.

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
- **Resources**: 16Gi memory, 16 CPU cores (increased to prevent OOM during parallel compilation)

### Matrix Axes
The job builds wheels for multiple combinations:

**Python Versions:**
- 3.9 (not supported on cuda13)
- 3.10
- 3.11
- 3.12
- 3.13

**Architectures:**
- x86_64
- aarch64

**Manylinux Versions:**
- 2_28

### Docker Image Configuration
```yaml
runs_on_dockers:
  - {
      file: 'contrib/Dockerfile.manylinux',
      name: 'manylinux',
      uri: '$arch/$name_12.8',
      tag: '20250701',
      build_args: '--no-cache --target wheel-base --build-arg NPROC=10 --build-arg ARCH=$arch --build-arg BASE_IMAGE=harbor.mellanox.com/nixl/$arch/cuda --build-arg BASE_IMAGE_TAG=12.8-devel-manylinux--25.03'
    }
```

## 2. Docker Build Environment (`contrib/Dockerfile.manylinux`)

### Multi-Stage Build Architecture

The Dockerfile uses a multi-stage build approach with two main stages:

```dockerfile
# Stage 1: Base stage with all dependencies and build environment
ARG BASE_IMAGE
ARG BASE_IMAGE_TAG
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as wheel-base

# ... (dependency installation and build setup)

# Stage 2: Default stage that builds and generates wheels
FROM base
# ... (NIXL build and wheel generation)
```

**Base Image**: `harbor.mellanox.com/nixl/$arch/cuda:12.8-devel-manylinux--25.03`

**This image is from**: `gitlab-master.nvidia.com:5005/dl/dgx/cuda:12.8-devel-manylinux--25.03`

### Stage Usage Patterns

#### CI Pipeline Usage (Base Stage Only)
In the CI pipeline, we only use the `base` stage to avoid redundant wheel generation:

```yaml
build_args: '--no-cache --target base --build-arg NPROC=10 --build-arg ARCH=$arch --build-arg BASE_IMAGE=harbor.mellanox.com/nixl/$arch/cuda --build-arg BASE_IMAGE_TAG=12.8-devel-manylinux--25.03'
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
- Use the `wheel-base` stage as foundation
- Build NIXL from source
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

The CI pipeline consists of five sequential steps for each matrix combination:

### Step 1: Prepare

Sets up the Python build environment and installs dependencies:

- Creates isolated Python virtual environment for the specific Python version
- Configures UV cache directory for faster dependency resolution
- Activates virtual environment by adding its bin directory to PATH
- Installs core build tools: `meson`, `ninja`, `meson-python`
- Installs packaging utilities: `pybind11`, `patchelf`, `auditwheel`, `tomlkit`
- Installs PyTorch with CUDA support from PyTorch's wheel repository
- **Critical**: `ninja` package provides the binary needed by meson-python's isolated build environment

### Step 2: Build NIXL

Compiles NIXL native C++ libraries:

- Configures build with meson (CUDA paths, release mode, installation prefix `/usr/local/nixl`)
- Compiles C++ code using ninja with controlled parallelism (`-j16`)
- **Memory-safe**: Limits parallel jobs to 16 to prevent OOM with 16GB memory
- Installs compiled libraries to system paths
- Updates system linker cache with `ldconfig` for shared library resolution

### Step 3: Build Wheel

Creates the Python wheel package:

- Configures library paths (`LD_LIBRARY_PATH`) for wheel building
- Sets up NIXL plugin directory environment variable
- Sets `CI_BUILD_NUMBER` to tag wheel with unique build identifier
- Invokes `build-wheel.sh` with platform-specific parameters
  - Modifies `pyproject.toml` with wheel name and build ID
  - Uses `uv build` with meson-python backend
  - Controls ninja parallelism to prevent OOM during wheel build
  - Applies `auditwheel repair` to bundle dependencies for manylinux compatibility
  - Integrates UCX and NIXL plugins into the wheel package
- Outputs wheels with unique versioning (e.g., `nixl-cu12-0.7.1+build.68-cp39-...whl`)

### Step 4: Test Wheel Install

Validates the generated wheel:

- Lists generated wheel files in the dist directory for verification
- Installs the built wheel in the virtual environment
- Ensures all dependencies are properly bundled and importable
- Verifies wheel compatibility with the target platform
- **Quality gate**: Catches packaging issues before publishing

### Step 5: Publish Wheels

Uploads wheels to Artifactory PyPI repository:

- Activates virtual environment to access installed tools
- Installs `twine` package manager for PyPI uploads
- Authenticates using Jenkins credentials (injected via credential store)
- Uploads wheels matching the specific Python version pattern
- Uses verbose logging for troubleshooting upload issues
- **Parallel-safe**: Each matrix job uploads only its unique wheels (filtered by Python version)
- **Security**: Credentials are managed via Jenkins credentials binding


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
- All dependencies bundled for manylinux compatibility

## 6. Matrix Job Execution

### Parallel Execution
- Each matrix combination runs in parallel
- Task naming: `${name}_${manylinux}/${arch}/python_${python_version}`
- Total combinations: 8 (4 Python versions × 2 architectures)

### Resource Allocation
- Each job gets 16Gi memory and 16 CPU cores
- Kubernetes namespace: `swx-media`
- Cloud provider: `il-ipp-blossom-prod`

## 7. Artifactory Integration

### PyPI Repository Configuration
```yaml
credentials:
  - credentialsId: 'svcnbu-swx-hpcx-artifactory-token'
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
   - **Prevention**: Current config uses 16GB memory with `-j16` ninja jobs

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

## 11. Maintenance

### Updating Dependencies
- Modify `contrib/Dockerfile.manylinux` for system package updates
- Update Python versions in matrix configuration
- Add new packages to `pyproject.toml` `[build-system] requires` if needed
- Test new dependencies in isolated environment

### Adding New Python Versions
1. Add version to matrix axes in YAML: `python_version: ["3.9", "3.10", "3.11", "3.12", "3.13"]`
2. Ensure manylinux base image supports the Python version
3. Test wheel build for new version
4. Update documentation

### Adding New Architectures
1. Update matrix axes in YAML configuration: `arch: [x86_64, aarch64]`
2. Ensure base Docker images support new architecture
3. Test build process on target architecture
4. Update wheel platform tags if needed
5. Verify UCX and CUDA libraries are available for new arch

### Performance Optimization
- **Memory**: Adjust Kubernetes memory limits based on build needs (currently 16Gi)
- **CPU**: Modify NPROC environment variable (currently 16 for 16 cores)
- **Ninja parallelism**: Balance between build speed and memory usage (`ninja -j${NPROC}`)
- **Docker layer caching**: Ensure frequently changed files are in later layers
- Monitor resource usage and adjust limits accordingly

### Managing Build Versions
- Build IDs are automatically incremented by Jenkins
- To manually version: `export CI_BUILD_NUMBER=123` before running build-wheel.sh
- For releases, consider using semantic versioning without build IDs
- Artifactory retains all versions; implement cleanup policies as needed

## 12. Related Files

- `.ci/jenkins/lib/build-wheel-matrix.yaml` - Main CI configuration
- `contrib/Dockerfile.manylinux` - Docker build environment
- `contrib/build-wheel.sh` - Wheel building script
- `contrib/tomlutil.py` - Build configuration utility (supports --build-id)
- `contrib/wheel_add_ucx_plugins.py` - UCX plugin integration
- `pyproject.toml` - Python package configuration (must include `ninja` in build requirements)
- `meson.build` - Native build configuration

## 13. References

- [ManyLinux Documentation](https://github.com/pypa/manylinux)
- [Auditwheel Documentation](https://github.com/pypa/auditwheel)
- [UV Package Manager](https://docs.astral.sh/uv/)
- [Meson Build System](https://mesonbuild.com/)
- [Meson Python](https://meson-python.readthedocs.io/)
- [UCX Documentation](https://openucx.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PEP 440 - Version Identification](https://peps.python.org/pep-0440/)
- [JFrog Artifactory PyPI Repositories](https://www.jfrog.com/confluence/display/JFROG/PyPI+Repositories)
