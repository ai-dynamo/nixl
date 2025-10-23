#!/bin/bash
# Script to reproduce Jenkins nixl-ci-test job build stage
# This script runs the build stage inside a GPU container matching the exact CI environment

set -e
set -u
set -o pipefail

# Colors for output
TEXT_GREEN="\033[1;32m"
TEXT_YELLOW="\033[1;33m"
TEXT_RED="\033[1;31m"
TEXT_BLUE="\033[1;34m"
TEXT_CLEAR="\033[0m"

# CI Environment Variables (exact match from test-matrix.yaml)
export CONTAINER_WORKSPACE="/workspace"
export INSTALL_DIR="${CONTAINER_WORKSPACE}/nixl_install"
export NPROC="16"
export TEST_TIMEOUT="30"

# Architecture detection (automatic)
RAW_ARCH=$(uname -m)
case "$RAW_ARCH" in
    "arm64"|"aarch64")
        ARCH="aarch64"
        ;;
    "x86_64"|"amd64")
        ARCH="x86_64"
        ;;
    *)
        echo "Warning: Unknown architecture '$RAW_ARCH', defaulting to x86_64"
        ARCH="x86_64"
        ;;
esac

# Docker images from CI configuration (exact match from test-matrix.yaml)
BASE_IMAGE="nvcr.io/nvidia/pytorch:25.02-py3"
UCX_VERSION="v1.19.0"

# Default configuration
CONTAINER_NAME="nixl-ci-test-reproduction-$$"
GPU_TEST_IMAGE="nixl-gpu-test-$$"
RUN_TESTS=false  # Default to build-only (faster)
EXTRACT_ARTIFACTS=false
INTERACTIVE=false

# Function to cleanup on exit (matches CI onfail behavior)
cleanup() {
    if [ "$INTERACTIVE" = false ]; then
        echo -e "${TEXT_YELLOW}Cleaning up container and image (matching CI cleanup)...${TEXT_CLEAR}"
        # Exact CI cleanup commands from test-matrix.yaml
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        docker image rm -f "$GPU_TEST_IMAGE" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

This script reproduces the Jenkins nixl-ci-test job build stage as it runs in CI.
It creates a GPU-enabled container and runs the build step (with optional tests).

Options:
  --ucx-ver VER       UCX version to test (master or v1.19.0) [default: v1.19.0]
  --with-tests        Run tests after build (default: build-only)
  --extract           Extract build artifacts after completion
  --interactive       Enter interactive shell after build
  --help, -h          Show this help message

Examples:
  $0                          # Build stage only (default, faster)
  $0 --with-tests             # Build + all tests (full CI reproduction)
  $0 --ucx-ver master         # Build with UCX master branch
  $0 --extract --interactive  # Build, extract artifacts, then interactive shell

The script will:
1. Check GPU availability (nvidia-smi required)
2. Build custom GPU test container (.ci/dockerfiles/Dockerfile.gpu_test)
3. Run container with full GPU access (--gpus all)
4. Execute build step: .gitlab/build.sh (default)
5. Optionally run test steps: test_cpp.sh, test_python.sh, test_nixlbench.sh (--with-tests)
6. Optionally extract artifacts and/or provide interactive shell
EOF
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ucx-ver)
                UCX_VERSION="$2"
                shift 2
                ;;
            --with-tests)
                RUN_TESTS=true
                shift
                ;;
            --extract)
                EXTRACT_ARTIFACTS=true
                shift
                ;;
            --interactive)
                INTERACTIVE=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                echo -e "${TEXT_RED}Unknown option: $1${TEXT_CLEAR}"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${TEXT_BLUE}=== Checking Prerequisites ===${TEXT_CLEAR}"
    
    # Check if we're in nixl directory
    if [ ! -f ".gitlab/build.sh" ] || [ ! -f ".ci/dockerfiles/Dockerfile.gpu_test" ]; then
        echo -e "${TEXT_RED}ERROR: Must be run from nixl project root directory${TEXT_CLEAR}"
        echo "Expected to find .gitlab/build.sh and .ci/dockerfiles/Dockerfile.gpu_test"
        exit 1
    fi
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${TEXT_RED}ERROR: Docker is required but not found${TEXT_CLEAR}"
        exit 1
    fi
    
    # Check if docker daemon is running
    if ! docker info &> /dev/null; then
        echo -e "${TEXT_RED}ERROR: Docker daemon is not running${TEXT_CLEAR}"
        echo "Please start Docker and try again"
        exit 1
    fi
    
    # Check if nvidia-smi is available (required for nixl-ci-test)
    if ! nvidia-smi &> /dev/null; then
        echo -e "${TEXT_RED}ERROR: nvidia-smi not found or no GPUs available${TEXT_CLEAR}"
        echo "nixl-ci-test requires GPU access. Run on a system with NVIDIA GPUs."
        exit 1
    fi
    
    echo "✓ Running from nixl project root"
    echo "✓ Docker is available"
    echo "✓ GPU access available"
    echo "✓ Architecture: $ARCH (auto-detected from $RAW_ARCH)"
    echo "Base Image: $BASE_IMAGE"
    echo "UCX Version: $UCX_VERSION"
    echo "Install Directory: $INSTALL_DIR"
}

# Function to setup CI environment (matching test-matrix.yaml)
setup_ci_environment() {
    echo -e "${TEXT_BLUE}=== Setting Up GPU Test Environment ===${TEXT_CLEAR}"
    
    echo -e "${TEXT_YELLOW}=== Step 2: Build GPU Test Environment ===${TEXT_CLEAR}"
    # Build GPU test container (exact CI command from test-matrix.yaml)
    if ! docker build -t "$GPU_TEST_IMAGE" \
        -f .ci/dockerfiles/Dockerfile.gpu_test \
        --build-arg BASE_IMAGE="$BASE_IMAGE" \
        --build-arg WORKSPACE="$CONTAINER_WORKSPACE" \
        .; then
        echo -e "${TEXT_RED}Build failed, cleaning up...${TEXT_CLEAR}"
        docker image rm -f "$GPU_TEST_IMAGE" 2>/dev/null || true
        exit 1
    fi
    
    echo -e "${TEXT_YELLOW}=== Step 3: Run GPU Test Environment ===${TEXT_CLEAR}"
    # Run container exactly as CI does (matching test-matrix.yaml)
    # NOTE: No volume mount needed - workspace is copied into image during build
    docker run -dt \
        --name "$CONTAINER_NAME" \
        --ulimit memlock=-1:-1 \
        --network=host \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --gpus all \
        --device=/dev/infiniband \
        --device=/dev/gdrdrv \
        -e "CONTAINER_WORKSPACE=$CONTAINER_WORKSPACE" \
        -e "INSTALL_DIR=$INSTALL_DIR" \
        -e "NPROC=$NPROC" \
        -e "TEST_TIMEOUT=$TEST_TIMEOUT" \
        "$GPU_TEST_IMAGE"
    
    echo "✓ GPU test container created: $CONTAINER_NAME"
    echo "✓ Container image: $GPU_TEST_IMAGE"
}

# Function to get environment info (exact match from test-matrix.yaml)
get_environment_info() {
    echo -e "${TEXT_GREEN}=== Step 1: Get Environment Info ===${TEXT_CLEAR}"
    
    echo "Getting environment information (exact CI commands)..."
    set +ex  # Don't exit on errors for info gathering (exact CI setting)
    
    # Print kernel version (exact CI command)
    uname -r
    
    # Print OFED info (exact CI command)
    ofed_info -s
    
    # Print nvidia drivers info (exact CI commands)
    lsmod | grep nvidia_peermem
    lsmod | grep gdrdrv
    lsmod | grep nvidia_fs
    
    # Print nvidia-smi (exact CI commands)
    nvidia-smi
    nvidia-smi topo -m
    
    # Print MPS info (exact CI command)
    pgrep -a mps
    
    # Print compute mode (exact CI command)
    nvidia-smi -q | grep -i "compute mode"
    
    # Check RDMA status (exact CI command)
    ibv_devinfo
    
    set -e  # Re-enable exit on errors
    echo -e "${TEXT_GREEN}✓ Environment info completed${TEXT_CLEAR}"
}

# Function to run CI build step (exact match from test-matrix.yaml)
run_build_step() {
    echo -e "${TEXT_GREEN}=== Step 4: Build ===${TEXT_CLEAR}"
    
    # Run build script (exact CI command from test-matrix.yaml)
    echo "Running: set -ex && UCX_VERSION=${UCX_VERSION} .gitlab/build.sh ${INSTALL_DIR}"
    docker exec -w "$CONTAINER_WORKSPACE" "$CONTAINER_NAME" \
        /bin/bash -c "set -ex && UCX_VERSION=${UCX_VERSION} .gitlab/build.sh ${INSTALL_DIR}"
    
    echo -e "${TEXT_GREEN}✓ Build step completed${TEXT_CLEAR}"
}

# Function to run CI test steps (exact match from test-matrix.yaml)
run_test_steps() {
    if [ "$RUN_TESTS" = false ]; then
        echo -e "${TEXT_YELLOW}Skipping tests (build-only mode, use --with-tests to run tests)${TEXT_CLEAR}"
        return
    fi
    
    echo -e "${TEXT_GREEN}=== Step 5: Test CPP ===${TEXT_CLEAR}"
    timeout "${TEST_TIMEOUT}m" docker exec -w "$CONTAINER_WORKSPACE" "$CONTAINER_NAME" \
        /bin/bash -c ".gitlab/test_cpp.sh ${INSTALL_DIR}"
    echo -e "${TEXT_GREEN}✓ C++ tests completed${TEXT_CLEAR}"
    
    echo -e "${TEXT_GREEN}=== Step 6: Test Python ===${TEXT_CLEAR}"
    timeout "${TEST_TIMEOUT}m" docker exec -w "$CONTAINER_WORKSPACE" "$CONTAINER_NAME" \
        /bin/bash -c ".gitlab/test_python.sh ${INSTALL_DIR}"
    echo -e "${TEXT_GREEN}✓ Python tests completed${TEXT_CLEAR}"
    
    echo -e "${TEXT_GREEN}=== Step 7: Test Nixlbench ===${TEXT_CLEAR}"
    timeout "${TEST_TIMEOUT}m" docker exec -w "$CONTAINER_WORKSPACE" "$CONTAINER_NAME" \
        /bin/bash -c ".gitlab/test_nixlbench.sh ${INSTALL_DIR}"
    echo -e "${TEXT_GREEN}✓ Nixlbench tests completed${TEXT_CLEAR}"
    
    echo -e "${TEXT_GREEN}=== Step 8: Test Rust ===${TEXT_CLEAR}"
    timeout "${TEST_TIMEOUT}m" docker exec -w "$CONTAINER_WORKSPACE" "$CONTAINER_NAME" \
        /bin/bash -c ".gitlab/test_rust.sh ${INSTALL_DIR}"
    echo -e "${TEXT_GREEN}✓ Rust tests completed${TEXT_CLEAR}"
}

# Function to extract build artifacts
extract_artifacts() {
    if [ "$EXTRACT_ARTIFACTS" = false ]; then
        return
    fi
    
    echo -e "${TEXT_BLUE}=== Extracting Build Artifacts ===${TEXT_CLEAR}"
    
    # Create local artifacts directory
    mkdir -p ./ci_artifacts
    
    # Extract install directory
    echo "Extracting install directory..."
    docker cp "$CONTAINER_NAME:$INSTALL_DIR" "./ci_artifacts/"
    
    # Extract build directory
    echo "Extracting build directory..."
    docker cp "$CONTAINER_NAME:$CONTAINER_WORKSPACE/nixl_build" "./ci_artifacts/" 2>/dev/null || true
    
    # Extract nixlbench build
    echo "Extracting nixlbench build..."
    docker cp "$CONTAINER_NAME:$CONTAINER_WORKSPACE/benchmark/nixlbench/nixlbench_build" "./ci_artifacts/" 2>/dev/null || true
    
    # Create summary
    cat > ./ci_artifacts/build_summary.txt << EOF
NIXL CI Test Artifacts
======================
Build Date: $(date)
Base Image: $BASE_IMAGE
UCX Version: $UCX_VERSION
Architecture: $ARCH
Install Directory: $INSTALL_DIR

Contents:
- nixl_install/: Main NIXL installation
- nixl_build/: Meson build directory (if available)
- nixlbench_build/: Nixlbench build directory (if available)

To use these artifacts:
export LD_LIBRARY_PATH=\$(pwd)/ci_artifacts/nixl_install/lib:\$(pwd)/ci_artifacts/nixl_install/lib/$ARCH-linux-gnu:\$LD_LIBRARY_PATH
export PATH=\$(pwd)/ci_artifacts/nixl_install/bin:\$PATH
EOF
    
    echo "✓ Artifacts extracted to ./ci_artifacts/"
    echo "✓ See ./ci_artifacts/build_summary.txt for usage instructions"
}

# Function to provide interactive shell
run_interactive() {
    if [ "$INTERACTIVE" = false ]; then
        return
    fi
    
    echo -e "${TEXT_BLUE}=== Starting Interactive Shell ===${TEXT_CLEAR}"
    echo "You are now in the CI container environment."
    echo "The build is complete and all environment variables are set."
    echo "Type 'exit' to leave the container."
    echo ""
    
    # Don't cleanup on exit for interactive mode
    trap - EXIT
    
    docker exec -it "$CONTAINER_NAME" /bin/bash
    
    # Cleanup manually after interactive session
    echo -e "${TEXT_YELLOW}Cleaning up container...${TEXT_CLEAR}"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
}

# Main execution function
main() {
    echo -e "${TEXT_GREEN}"
    echo "=============================================="
    echo "  NIXL Jenkins CI Test Build Reproduction"
    echo "=============================================="
    echo -e "${TEXT_CLEAR}"
    echo "This script reproduces the nixl-ci-test build stage"
    echo "as it runs in Jenkins CI environment."
    echo ""
    
    parse_args "$@"
    check_prerequisites
    setup_ci_environment
    
    echo -e "${TEXT_BLUE}=== Starting CI Test Pipeline ===${TEXT_CLEAR}"
    
    get_environment_info
    run_build_step
    run_test_steps
    
    echo -e "${TEXT_GREEN}"
    echo "=============================================="
    echo "  CI Test Pipeline Completed Successfully!"
    echo "=============================================="
    echo -e "${TEXT_CLEAR}"
    
    extract_artifacts
    run_interactive
    
    if [ "$INTERACTIVE" = false ]; then
        if [ "$RUN_TESTS" = true ]; then
            echo -e "${TEXT_YELLOW}Build and tests completed. Container will be cleaned up.${TEXT_CLEAR}"
        else
            echo -e "${TEXT_YELLOW}Build completed. Container will be cleaned up.${TEXT_CLEAR}"
            echo -e "${TEXT_BLUE}Tip: Use --with-tests to run the full test suite${TEXT_CLEAR}"
        fi
        if [ "$EXTRACT_ARTIFACTS" = true ]; then
            echo -e "${TEXT_BLUE}Artifacts available in ./ci_artifacts/${TEXT_CLEAR}"
        fi
    fi
}

# Run main function with all arguments
main "$@"
