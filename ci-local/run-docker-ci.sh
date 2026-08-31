#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Marvell
# SPDX-License-Identifier: Apache-2.0
#
# Local approximations of NVIDIA Jenkins / AWS CI jobs that external PRs
# cannot trigger without a maintainer /build comment.
#
# Usage:
#   ./ci-local/run-docker-ci.sh all          # run everything below (long)
#   ./ci-local/run-docker-ci.sh non-gpu    # nixl-ci-non-gpu style build+tests
#   ./ci-local/run-docker-ci.sh container    # contrib/Dockerfile full image
#   ./ci-local/run-docker-ci.sh wheel        # wheel build (manylinux)
#   ./ci-local/run-docker-ci.sh sanitizer    # asan build (subset)
#   ./ci-local/run-docker-ci.sh github       # GitHub Actions checks only

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/cuda-dl-base}"
BASE_TAG="${BASE_TAG:-25.10-cuda13.0-devel-ubuntu24.04}"
INSTALL_DIR="${INSTALL_DIR:-/opt/nixl}"
DOCKER_NET="${DOCKER_NET:---network=host}"
NPROC="${NPROC:-$(nproc)}"

log() { printf '\033[0;34m[ci-local]\033[0m %s\n' "$*"; }
die() { printf '\033[0;31m[ci-local]\033[0m %s\n' "$*" >&2; exit 1; }

need_docker() {
    command -v docker >/dev/null || die "docker not found"
}

run_github_checks() {
    log "GitHub Actions checks (clang-format, pr-size, python, copyright)"
    if [[ -x "$ROOT/ci-local/check" ]]; then
        "$ROOT/ci-local/check" || true
    else
        log "ci-local/check not found; running clang-format diff manually"
        git fetch upstream main 2>/dev/null || true
        git diff -U0 upstream/main...HEAD -- '*.cpp' '*.h' '*.hpp' '*.c' | \
            clang-format-diff-19 -p1 -style=file | head -20
    fi
}

run_non_gpu() {
    need_docker
    log "nixl-ci-non-gpu approximation: .gitlab/build.sh + test_cpp in $BASE_IMAGE"
    docker run --rm $DOCKER_NET \
        -v "$ROOT:/workspace/nixl" \
        -w /workspace/nixl \
        -e PRE_INSTALLED_ENV=true \
        -e PRE_INSTALLED_UCX_ENV=true \
        "${BASE_IMAGE}:${BASE_TAG}" \
        bash -lc "
            set -euo pipefail
            export NIXL_INSTALL_DIR=$INSTALL_DIR
            .gitlab/build.sh \$NIXL_INSTALL_DIR '' '-Denable_plugins=ODM'
            .gitlab/test_cpp.sh \$NIXL_INSTALL_DIR
        "
}

run_sanitizer() {
    need_docker
    log "nixl-ci-test-sanitizers approximation (asan+ubsan, x86_64)"
    docker run --rm $DOCKER_NET \
        -v "$ROOT:/workspace/nixl" \
        -w /workspace/nixl \
        -e PRE_INSTALLED_ENV=true \
        -e PRE_INSTALLED_UCX_ENV=true \
        "${BASE_IMAGE}:${BASE_TAG}" \
        bash -lc "
            set -euo pipefail
            export NIXL_INSTALL_DIR=$INSTALL_DIR
            .gitlab/build.sh \$NIXL_INSTALL_DIR '' '-Dsanitizer=address,undefined -Denable_plugins=ODM'
            .gitlab/test_sanitizer.sh \$NIXL_INSTALL_DIR
        "
}

run_container() {
    need_docker
    log "contrib/Dockerfile full container build (release image)"
    docker build $DOCKER_NET --platform linux/x86_64 -f contrib/Dockerfile \
        --build-arg BASE_IMAGE="$BASE_IMAGE" \
        --build-arg BASE_IMAGE_TAG="$BASE_TAG" \
        --build-arg BUILD_TYPE=release \
        --build-arg BUILD_NIXL_EP=true \
        --build-arg NPROC="$NPROC" \
        --build-arg GRPC_NPROC="$NPROC" \
        --tag nixl-full:local \
        .
}

run_wheel() {
    need_docker
    log "nixl-ci-build-wheel approximation (manylinux, python 3.12)"
    ./contrib/build-container.sh \
        --dockerfile contrib/Dockerfile.manylinux \
        --python-versions 3.12 \
        --tag nixl-wheel:local
}

run_gpu_note() {
    cat <<'EOF'
[ci-local] nixl-ci-gpu / nixl-ci-dl-gpu require NVIDIA Slurm + Artifactory images.
           Run on hardware after non-gpu build succeeds:
             .gitlab/test_gpu.sh /opt/nixl
             .gitlab/test_dl.sh /opt/nixl
[ci-local] Run AWS Tests requires AWS credentials and EFA cluster (aws_efa_validation.yml).
EOF
}

cmd="${1:-all}"
case "$cmd" in
    github)   run_github_checks ;;
    non-gpu)  run_non_gpu ;;
    sanitizer) run_sanitizer ;;
    container) run_container ;;
    wheel)    run_wheel ;;
    all)
        run_github_checks
        run_container
        run_non_gpu
        run_sanitizer
        run_wheel
        run_gpu_note
        ;;
    *)
        die "usage: $0 {all|github|non-gpu|sanitizer|container|wheel}"
        ;;
esac
