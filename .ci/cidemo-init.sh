#!/bin/bash -eE
set -o pipefail

# CI source files whose last-touching commit determines CI_IMAGE_TAG.
# When any of these files change in a commit, the derived tag changes
# and the matrix jobs rebuild their base Docker images automatically.
CI_FILES=(
    ".ci/dockerfiles/Dockerfile.base"
    ".ci/dockerfiles/Dockerfile.gpu-test"
    ".ci/dockerfiles/Dockerfile.build_helper"
    ".ci/patches/nixl_ep_vllm_release_test.patch"
    ".gitlab/build.sh"
    ".ci/scripts/common.sh"
    "contrib/Dockerfile.manylinux"
)

# Matrix YAML files that contain the CI_MANAGED placeholder.
# These are patched in the Jenkins workspace before the matrix library
# reads them — no commit or push is made.
YAML_FILES=(
    ".ci/jenkins/lib/build-matrix.yaml"
    ".ci/jenkins/lib/test-matrix.yaml"
    ".ci/jenkins/lib/test-dl-matrix.yaml"
    ".ci/jenkins/lib/test-dl-ep-matrix.yaml"
    ".ci/jenkins/lib/test-sanitizer-matrix.yaml"
    ".ci/jenkins/lib/build-wheel-matrix.yaml"
)

# Derive the tag from the most recent commit that touched any CI file.
NEW_TAG=$(git log -1 --format=%h -- "${CI_FILES[@]}")

# Fallback: if no commit has ever touched those files (should not happen
# in practice), use a sha256sum of their content truncated to 12 chars.
if [ -z "$NEW_TAG" ]; then
    echo "Warning: git log returned empty for CI files. Falling back to content hash."
    NEW_TAG=$(cat "${CI_FILES[@]}" | sha256sum | cut -c1-12)
fi

echo "CI_IMAGE_TAG derived as: ${NEW_TAG}"

for yaml in "${YAML_FILES[@]}"; do
    grep -q 'CI_IMAGE_TAG: "CI_MANAGED"' "$yaml" || { echo "ERROR: CI_MANAGED placeholder missing in $yaml" >&2; exit 1; }
    sed -i "s/CI_IMAGE_TAG: \"CI_MANAGED\"/CI_IMAGE_TAG: \"${NEW_TAG}\"/" "$yaml"
    echo "Patched: $yaml"
done

# --- nixl-ci-build-wheel-nightly -------------------------------------------
# The nightly builds its wheel_base through runs_on_dockers so the deps compile
# once per arch instead of once per matrix cell. Those images are built before
# any step runs, so the base image cannot be selected by the shell conditional
# the Build Wheel step used to carry - and build_args is a static string, so it
# cannot branch on CUDA_MAJOR either. Resolve it here instead, where the job
# parameter is readable, and patch the values in before the matrix is read.
# CUDA_MAJOR is a parameter of that job alone, so an unset one means this is
# some other job and there is nothing to patch — not a CUDA 13 default.
NIGHTLY_YAML=".ci/jenkins/lib/build-wheel-nightly-matrix.yaml"
if [ -n "${CUDA_MAJOR:-}" ] && grep -q 'NIGHTLY_MANAGED' "$NIGHTLY_YAML" 2>/dev/null; then
    # Empty for CUDA 13 so contrib/Dockerfile.manylinux's own pins apply, as
    # everywhere else since #2205. CUDA 12 is the one line that needs an
    # override, and it cannot be composed from the major - the tags differ in
    # the minor.
    case "${CUDA_MAJOR}" in
        12) BASE_ARGS="--build-arg BASE_IMAGE=nvcr.io/nvidia/cuda --build-arg BASE_IMAGE_TAG=12.9.1-devel-ubi8 --build-arg CUDA_VERSION=12.9" ;;
        13) BASE_ARGS="" ;;
        *)  echo "ERROR: unsupported CUDA_MAJOR=${CUDA_MAJOR}" >&2; exit 1 ;;
    esac
    # Unique per run: must not collide with the CI_IMAGE_TAG-keyed wheel_base
    # that per-PR builds cache on, since this one is built from NIXL_VERSION's
    # Dockerfile and configured for a specific CUDA line.
    WHEEL_BASE_TAG="nightly-cu${CUDA_MAJOR}-${BUILD_NUMBER:-0}"

    sed -i \
        -e "s|NIGHTLY_MANAGED_BASE_ARGS|${BASE_ARGS}|" \
        -e "s|NIGHTLY_MANAGED_WHEEL_BASE_TAG|${WHEEL_BASE_TAG}|" \
        "$NIGHTLY_YAML"
    echo "Patched: $NIGHTLY_YAML (cu${CUDA_MAJOR}, wheel_base tag ${WHEEL_BASE_TAG})"
fi
