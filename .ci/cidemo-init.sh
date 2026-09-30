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
# The nightly builds its wheel_base images through runs_on_dockers so the deps
# compile once per arch and CUDA major instead of once per matrix cell. Those
# images are built before any step runs, so their tag cannot be set by a step -
# and it has to be unique per run, both because the nightly rebuilds them every
# time and so they cannot collide with the CI_IMAGE_TAG-keyed image that per-PR
# builds cache on.
NIGHTLY_YAML=".ci/jenkins/lib/build-wheel-nightly-matrix.yaml"
if grep -q 'NIGHTLY_MANAGED_WHEEL_BASE_TAG' "$NIGHTLY_YAML" 2>/dev/null; then
    WHEEL_BASE_TAG="nightly-${BUILD_NUMBER:-0}"
    sed -i "s|NIGHTLY_MANAGED_WHEEL_BASE_TAG|${WHEEL_BASE_TAG}|" "$NIGHTLY_YAML"
    echo "Patched: $NIGHTLY_YAML (wheel_base tag ${WHEEL_BASE_TAG})"
fi
