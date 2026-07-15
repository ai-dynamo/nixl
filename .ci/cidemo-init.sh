#!/bin/bash -eE
set -o pipefail

# CI source files whose last-touching commit determines CI_IMAGE_TAG.
# When any of these files change in a commit, the derived tag changes
# and the matrix jobs rebuild their base Docker images automatically.
CI_FILES=(
    ".ci/dockerfiles/Dockerfile.base"
    ".ci/dockerfiles/Dockerfile.gpu-test"
    ".ci/dockerfiles/Dockerfile.build_helper"
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
