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

# EP path filter: flip EP_ENABLED to "false" in test-dl-ep-matrix.yaml when the
# PR changes no EP-relevant file. On a PR trigger HEAD is the refs/pull/<n>/merge
# commit, so HEAD^1..HEAD is the PR's diff against whatever branch it targets.
#
# EP sources/tests plus the CI files feeding this job's images: CI_FILES above
# minus the wheel-only manylinux Dockerfile.
EP_PATHS=(
    "examples/device/ep/"
    ".gitlab/test_ep.sh"
    ".gitlab/build.sh"
    ".ci/scripts/common.sh"
    ".ci/jenkins/lib/test-dl-ep-matrix.yaml"
    ".ci/dockerfiles/Dockerfile.base"
    ".ci/dockerfiles/Dockerfile.gpu-test"
    ".ci/dockerfiles/Dockerfile.build_helper"
)

if [ -z "${githubData:-}" ]; then
    # Manual run: a hand-picked ref could itself be a merge commit.
    echo "EP filter: no githubData (manual run) - EP steps enabled"
elif ! git rev-parse --verify -q HEAD^2 >/dev/null; then
    echo "EP filter: HEAD is not a merge commit - EP steps enabled"
else
    CHANGED_FILES=$(git diff --name-only HEAD^1 HEAD)
    echo "EP filter: changed files (HEAD^1..HEAD):"
    echo "$CHANGED_FILES" | sed 's/^/    /'
    EP_TOUCHED="false"
    for ep_path in "${EP_PATHS[@]}"; do
        # Here-string, so no pipeline runs under `set -o pipefail`.
        if grep -q "^${ep_path}" <<< "$CHANGED_FILES"; then
            EP_TOUCHED="true"
            break
        fi
    done
    if [ "$EP_TOUCHED" = "false" ]; then
        sed -i 's/EP_ENABLED: "true"/EP_ENABLED: "false"/' \
            ".ci/jenkins/lib/test-dl-ep-matrix.yaml"
        echo "No EP-relevant files changed - EP steps disabled"
    else
        echo "EP-relevant files changed - EP steps enabled"
    fi
fi
