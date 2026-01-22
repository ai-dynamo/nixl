#!/bin/bash -eE

# Files to check for changes
CI_FILES=(
    ".ci/dockerfiles/Dockerfile.ci"
    ".gitlab/build.sh"
    ".ci/scripts/common.sh"
)

# YAML files containing CI_IMAGE_TAG
BUILD_MATRIX_YAML=".ci/jenkins/lib/build-matrix.yaml"
TEST_MATRIX_YAML=".ci/jenkins/lib/test-matrix.yaml"

# Function to extract CI_IMAGE_TAG from a YAML file
get_ci_image_tag() {
    local file=$1
    local ref=$2  # git ref (HEAD, HEAD~1, etc.)

    if [ -z "$ref" ]; then
        # Current working tree
        grep -E '^\s*CI_IMAGE_TAG:' "$file" | sed 's/.*CI_IMAGE_TAG:\s*"\(.*\)".*/\1/' | tr -d ' '
    else
        # Previous commit
        git show "${ref}:${file}" 2>/dev/null | grep -E '^\s*CI_IMAGE_TAG:' | sed 's/.*CI_IMAGE_TAG:\s*"\(.*\)".*/\1/' | tr -d ' '
    fi
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Not a git repository. Skipping CI_IMAGE_TAG check."
    echo "Ok"
    exit 0
fi

# Check if any CI files were changed in the last commit or working tree
files_changed=false
for file in "${CI_FILES[@]}"; do
    # Check if file was changed in HEAD commit or has uncommitted changes
    if git diff --name-only HEAD~1 HEAD 2>/dev/null | grep -q "^${file}$" || \
       git diff --name-only 2>/dev/null | grep -q "^${file}$" || \
       git diff --cached --name-only 2>/dev/null | grep -q "^${file}$"; then
        echo "Detected changes in: $file"
        files_changed=true
    fi
done

if [ "$files_changed" = false ]; then
    echo "No changes detected in CI files. Skipping CI_IMAGE_TAG check."
    echo "Ok"
    exit 0
fi

# Files were changed, now check if CI_IMAGE_TAG was increased
echo "CI files were modified. Checking if CI_IMAGE_TAG was increased..."

# Get current and previous CI_IMAGE_TAG values
current_build_image_tag=$(get_ci_image_tag "$BUILD_MATRIX_YAML" "")
current_test_image_tag=$(get_ci_image_tag "$TEST_MATRIX_YAML" "")

previous_build_image_tag=$(get_ci_image_tag "$BUILD_MATRIX_YAML" "HEAD~1")
previous_test_image_tag=$(get_ci_image_tag "$TEST_MATRIX_YAML" "HEAD~1")

echo "Build Matrix CI_IMAGE_TAG: $previous_build_image_tag -> $current_build_image_tag"
echo "Test Matrix CI_IMAGE_TAG:  $previous_test_image_tag -> $current_test_image_tag"

# Check if CI_IMAGE_TAG was changed in both files
build_tag_changed=false
test_tag_changed=false

if [ "$current_build_image_tag" != "$previous_build_image_tag" ]; then
    echo "✓ CI_IMAGE_TAG in build-matrix.yaml was updated"
    build_tag_changed=true
fi

if [ "$current_test_image_tag" != "$previous_test_image_tag" ]; then
    echo "✓ CI_IMAGE_TAG in test-matrix.yaml was updated"
    test_tag_changed=true
fi

if [ "$build_tag_changed" = false ] || [ "$test_tag_changed" = false ]; then
    echo ""
    echo "❌ ERROR: You have changed CI files but forgot to increase CI_IMAGE_TAG!"
    echo ""
    echo "Changed files:"
    for file in "${CI_FILES[@]}"; do
        if git diff --name-only HEAD~1 HEAD 2>/dev/null | grep -q "^${file}$" || \
           git diff --name-only 2>/dev/null | grep -q "^${file}$" || \
           git diff --cached --name-only 2>/dev/null | grep -q "^${file}$"; then
            echo "  - $file"
        fi
    done
    echo ""
    echo "Please update CI_IMAGE_TAG in:"
    echo "  - $BUILD_MATRIX_YAML (line 45)"
    echo "  - $TEST_MATRIX_YAML (line 33)"
    echo ""
    exit 1
fi
