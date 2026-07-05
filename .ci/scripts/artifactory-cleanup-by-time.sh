#!/bin/bash
# Deletes Artifactory artifacts matching the given JFrog CLI file spec.
# Docker image tag directories are deleted recursively, PyPI wheels are deleted individually.
#
# Usage:
#   export ARTIFACTORY_URL="https://your-instance.jfrog.io/artifactory"
#   export ARTIFACTORY_USER="user"
#   export ARTIFACTORY_TOKEN="password-or-api-key"
#   ./artifactory-cleanup-by-time.sh [--dry-run] --spec <cleanup-spec.json>
#
# Environment:
#   ARTIFACTORY_URL       Base Artifactory URL (default: https://artifactory.nvidia.com/artifactory)
#   ARTIFACTORY_USER      Artifactory username (required)
#   ARTIFACTORY_TOKEN     Password or API key (required)
#   ARTIFACTORY_API_KEY   Alternative to ARTIFACTORY_TOKEN

set -e

DRY_RUN_FLAG=""
DRY_RUN=false
SPEC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN_FLAG="--dry-run"; DRY_RUN=true; shift ;;
        --spec)
            [[ $# -lt 2 ]] && { echo "Missing value for --spec" >&2; exit 1; }
            SPEC="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$SPEC" ]] && { echo "Usage: $0 [--dry-run] --spec <cleanup-spec.json>" >&2; exit 1; }
[[ -f "$SPEC" ]] || { echo "Spec file not found: $SPEC" >&2; exit 1; }

ARTIFACTORY_URL="${ARTIFACTORY_URL:-https://artifactory.nvidia.com/artifactory}"
TOKEN="${ARTIFACTORY_TOKEN:-${ARTIFACTORY_API_KEY:?Set ARTIFACTORY_TOKEN or ARTIFACTORY_API_KEY}}"
USER="${ARTIFACTORY_USER:?Set ARTIFACTORY_USER}"

export JFROG_CLI_LOG_LEVEL=ERROR

jf config add artifactory \
    --artifactory-url "${ARTIFACTORY_URL}" \
    --password "${TOKEN}" \
    --user "${USER}" \
    --interactive=false

jf rt ping

# For Docker: delete the tag directory (repo/path/) recursively.
# For PyPI: delete the .whl file directly.
targets=$(jf rt search --spec "$SPEC" | jq -r '.[] | .path | ltrimstr("/") |
    if endswith("/manifest.json") or endswith("/list.manifest.json")
    then split("/")[:-1] | join("/") | . + "/"
    else .
    end' | sort -u)

count=$(echo "$targets" | grep -c . || true)
echo "=== Deleting ${count} artifact(s) (dry-run: $DRY_RUN) ==="
while IFS= read -r target; do
    [[ -z "$target" ]] && continue
    if jf rt del "$target" --recursive --quiet $DRY_RUN_FLAG > /dev/null; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would delete: $target"
        else
            echo "Deleted: $target"
        fi
    else
        echo "WARNING: failed to delete $target" >&2
    fi
done <<< "$targets"

echo "=== Done ==="
