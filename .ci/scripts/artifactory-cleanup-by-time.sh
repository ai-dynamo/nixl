#!/bin/bash
# Artifactory cleanup by time (Docker images)
# Deletes Docker images in a JFrog Artifactory repository that are older than a given age.
# Only manifest.json (and list.manifest.json) are queried; image age is based on the
# manifest's last modified date. The entire image folder is then deleted.
#
# Usage:
#   export ARTIFACTORY_URL="https://your-instance.jfrog.io/artifactory"
#   export ARTIFACTORY_USER="user"
#   export ARTIFACTORY_TOKEN="password-or-api-key"
#   ./artifactory-cleanup-by-time.sh [OPTIONS] REPO_KEY [AGE]
#
# Arguments:
#   REPO_KEY   Repository key (e.g. sw-nbu-swx-nixl-docker-local)
#   AGE        Age threshold: delete images whose manifest was last modified before this (default: 4w)
#              Examples: 4w (weeks), 30d (days), 2m (months)
#
# Options:
#   --dry-run       Only list images that would be deleted, do not delete
#   --path PATH     Only consider images under this path (AQL path match)
#
# Environment:
#   ARTIFACTORY_URL       Base Artifactory URL (required)
#   ARTIFACTORY_USER      Username for API auth (required unless using token only)
#   ARTIFACTORY_TOKEN     Password or API key (required)
#   ARTIFACTORY_API_KEY   Alternative to ARTIFACTORY_TOKEN
#
# Example:
#   ./artifactory-cleanup-by-time.sh --dry-run sw-nbu-swx-nixl-docker-local 4w
#   ./artifactory-cleanup-by-time.sh --path "verification/nixl/" sw-nbu-swx-nixl-docker-local 4w

set -e

DRY_RUN=false
PATH_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --path)
            PATH_FILTER="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--dry-run] [--path PATH] REPO_KEY [AGE]" >&2
    echo "  AGE default: 4w (4 weeks)" >&2
    exit 1
fi

REPO_KEY="$1"
AGE="${2:-4w}"

# Parse age (e.g. 4w, 30d, 2m) and compute cutoff via epoch (portable)
if [[ "$AGE" =~ ^([0-9]+)([wdm])$ ]]; then
    NUM="${BASH_REMATCH[1]}"
    UNIT="${BASH_REMATCH[2]}"
else
    echo "Invalid AGE format: $AGE (e.g. 4w, 30d, 2m)" >&2
    exit 1
fi

NOW_EPOCH=$(date +%s)
case "$UNIT" in
    w) SECS_AGO=$((NUM * 7 * 24 * 3600)) ;;
    d) SECS_AGO=$((NUM * 24 * 3600)) ;;
    m) SECS_AGO=$((NUM * 30 * 24 * 3600)) ;;
    *) echo "Invalid age unit: $UNIT (use w, d, or m)" >&2; exit 1 ;;
esac
CUTOFF_EPOCH=$((NOW_EPOCH - SECS_AGO))

if date -u -d "@${CUTOFF_EPOCH}" +"%Y-%m-%dT%H:%M:%S.000Z" &>/dev/null; then
    CUTOFF=$(date -u -d "@${CUTOFF_EPOCH}" +"%Y-%m-%dT%H:%M:%S.000Z")
elif date -u -r "${CUTOFF_EPOCH}" +"%Y-%m-%dT%H:%M:%S.000Z" &>/dev/null; then
    CUTOFF=$(date -u -r "${CUTOFF_EPOCH}" +"%Y-%m-%dT%H:%M:%S.000Z")
else
    echo "Cannot convert epoch to ISO date" >&2
    exit 1
fi

ARTIFACTORY_URL="${ARTIFACTORY_URL:-https://artifactory.nvidia.com/artifactory}"
TOKEN="${ARTIFACTORY_TOKEN:-${ARTIFACTORY_API_KEY:?Set ARTIFACTORY_TOKEN or ARTIFACTORY_API_KEY}}"
USER="${ARTIFACTORY_USER:-}"

if [[ -n "$USER" ]]; then
    CURL_AUTH=(-u "$USER:$TOKEN")
else
    CURL_AUTH=(-H "X-JFrog-Art-Api: $TOKEN")
fi

# Build AQL: only manifest files; filter by last modified date. Use $and so $or is valid.
if [[ -n "$PATH_FILTER" ]]; then
    PATH_MATCH="${PATH_FILTER%/}"
    AQL='items.find({"$and":[{"repo":"'"$REPO_KEY"'"},{"path":{"$match":"'"$PATH_MATCH"'/*"}},{"$or":[{"name":"manifest.json"},{"name":"list.manifest.json"}]},{"modified":{"$lt":"'"$CUTOFF"'"}}]}).include("repo","path","name")'
else
    AQL='items.find({"$and":[{"repo":"'"$REPO_KEY"'"},{"$or":[{"name":"manifest.json"},{"name":"list.manifest.json"}]},{"modified":{"$lt":"'"$CUTOFF"'"}}]}).include("repo","path","name")'
fi

echo "Current time (UTC): $(date -u +"%Y-%m-%dT%H:%M:%S.000Z")"
echo "Cutoff: images with manifest last modified before $CUTOFF will be deleted (repo: $REPO_KEY)"
if [[ -n "$PATH_FILTER" ]]; then
    echo "Path filter: $PATH_FILTER"
fi
if [[ "$DRY_RUN" = true ]]; then echo "DRY RUN: no artifacts will be deleted"; fi

RESPONSE=$(curl -s -w "\n%{http_code}" "${CURL_AUTH[@]}" -X POST \
    "${ARTIFACTORY_URL}/api/search/aql" \
    -H "Content-Type: text/plain" \
    -d "$AQL") || { echo "AQL request failed"; exit 1; }

# Split body and HTTP code (last line)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$RESPONSE" | sed '$d')

if ! echo "$RESPONSE_BODY" | jq -e . >/dev/null 2>&1; then
    echo "Artifactory response is not valid JSON (HTTP $HTTP_CODE). First 500 chars:"
    echo "$RESPONSE_BODY" | head -c 500
    echo ""
    exit 1
fi

# Check for AQL errors in JSON body
if echo "$RESPONSE_BODY" | jq -e '.errors' >/dev/null 2>&1; then
    echo "AQL error:"
    echo "$RESPONSE_BODY" | jq -r '.errors[]? // .'
    exit 1
fi

RESULTS_COUNT=$(echo "$RESPONSE_BODY" | jq -r '(.results // []) | length')
echo "AQL returned ${RESULTS_COUNT} manifest(s) matching criteria."

# Each result is a manifest file; the image folder is repo/path (parent of manifest). Dedupe by folder.
COUNT=0
while IFS= read -r image_path; do
    [[ -z "$image_path" ]] && continue
    COUNT=$((COUNT + 1))
    if [[ "$DRY_RUN" = true ]]; then
        echo "  [dry-run] would delete image: $image_path"
    else
        echo "  Deleting image: $image_path"
        curl -s "${CURL_AUTH[@]}" -X DELETE "${ARTIFACTORY_URL}/${image_path}" || true
    fi
done < <(echo "$RESPONSE_BODY" | jq -r '(.results // [])[] | "\(.repo)/\(.path)"' | sort -u)

echo "Done. $COUNT image(s) ${DRY_RUN:+would be }deleted."
