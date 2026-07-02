#!/bin/bash
# Artifactory cleanup by time
# Deletes artifacts in a JFrog Artifactory repository that are older than a given age,
# using last-downloaded time when available, modified time otherwise.
#
# Usage:
#   export ARTIFACTORY_URL="https://your-instance.jfrog.io/artifactory"
#   export ARTIFACTORY_USER="user"
#   export ARTIFACTORY_TOKEN="password-or-api-key"
#   ./artifactory-cleanup-by-time.sh [OPTIONS] --policy <policy.json>
#
# Options:
#   --dry-run       Only list images that would be deleted, do not delete
#   --policy        Path to a JSON policy file describing what and when to delete
#
# Environment:
#   ARTIFACTORY_URL       Base Artifactory URL (required)
#   ARTIFACTORY_USER      Username for API auth (required unless using token only)
#   ARTIFACTORY_TOKEN     Password or API key (required)
#   ARTIFACTORY_API_KEY   Alternative to ARTIFACTORY_TOKEN
#
# Example:
#   ./artifactory-cleanup-by-time.sh --dry-run --policy cleanup-policy.json
#   ./artifactory-cleanup-by-time.sh --policy cleanup-policy.json

set -e
export JFROG_CLI_LOG_LEVEL=ERROR

DRY_RUN=false
POLICY_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --policy)
            [[ $# -lt 2 ]] && { echo "Missing value for --policy" >&2; exit 1; }
            POLICY_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$POLICY_FILE" ]] && { echo "Usage: $0 [--dry-run] --policy <policy.json>" >&2; exit 1; }
[[ -f "$POLICY_FILE" ]] || { echo "Policy file not found: $POLICY_FILE" >&2; exit 1; }

ARTIFACTORY_URL="${ARTIFACTORY_URL:-https://artifactory.nvidia.com/artifactory}"
TOKEN="${ARTIFACTORY_TOKEN:-${ARTIFACTORY_API_KEY:?Set ARTIFACTORY_TOKEN or ARTIFACTORY_API_KEY}}"
USER="${ARTIFACTORY_USER:?Set ARTIFACTORY_USER}"

# Configure JFrog CLI for Artifactory access
JF_CONFIG_ARGS=(
    --artifactory-url "${ARTIFACTORY_URL}"
    --password "${TOKEN}"
    --interactive=false
    --user "${USER}"
    )
jf config add artifactory "${JF_CONFIG_ARGS[@]}"
if ! jf rt ping; then
    echo "ERROR: Artifactory ping failed — check ARTIFACTORY_URL and credentials" >&2
    exit 1
fi

compute_cutoff() {
    if [[ "$1" =~ ^([0-9]+)([wdm])$ ]]; then
        NUM="${BASH_REMATCH[1]}"
        UNIT="${BASH_REMATCH[2]}"
    else
        echo "Invalid age: $1" >&2; return 1
    fi
    case "$UNIT" in
        d) SECS_AGO=$((NUM * 86400)) ;;
        w) SECS_AGO=$((NUM * 604800)) ;;
        m) SECS_AGO=$((NUM * 2592000)) ;;
        *) echo "Invalid age unit: $UNIT (use d, w, m)" >&2; return 1 ;;
    esac
    CUTOFF_EPOCH=$(( $(date +%s) - SECS_AGO ))
    date -u -d "@${CUTOFF_EPOCH}" +"%Y-%m-%dT%H:%M:%S.000Z" 2>/dev/null || \
    date -u -r "${CUTOFF_EPOCH}" +"%Y-%m-%dT%H:%M:%S.000Z"
}

# Build AQL path/exclude clauses and name filter, then run the query.
run_aql() {
    local repo="$1" path="$2" exclude_path="$3" name_filter="$4" cutoff="$5"
    local path_clause="" exclude_clause=""
    [[ -n "$path" ]]         && path_clause=',{"path":{"$match":"'"${path%/}/*"'"}}'
    [[ -n "$exclude_path" ]] && exclude_clause=',{"path":{"$nmatch":"'"${exclude_path%/}/*"'"}}'
    local aql='items.find({"$and":[{"repo":"'"${repo}"'"}'"${path_clause}${exclude_clause},${name_filter}"',{"modified":{"$lt":"'"${cutoff}"'"}}]}).include("repo","path","name","modified","stat.downloaded")'
    jf rt curl -s -XPOST "/api/search/aql" -H "Content-Type: text/plain" --data "$aql"
}

TOTAL_COUNT=0
TOTAL_FAILED=0
ENTRY_COUNT=$(jq 'length' "$POLICY_FILE")

echo "Policy: $POLICY_FILE ($ENTRY_COUNT entries)"
echo "Dry-run: $DRY_RUN"
echo "Current time (UTC): $(date -u +"%Y-%m-%dT%H:%M:%S.000Z")"
echo ""

for (( i=0; i<ENTRY_COUNT; i++ )); do
    DESCRIPTION=$(jq -r ".[$i].description"   "$POLICY_FILE")
    REPO=$(        jq -r ".[$i].repo"          "$POLICY_FILE")
    PATH_FILTER=$( jq -r ".[$i].path // \"\""  "$POLICY_FILE")
    EXCLUDE_PATH=$(jq -r ".[$i].exclude_path // \"\"" "$POLICY_FILE")
    TYPE=$(        jq -r ".[$i].type"           "$POLICY_FILE")
    AGE=$(         jq -r ".[$i].age"            "$POLICY_FILE")
    CUTOFF=$(compute_cutoff "$AGE") || { echo "  ERROR: bad age '$AGE' in entry $((i+1)) — skipping" >&2; TOTAL_FAILED=$((TOTAL_FAILED + 1)); continue; }

    echo "--- [$((i+1))/$ENTRY_COUNT] $DESCRIPTION ---"
    echo "Repo: $REPO | Path: ${PATH_FILTER:-<root>} | Age: $AGE | Cutoff: $CUTOFF"

    case "$TYPE" in
        docker) NAME_FILTER='{"$or":[{"name":"manifest.json"},{"name":"list.manifest.json"}]}' ;;
        pypi)   NAME_FILTER='{"name":{"$match":"*.whl"}}' ;;
        *)      echo "Unknown type: $TYPE - skipping" >&2; continue ;;
    esac

    if ! RESPONSE=$(run_aql "$REPO" "$PATH_FILTER" "$EXCLUDE_PATH" "$NAME_FILTER" "$CUTOFF"); then
        echo "  ERROR: AQL query failed for $REPO — skipping" >&2
        TOTAL_FAILED=$((TOTAL_FAILED + 1)); continue
    fi

    case "$TYPE" in
        docker)
            TARGETS=$(jq -r --arg cutoff "$CUTOFF" '
              def newest_ts: if .stats then (.stats[0].downloaded // .modified) else .modified end;
              (.results // []) | group_by("\(.repo)/\(.path)") | .[] |
              { folder: "\(.[0].repo)/\(.[0].path)", newest: (map(newest_ts) | max) } |
              select(.newest < $cutoff) | "\(.folder)\t\(.newest)"' <<< "$RESPONSE")
            ;;
        pypi)
            TARGETS=$(jq -r --arg cutoff "$CUTOFF" '
              def newest_ts: if .stats then (.stats[0].downloaded // .modified) else .modified end;
              (.results // [])[] |
              { path: "\(.repo)/\(.path)/\(.name)", newest: newest_ts } |
              select(.newest < $cutoff) | "\(.path)\t\(.newest)"' <<< "$RESPONSE")
            ;;
    esac

    COUNT=0
    FAILED=0
    while IFS=$'\t' read -r target ts; do
        [[ -z "$target" ]] && continue
        if [[ "$DRY_RUN" = true ]]; then
            echo "  [dry-run] would delete: $target  ($ts)"
            COUNT=$((COUNT + 1))
        else
            echo "  Deleting: $target  ($ts)"
            if DEL_OUTPUT=$(jf rt del "$target" --quiet 2>&1); then
                COUNT=$((COUNT + 1))
            else
                echo "  ERROR: delete failed for $target" >&2
                echo "$DEL_OUTPUT" >&2
                FAILED=$((FAILED + 1))
            fi
        fi
    done <<< "$TARGETS"

    echo "  Result: $COUNT deleted${FAILED:+, $FAILED failed}"
    echo ""
    TOTAL_COUNT=$((TOTAL_COUNT + COUNT))
    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
done

{ [[ "$DRY_RUN" = true ]] && DRY_RUN_LABEL="would be "; } || DRY_RUN_LABEL=""

echo "=== Total: $TOTAL_COUNT artifact(s) ${DRY_RUN_LABEL}deleted${TOTAL_FAILED:+, $TOTAL_FAILED failed} ==="
[[ "$TOTAL_FAILED" -gt 0 ]] && exit 1 || exit 0
