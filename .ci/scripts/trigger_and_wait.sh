#!/usr/bin/env bash
#
# Trigger the generic `lyris-exec` pipeline, wait for it, and fetch its results.
#
# Usage:
#   TRIGGER_TOKEN=... GITLAB_TOKEN=... \
#   trigger_and_wait.sh --ref <branch> --partition <gb200|gb300> \
#                       --image <pyxis-image> --test-cmds-file <file> \
#                       [--allow-fail "<names>"] [--retries N]
#
# Downloads the job artifacts into ./results/ (per-test <name>.rc / <name>.log,
# summary.txt, junit.xml). Exit code: 0 = pipeline ran (read ./results/ for
# per-test pass/fail), 90 = infra failure (allocation/runner/queue) after
# retries. Per-test gating is left to the caller (each test has its own .rc).
set -uo pipefail

PROJECT_API="https://gitlab-master.nvidia.com/api/v4/projects/231686"
POLL_INTERVAL=30

# --- arguments ---
ref=""; partition=""; image=""; test_cmds_file=""; allow_fail=""; retries=2
while [ $# -gt 0 ]; do
  case "$1" in
    --ref)            ref=$2;            shift 2 ;;
    --partition)      partition=$2;      shift 2 ;;
    --image)          image=$2;          shift 2 ;;
    --test-cmds-file) test_cmds_file=$2; shift 2 ;;
    --allow-fail)     allow_fail=$2;     shift 2 ;;   # passed through to the pipeline (optional)
    --retries)        retries=$2;        shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
: "${TRIGGER_TOKEN:?}" "${GITLAB_TOKEN:?}"
: "${ref:?}" "${partition:?}" "${image:?}" "${test_cmds_file:?}"

# GitLab API GET (path relative to the project), prints the response body.
api_get() { curl -sS -H "PRIVATE-TOKEN: $GITLAB_TOKEN" "$PROJECT_API$1"; }
# Extract one field from JSON on stdin, e.g. json_field "['status']".
json_field() { python3 -c "import sys,json;print(json.load(sys.stdin)$1)"; }

test_cmds_b64=$(base64 -w0 < "$test_cmds_file" 2>/dev/null || base64 < "$test_cmds_file" | tr -d '\n')

attempt=0
while :; do
  attempt=$((attempt + 1))

  # 1. trigger the pipeline with the test inputs as variables
  pipeline_id=$(curl -sS -X POST \
    -F "token=$TRIGGER_TOKEN" -F "ref=$ref" \
    -F "variables[PARTITION]=$partition" \
    -F "variables[IMAGE]=$image" \
    -F "variables[TEST_CMDS]=$test_cmds_b64" \
    -F "variables[ALLOW_FAIL]=$allow_fail" \
    "$PROJECT_API/trigger/pipeline" | json_field "['id']")
  echo "pipeline=$pipeline_id attempt=$attempt"

  # 2. poll until it reaches a terminal state
  while :; do
    status=$(api_get "/pipelines/$pipeline_id" | json_field "['status']")
    case "$status" in success|failed|canceled|skipped) break ;; esac
    sleep "$POLL_INTERVAL"
  done

  # 3. download the lyris-exec job artifacts (results/) - kept even on failure
  job_id=$(api_get "/pipelines/$pipeline_id/jobs" \
    | python3 -c "import sys,json;print(next(j['id'] for j in json.load(sys.stdin) if j['name']=='lyris-exec'))")
  rm -rf results results.zip
  api_get "/jobs/$job_id/artifacts" > results.zip || true
  python3 -c "import zipfile;zipfile.ZipFile('results.zip').extractall()" 2>/dev/null || true

  # 4. infra failure -> retry then give up (90); otherwise hand off to the caller
  if [ "$status" = canceled ] || [ ! -f results/summary.txt ] \
     || grep -q '^INFRA_FAILURE' results/summary.txt 2>/dev/null; then
    if [ "$attempt" -le "$retries" ]; then echo "infra failure, retrying"; continue; fi
    echo "infra failure after $retries retries"; exit 90
  fi
  cat results/summary.txt
  exit 0
done
