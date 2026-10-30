#!/usr/bin/env bash
#
# Trigger the generic `slurm-exec` pipeline, wait for it, and fetch its results.
#
# Usage:
#   TRIGGER_TOKEN=... GITLAB_TOKEN=... \
#   trigger_and_wait.sh --ref <branch> --partition <partition> \
#                       --image <pyxis-image> --test-cmds-file <file> \
#                       [--cluster <runner-tag>] [--allow-fail "<names>"] [--retries N]
#
# --cluster selects the target pre-cluster by GitLab runner tag (default: lyris).
#
# Downloads the job artifacts into ./results/ (per-test <name>.rc / <name>.log,
# summary.txt, junit.xml). Exit code: 0 = pipeline ran (read ./results/ for
# per-test pass/fail), 90 = infra failure (allocation/runner/queue) after
# retries. Per-test gating is left to the caller (each test has its own .rc).
set -uo pipefail

PROJECT_API="https://gitlab-master.nvidia.com/api/v4/projects/231686"
PROJECT_WEB="https://gitlab-master.nvidia.com/nbu-swx/ai/precluster-poc"
POLL_INTERVAL=30

# --- arguments ---
ref=""; partition=""; image=""; test_cmds_file=""; allow_fail=""; retries=2
cluster="lyris"
while [ $# -gt 0 ]; do
  case "$1" in
    --ref)            ref=$2;            shift 2 ;;
    --cluster)        cluster=$2;        shift 2 ;;
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
  trigger_resp=$(curl -sS -X POST \
    -F "token=$TRIGGER_TOKEN" -F "ref=$ref" \
    -F "variables[CLUSTER]=$cluster" \
    -F "variables[PARTITION]=$partition" \
    -F "variables[IMAGE]=$image" \
    -F "variables[TEST_CMDS]=$test_cmds_b64" \
    -F "variables[ALLOW_FAIL]=$allow_fail" \
    "$PROJECT_API/trigger/pipeline")
  pipeline_id=$(printf '%s' "$trigger_resp" | json_field "['id']" 2>/dev/null)
  case "$pipeline_id" in
    ''|*[!0-9]*)
      echo "ERROR: pipeline trigger failed (bad token/ref?): $trigger_resp" >&2
      exit 2 ;;
  esac
  echo "pipeline=$pipeline_id attempt=$attempt"
  echo "GitLab pipeline: $PROJECT_WEB/-/pipelines/$pipeline_id"

  # 2. poll until it reaches a terminal state
  while :; do
    status=$(api_get "/pipelines/$pipeline_id" | json_field "['status']")
    case "$status" in success|failed|canceled|skipped) break ;; esac
    sleep "$POLL_INTERVAL"
  done

  # 3. download the slurm-exec job artifacts (results/) - kept even on failure
  job_id=$(api_get "/pipelines/$pipeline_id/jobs" \
    | python3 -c "import sys,json;print(next(j['id'] for j in json.load(sys.stdin) if j['name']=='slurm-exec'))")
  echo "GitLab slurm-exec job (full logs + artifacts): $PROJECT_WEB/-/jobs/$job_id"
  rm -rf results results.zip
  api_get "/jobs/$job_id/artifacts" > results.zip || true
  python3 -c "import zipfile;zipfile.ZipFile('results.zip').extractall()" 2>/dev/null || true

  # 4. classify: config error -> fail fast (retrying won't help); infra failure
  # -> retry then give up (90); otherwise hand off to the caller
  if grep -q '^CONFIG_ERROR' results/summary.txt 2>/dev/null; then
    cat results/summary.txt >&2
    echo "configuration error - not retrying" >&2
    exit 2
  fi
  if [ "$status" = canceled ] || [ ! -f results/summary.txt ] \
     || grep -q '^INFRA_FAILURE' results/summary.txt 2>/dev/null; then
    if [ "$attempt" -le "$retries" ]; then echo "infra failure, retrying"; continue; fi
    echo "infra failure after $retries retries"; exit 90
  fi
  cat results/summary.txt
  exit 0
done
