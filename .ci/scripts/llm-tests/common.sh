#!/bin/bash
# Helpers shared between test_vllm.sh and test_sglang.sh.
# Sourced, not executed.

# Poll an HTTP endpoint until it returns 2xx or timeout. Used to gate test
# execution on prefill/decode/proxy servers being actually ready (model load
# can take 60–120s for TinyLlama).
wait_for_endpoint() {
    local url="$1" timeout="${2:-300}" what="${3:-endpoint}"
    local deadline=$(( $(date +%s) + timeout ))
    echo "Waiting up to ${timeout}s for ${what}: ${url}"
    while [[ $(date +%s) -lt ${deadline} ]]; do
        if curl -fsS -o /dev/null -m 5 "${url}"; then
            echo "READY: ${what}"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: ${what} did not become ready within ${timeout}s"
    return 1
}

# Dump every *.log in the given directory to stdout, banner-delimited so it's
# easy to find each server's output in a SLURM job log. Files inside the
# container are gone once the job ends, so without this you have no clue why
# prefill/decode failed to come up.
dump_logs() {
    local dir="$1"
    [[ -d "${dir}" ]] || return 0
    shopt -s nullglob
    local f
    for f in "${dir}"/*.log "${dir}"/*.out; do
        echo
        echo "===== ${f} ====="
        cat "${f}"
        echo "===== end ${f} ====="
    done
    shopt -u nullglob
}

# Print a one-shot snapshot of what the container sees for RDMA/UCX/GPU
# topology. Cheap, always-on; rendered once before the servers launch so any
# subsequent NIXL transport failure is debuggable from the Jenkins log alone.
# Every probe is best-effort: a missing tool or denied permission must not
# abort the test run.
dump_env_diagnostics() {
    echo "===== env diagnostics ====="
    echo "--- /dev/infiniband ---"
    ls -la /dev/infiniband 2>&1 || true
    # Some LLM containers ship UCX as a library only, without the CLI tools.
    # Fall back to /sys/class/infiniband so we still get HCA names + ports.
    if command -v ibv_devinfo >/dev/null 2>&1; then
        echo "--- ibv_devinfo -l ---"
        ibv_devinfo -l 2>&1 || true
    else
        echo "--- /sys/class/infiniband (ibv_devinfo not installed) ---"
        for d in /sys/class/infiniband/*; do
            [[ -e "${d}" ]] || continue
            hca=$(basename "${d}")
            state=$(cat "${d}/ports/1/state" 2>/dev/null || echo "?")
            ltype=$(cat "${d}/ports/1/link_layer" 2>/dev/null || echo "?")
            rate=$(cat "${d}/ports/1/rate" 2>/dev/null || echo "?")
            echo "  ${hca}: state=${state} link=${ltype} rate=${rate}"
        done
    fi
    if command -v ucx_info >/dev/null 2>&1; then
        echo "--- ucx_info -d (head) ---"
        ucx_info -d 2>&1 | head -40 || true
    else
        echo "--- ucx_info not installed ---"
    fi
    echo "--- nvidia-smi topo -m ---"
    nvidia-smi topo -m 2>&1 || true
    echo "--- nvidia-smi nvlink --status (head) ---"
    nvidia-smi nvlink --status 2>&1 | head -40 || true
    echo "===== end env diagnostics ====="
}

# Kill a process tree by PID, swallowing failures (best-effort cleanup).
kill_tree() {
    local pid="$1"
    [[ -z "${pid}" ]] && return 0
    if kill -0 "${pid}" 2>/dev/null; then
        echo "Stopping pid ${pid} (and children)"
        pkill -P "${pid}" 2>/dev/null || true
        kill "${pid}" 2>/dev/null || true
        sleep 2
        kill -9 "${pid}" 2>/dev/null || true
    fi
}

# Send the standard smoke-test prompt and verify the response is non-empty.
# Retries on curl failure / 5xx to absorb router worker-activation races
# (sglang_router's /v1/models returns 200 as soon as ONE worker registers,
# but /v1/completions needs both prefill+decode healthy).
# Args: <completions-url> <model> [timeout-seconds]
smoke_request() {
    local url="$1" model="$2" timeout="${3:-120}"
    local body resp deadline
    body=$(cat <<EOF
{"model":"${model}","prompt":"San Francisco is a","max_tokens":10,"temperature":0}
EOF
)
    echo "POST ${url} with body: ${body}"
    deadline=$(( $(date +%s) + timeout ))
    while [[ $(date +%s) -lt ${deadline} ]]; do
        # --max-time bounds each attempt: 10 tokens on TinyLlama should
        # finish in seconds; anything past 30s is a server-side hang (e.g.
        # NIXL KV transfer wedged) and we want to fail-and-dump-logs
        # rather than block the CI job for hours.
        if resp=$(curl -fsS --max-time 30 -H 'Content-Type: application/json' -d "${body}" "${url}" 2>&1); then
            echo "Response: ${resp}"
            # Bare-minimum sanity check: response must contain a "text" field
            # with something non-empty in it.
            if echo "${resp}" | python3 -c "import sys, json; r=json.load(sys.stdin); t=r['choices'][0]['text']; sys.exit(0 if t else 1)"; then
                echo "SMOKE OK"
                return 0
            fi
            echo "smoke retry: empty or malformed completion"
        else
            echo "smoke retry: curl failed: ${resp}"
        fi
        sleep 3
    done
    echo "SMOKE FAILED: did not succeed within ${timeout}s"
    return 1
}

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
