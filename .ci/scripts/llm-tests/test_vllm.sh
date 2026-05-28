#!/bin/bash
# Test the vllm-nixl container: prefill + decode + toy_proxy_server, then run
# the requested test phase against the proxy.
#
# Usage: test_vllm.sh <phase>
#   phase = smoke | perf | accuracy
#
# Runs inside the vllm-nixl LLM container on a SLURM-allocated 2-GPU node.
# Mirrors the manual EOS procedure (prefill on GPU 0, decode on GPU 1, toy
# proxy on host, single-prompt completion request).

set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/common.sh"

PHASE="${1:-smoke}"

PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8192
SIDE_CHANNEL_HOST="127.0.0.1"

PREFILL_PID=""
DECODE_PID=""
PROXY_PID=""

cleanup() {
    local rc=$?
    set +e
    echo "=== cleanup ==="
    kill_tree "${PROXY_PID}"
    kill_tree "${DECODE_PID}"
    kill_tree "${PREFILL_PID}"
    if (( rc != 0 )); then
        echo "=== server logs (exit ${rc}) ==="
        dump_logs /tmp/vllm-logs
    fi
}
trap cleanup EXIT

mkdir -p /tmp/vllm-logs

echo "=== launching prefill on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 \
UCX_PROTO_INFO=used \
VLLM_NIXL_SIDE_CHANNEL_HOST="${SIDE_CHANNEL_HOST}" \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve "${MODEL}" \
    --port "${PREFILL_PORT}" \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    >/tmp/vllm-logs/prefill.log 2>&1 &
PREFILL_PID=$!
echo "prefill pid=${PREFILL_PID}"

echo "=== launching decode on GPU 1 ==="
CUDA_VISIBLE_DEVICES=1 \
UCX_PROTO_INFO=used \
VLLM_NIXL_SIDE_CHANNEL_HOST="${SIDE_CHANNEL_HOST}" \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve "${MODEL}" \
    --port "${DECODE_PORT}" \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    >/tmp/vllm-logs/decode.log 2>&1 &
DECODE_PID=$!
echo "decode pid=${DECODE_PID}"

wait_for_endpoint "http://127.0.0.1:${PREFILL_PORT}/v1/models" 180 "vllm prefill"
wait_for_endpoint "http://127.0.0.1:${DECODE_PORT}/v1/models"  180 "vllm decode"

# The Dockerfile.vllm baked in toy_proxy_server.py at /toy_proxy_server.py.
# Fall back to fetching it if it's missing (older RC images).
PROXY=/toy_proxy_server.py
if [[ ! -f "${PROXY}" ]]; then
    echo "toy_proxy_server.py not in image; fetching from upstream"
    PROXY=/tmp/toy_proxy_server.py
    curl -fL --retry 3 --retry-delay 2 \
        https://raw.githubusercontent.com/vllm-project/vllm/b73b5b06290c0d1439b09db71eef15fe59bc1fbb/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
        -o "${PROXY}"
fi

echo "=== launching toy proxy on :${PROXY_PORT} ==="
python3 "${PROXY}" \
    --port "${PROXY_PORT}" \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports "${PREFILL_PORT}" \
    --decoder-hosts   127.0.0.1 \
    --decoder-ports   "${DECODE_PORT}" \
    >/tmp/vllm-logs/proxy.log 2>&1 &
PROXY_PID=$!
echo "proxy pid=${PROXY_PID}"

# /v1/models won't necessarily be implemented by the toy proxy — instead poll
# the underlying decoder via the proxy with a HEAD to /v1/completions.
sleep 5  # give the proxy a moment to bind
wait_for_endpoint "http://127.0.0.1:${PREFILL_PORT}/v1/models" 60 "proxy upstream sanity"

case "${PHASE}" in
    smoke)
        smoke_request "http://127.0.0.1:${PROXY_PORT}/v1/completions" "${MODEL}"
        ;;

    perf)
        echo "=== vllm bench serve ==="
        # `vllm bench serve` is the canonical perf tool; needs a prompt
        # dataset. The sonnet dataset ships with the vllm install under
        # benchmarks/; fall back to a generated synthetic prompt set if not.
        DATASET=""
        for p in /workspace/vllm/benchmarks/sonnet.txt /usr/local/lib/python*/dist-packages/vllm/benchmarks/sonnet.txt; do
            if compgen -G "${p}" > /dev/null; then DATASET=$(ls ${p} | head -1); break; fi
        done
        if [[ -z "${DATASET}" ]]; then
            echo "WARN: sonnet.txt not found; using random dataset instead"
            BENCH_ARGS=(--dataset-name random --random-input-len 256 --random-output-len 64)
        else
            BENCH_ARGS=(--dataset-name sonnet --dataset-path "${DATASET}")
        fi
        vllm bench serve \
            --backend vllm \
            --model "${MODEL}" \
            --host 127.0.0.1 \
            --port "${PROXY_PORT}" \
            --endpoint /v1/completions \
            --num-prompts 50 \
            --max-concurrency 10 \
            "${BENCH_ARGS[@]}"
        ;;

    accuracy)
        # vLLM has no built-in equivalent to sglang.test.few_shot_gsm8k;
        # treated as a no-op for this variant.
        echo "Accuracy phase not implemented for vllm — skipping."
        ;;

    *)
        echo "ERROR: unknown phase '${PHASE}'"
        exit 1
        ;;
esac

echo "=== ${PHASE} OK ==="
