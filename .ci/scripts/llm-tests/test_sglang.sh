#!/bin/bash
# Test the sglang-nixl (and sglang-cu13-nixl) container: prefill + decode +
# sglang_router, then run the requested test phase against the router.
#
# Usage: test_sglang.sh <phase>
#   phase = smoke | perf | accuracy
#
# Runs inside the sglang LLM container on a SLURM-allocated 2-GPU node.

set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/common.sh"

PHASE="${1:-smoke}"

# The accuracy phase serves a stronger model (see ACCURACY_MODEL in common.sh)
# so gsm8k can clear the >0.9 threshold; the prefill/decode servers below are
# launched with whatever MODEL ends up being for this phase.
if [[ "${PHASE}" == "accuracy" ]]; then
    MODEL="${ACCURACY_MODEL}"
fi
echo "Using model: ${MODEL}"

PREFILL_PORT=30000
DECODE_PORT=30001
ROUTER_PORT=8000

PREFILL_PID=""
DECODE_PID=""
ROUTER_PID=""

cleanup() {
    local rc=$?
    set +e
    echo "=== cleanup ==="
    kill_tree "${ROUTER_PID}"
    kill_tree "${DECODE_PID}"
    kill_tree "${PREFILL_PID}"
    if (( rc != 0 )); then
        echo "=== server logs (exit ${rc}) ==="
        dump_logs /tmp/sglang-logs
    fi
}
trap cleanup EXIT

mkdir -p /tmp/sglang-logs

dump_env_diagnostics

# --sampling-defaults openai forces deterministic defaults instead of the
# newer sglang default 'model', which inherits TinyLlama-Chat's stochastic
# generation_config (temperature/top_p > 0) and collapses gsm8k accuracy
# under the harness's parallel-request load.
echo "=== launching sglang prefill on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 \
python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PREFILL_PORT}" \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --sampling-defaults openai \
    --tp 1 \
    >/tmp/sglang-logs/prefill.log 2>&1 &
PREFILL_PID=$!
echo "prefill pid=${PREFILL_PID}"

echo "=== launching sglang decode on GPU 1 ==="
CUDA_VISIBLE_DEVICES=1 \
python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${DECODE_PORT}" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --sampling-defaults openai \
    --tp 1 \
    >/tmp/sglang-logs/decode.log 2>&1 &
DECODE_PID=$!
echo "decode pid=${DECODE_PID}"

# SGLang exposes /v1/models once the server is fully up.
wait_for_endpoint "http://127.0.0.1:${PREFILL_PORT}/health" 300 "sglang prefill"
wait_for_endpoint "http://127.0.0.1:${DECODE_PORT}/health"  300 "sglang decode"

echo "=== launching sglang router on :${ROUTER_PORT} ==="
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://127.0.0.1:${PREFILL_PORT}" \
    --decode  "http://127.0.0.1:${DECODE_PORT}" \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}" \
    >/tmp/sglang-logs/router.log 2>&1 &
ROUTER_PID=$!
echo "router pid=${ROUTER_PID}"

wait_for_endpoint "http://127.0.0.1:${ROUTER_PORT}/v1/models" 120 "sglang router"

case "${PHASE}" in
    smoke)
        smoke_request "http://127.0.0.1:${ROUTER_PORT}/v1/completions" "${MODEL}"
        ;;

    perf)
        echo "=== sglang.bench_serving ==="
        python3 -m sglang.bench_serving \
            --backend sglang \
            --base-url "http://127.0.0.1:${ROUTER_PORT}" \
            --model "${MODEL}" \
            --dataset-name random \
            --num-prompts 100 \
            --max-concurrency 10 \
            --random-input 1024 \
            --random-output 1024 \
            --warmup-requests 100 \
            --output-details \
            --random-range-ratio 1
        ;;

    accuracy)
        echo "=== sglang gsm8k (200 examples) on ${MODEL} ==="
        # Run gsm8k and capture stdout; parse the "Accuracy:" line and fail
        # if it's below the threshold. Qwen3-8B comfortably exceeds 0.9 on
        # gsm8k, so 0.9 is the gate with headroom for run-to-run variance.
        OUT=/tmp/sglang-logs/gsm8k.out
        # --max-tokens 2048: Qwen3-8B has a 32k+ context, so the 5-shot prompt
        # plus a full chain-of-thought answer fits easily; 2048 leaves ample
        # room for the CoT to finish without truncating the final answer.
        python3 -m sglang.test.run_eval \
            --eval-name gsm8k \
            --num-examples 200 \
            --max-tokens 2048 \
            --host 127.0.0.1 \
            --port "${ROUTER_PORT}" \
            | tee "${OUT}"
        # run_eval prints "Score: 0.935" (older builds printed "Accuracy:");
        # accept either so the gate survives the harness's output rename.
        ACC=$(grep -E '^(Score|Accuracy):' "${OUT}" | head -1 | awk '{print $2}')
        if [[ -z "${ACC}" ]]; then
            echo "ERROR: could not parse Score/Accuracy from gsm8k output"
            exit 1
        fi
        # Numeric compare in awk; bash doesn't do floats.
        if awk -v a="${ACC}" 'BEGIN { exit !(a+0 >= 0.9) }'; then
            echo "ACCURACY OK: ${ACC} (>= 0.9)"
        else
            echo "ACCURACY FAILED: ${ACC} (< 0.9)"
            exit 1
        fi
        ;;

    *)
        echo "ERROR: unknown phase '${PHASE}'"
        exit 1
        ;;
esac

echo "=== ${PHASE} OK ==="
