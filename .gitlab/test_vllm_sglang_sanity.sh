#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 1-prefill / 1-decode NIXL KV-transfer sanity for vLLM or SGLang.
#
# Runs INSIDE the per-PR framework image on a GPU node. The image carries the freshly
# built NIXL wheels; the test model (NIXL_SANITY_MODEL_ID) is prefetched at runtime rather
# than baked into the image. Brings up a prefill instance and a decode instance wired
# through the framework's NIXL disaggregation path, sends one request, and asserts a
# non-empty completion came back through the transfer.
#
# Usage: test_vllm_sglang_sanity.sh <vllm|sglang>
set -euo pipefail

FRAMEWORK="${1:?usage: test_vllm_sglang_sanity.sh <vllm|sglang>}"
# Weights are NOT baked into the base image (keeps it small / the pull fast); vllm/sglang
# resolve the model by its HuggingFace repo id from the cache prefetched below.
MODEL="${NIXL_SANITY_MODEL_ID:?NIXL_SANITY_MODEL_ID not set (set by the base image)}"

# Multiple sanity jobs can share a SLURM node (8 GPUs, 2 per job) and the srun/enroot
# containers use HOST networking, so fixed ports would collide across concurrent jobs.
# Reuse the CI helper (from the workspace copied into the image) to hand out free TCP
# ports that roll over. common.sh references unset env vars (CI_CONCURRENT_ID/
# EXECUTOR_NUMBER) at source time, so disable nounset while sourcing, then restore it.
#
# Neither CI_CONCURRENT_ID nor EXECUTOR_NUMBER is set inside the enroot container, so
# common.sh would fall back to a random concurrency id over its FULL port space (base
# ports up to ~65500). SGLang derives internal ports from the server port (e.g.
# SGLANG_GRPC_PORT = port + 10000), which overflows 65535 for base ports > 55535 and
# kills the server on startup. Pin the id so every handed-out port stays <= 55500.
CI_CONCURRENT_ID="${CI_CONCURRENT_ID:-$((RANDOM % 45))}"
export CI_CONCURRENT_ID
set +u
# shellcheck disable=SC1091
. /workspace/nixl/.ci/scripts/common.sh
set -u
PREFILL_PORT="${PREFILL_PORT:-$(get_next_tcp_port)}"
DECODE_PORT="${DECODE_PORT:-$(get_next_tcp_port)}"
PROXY_PORT="${PROXY_PORT:-$(get_next_tcp_port)}"
PROMPT="${PROMPT:-San Francisco is a}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-300}"
# Fraction of GPU memory each server may claim. The frameworks default to ~0.9, which
# demands nearly the whole GPU to be free at startup; on the shared GB200 CI nodes other
# processes may already hold GPU memory, and the sanity model needs only a small slice
# anyway (0.3 of a 186 GiB GB200 is ~56 GiB, ample for 8B weights + KV cache).
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.3}"
PROXY_TIMEOUT="${PROXY_TIMEOUT:-120}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"

# SGLang gsm8k accuracy test (runs through the PD router). Dataset is baked into the
# image (.ci/dockerfiles/Dockerfile.sglang-base) so the test never downloads at runtime.
# GSM8K_MIN_ACCURACY is a correctness floor: Qwen3-8B clears 0.9 on a healthy NIXL
# transfer (0.8 leaves headroom for run-to-run variance), while a broken transfer
# collapses toward 0.
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-/opt/gsm8k/test.jsonl}"
GSM8K_NUM_QUESTIONS="${GSM8K_NUM_QUESTIONS:-200}"
GSM8K_MIN_ACCURACY="${GSM8K_MIN_ACCURACY:-0.8}"

log() { echo "[sanity:${FRAMEWORK}] $*"; }

# Per-request curl caps so a stalled/half-open connection can't hang past the loop's
# timeout budget (each loop iteration is accounted as ~1s).
CURL_OPTS=(--connect-timeout 5 --max-time 30)

wait_for() {  # wait_for <url> <timeout_s>
  local url="$1" t="${2:-180}" i=0
  until curl -sf "${CURL_OPTS[@]}" "$url" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge "$t" ]; then
      log "timeout (${t}s) waiting for ${url}"
      return 1
    fi
    sleep 1
  done
  log "ready: ${url}"
}

# Servers are started with setsid so each gets its own process group; cleanup signals the
# whole group so forked engine/worker subprocesses can't survive and keep holding GPUs on
# the shared SLURM node.
pids=()
cleanup() {
  for p in "${pids[@]:-}"; do
    kill -TERM -- "-$p" 2>/dev/null || true
  done
}
trap cleanup EXIT

log "model: ${MODEL}"
nvidia-smi -L
python3 -c "import nixl; from importlib.metadata import version; print('nixl', version('nixl'))"

# Prefetch the model weights before launching the servers. The weights are not baked into
# the image, and vllm/sglang otherwise discover the cache miss deep in their own startup
# path (burning the health-check timeout); an explicit download fails fast and clearly.
# huggingface_hub ships in both framework images, so no extra install is needed.
log "prefetching ${MODEL}"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"

if [ "$FRAMEWORK" = "vllm" ]; then
  python3 -c "from importlib.metadata import version; print('vllm', version('vllm'))"

  # Prefill and decode are co-located on one node, so each needs its own NIXL
  # side-channel handshake port (default 5600 for both -> "Address already in use"),
  # and they must be host-unique so concurrent jobs on the same node don't collide.
  PREFILL_SIDE_PORT="$(get_next_tcp_port)"
  DECODE_SIDE_PORT="$(get_next_tcp_port)"
  setsid env CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_PORT" \
    vllm serve "$MODEL" --port "$PREFILL_PORT" \
    --enforce-eager --gpu-memory-utilization "$GPU_MEM_FRACTION" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' &
  pids+=($!)

  setsid env CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_PORT" \
    vllm serve "$MODEL" --port "$DECODE_PORT" \
    --enforce-eager --gpu-memory-utilization "$GPU_MEM_FRACTION" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' &
  pids+=($!)

  wait_for "http://localhost:${PREFILL_PORT}/health" "$SERVER_TIMEOUT"
  wait_for "http://localhost:${DECODE_PORT}/health" "$SERVER_TIMEOUT"

  # toy_proxy_server.py was baked into the image at a fixed path by contrib/Dockerfile.vllm.
  # Its health endpoint is /healthcheck (polling /health only 404s until the timeout).
  setsid python3 /usr/local/bin/toy_proxy_server.py --port "$PROXY_PORT" \
    --prefiller-ports "$PREFILL_PORT" --decoder-ports "$DECODE_PORT" &
  pids+=($!)
  wait_for "http://localhost:${PROXY_PORT}/healthcheck" "$PROXY_TIMEOUT"
  ENDPOINT="http://localhost:${PROXY_PORT}/v1/completions"

elif [ "$FRAMEWORK" = "sglang" ]; then
  python3 -c "from importlib.metadata import version; print('sglang', version('sglang'))"

  # The prefill binds a bootstrap server on --disaggregation-bootstrap-port (default
  # 8998); make it host-unique so concurrent jobs on the same node don't collide. Both
  # sides are configured with the same port (decode connects to the prefill's bootstrap).
  BOOTSTRAP_PORT="$(get_next_tcp_port)"
  setsid env CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path "$MODEL" \
    --mem-fraction-static "$GPU_MEM_FRACTION" \
    --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --trust-remote-code --host 0.0.0.0 --port "$PREFILL_PORT" &
  pids+=($!)

  setsid env CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path "$MODEL" \
    --mem-fraction-static "$GPU_MEM_FRACTION" \
    --disaggregation-mode decode --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --trust-remote-code --host 0.0.0.0 --port "$DECODE_PORT" &
  pids+=($!)

  wait_for "http://localhost:${PREFILL_PORT}/health" "$SERVER_TIMEOUT"
  wait_for "http://localhost:${DECODE_PORT}/health" "$SERVER_TIMEOUT"

  # sglang-router fronts the prefill/decode pair (replaces the removed in-core mini_lb).
  # Installed into the sglang weights-base image (contrib/Dockerfile.sglang-base).
  setsid python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill "http://localhost:${PREFILL_PORT}" \
    --decode "http://localhost:${DECODE_PORT}" \
    --host 0.0.0.0 --port "$PROXY_PORT" &
  pids+=($!)
  wait_for "http://localhost:${PROXY_PORT}/health" "$PROXY_TIMEOUT" || true
  ENDPOINT="http://localhost:${PROXY_PORT}/v1/completions"

else
  log "unknown framework: ${FRAMEWORK}"
  exit 2
fi

# The proxy/router may report /health ready before its workers are registered
# (the sglang-router initializes workers in the background and 503s with
# "No prefill workers available" until then). Wait until it actually serves a
# request before running the real check.
log "waiting for the proxy to accept requests"
i=0
until curl -sf "${CURL_OPTS[@]}" "$ENDPOINT" -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}" >/dev/null; do
  i=$((i + 1))
  if [ "$i" -ge "$REQUEST_TIMEOUT" ]; then
    log "timeout (${REQUEST_TIMEOUT}s) waiting for the proxy to serve a request"
    exit 1
  fi
  sleep 1
done
log "proxy is serving"

if [ "$FRAMEWORK" = "sglang" ]; then
  # Correctness check: run gsm8k through the PD router and assert accuracy. A healthy
  # NIXL KV transfer scores at the model baseline; a broken one collapses toward 0.
  log "running gsm8k accuracy (${GSM8K_NUM_QUESTIONS} questions) through the router"
  OUT="$(python3 -m sglang.test.few_shot_gsm8k \
    --num-questions "$GSM8K_NUM_QUESTIONS" \
    --host 127.0.0.1 --port "$PROXY_PORT" \
    --data-path "$GSM8K_DATA_PATH")"
  echo "$OUT"
  ACC="$(printf '%s\n' "$OUT" | sed -n 's/^Accuracy:[[:space:]]*\([0-9.]*\).*/\1/p' | tail -1)"
  [ -n "$ACC" ] || { log "could not parse gsm8k accuracy from output"; exit 1; }
  ACC="$ACC" THR="$GSM8K_MIN_ACCURACY" python3 - <<'PY'
import os
import sys

acc = float(os.environ["ACC"])
thr = float(os.environ["THR"])
print(f"[sanity] gsm8k accuracy={acc:.3f} threshold={thr:.3f}")
if acc < thr:
    print(f"[sanity] FAIL: accuracy {acc:.3f} below threshold {thr:.3f}")
    sys.exit(1)
print("[sanity] transfer OK")
PY

else
  # vLLM: assert a non-empty completion for our prompt through toy_proxy.
  log "sending completion request through the disaggregation proxy"
  RESP="$(curl -sf "${CURL_OPTS[@]}" "$ENDPOINT" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":16,\"temperature\":0}")"
  echo "$RESP"
  RESP="$RESP" python3 - <<'PY'
import json
import os

resp = json.loads(os.environ["RESP"])
text = resp["choices"][0]["text"]
assert text and text.strip(), f"empty completion: {resp!r}"
print("[sanity] transfer OK, completion:", repr(text))
PY
fi

log "PASS"
