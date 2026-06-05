<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang PD Staged VRAM Integration Plan

This document is the runbook for validating NIXL UCX staged VRAM with SGLang PD
disaggregation on consumer GPUs.

It covers two paths:

```text
single-node no-P2P:
  source GPU -> source shared pinned host staging -> target GPU

multi-node no-GDR:
  source GPU -> source pinned host staging -> RDMA -> target pinned host staging -> target GPU
```

The goal is to keep SGLang on the standard NIXL API and the `UCX` backend. SGLang should continue
to register `VRAM_SEG`, submit WRITE transfers, and wait for NIXL completion or notification.

## Hard Rules

- Do not change `ip route`, default routes, policy routes, or host networking routes during
  validation.
- Keep all staged modes opt-in.
- Keep a fast rollback: unset the staged env vars or switch SGLang back to its previous transfer
  backend.
- Treat NIXL completion as valid only after the target GPU memory has been updated.
- Validate with NIXL transfer-only timing before interpreting SGLang TTFT or TPOT.

## Current Backend Boundaries

Supported:

- `NIXL_WRITE`.
- `VRAM_SEG` source and target descriptors.
- Multi-descriptor WRITE requests.
- Source D2H prefetch.
- Multi-node staged host RDMA.
- Single-node source-owned shared pinned staging.
- Per-chunk fallback from local shared staging to UCX staged host transfer on ACK error.

Not supported yet:

- `NIXL_READ`.
- Explicit application CUDA stream integration.
- Target-owned shared staging.
- Full stale shared-file cleanup after process crash.

Before SGLang integration, confirm the active SGLang NIXL path does not require READ. If READ is
observed, stop and implement READ as a separate phase.

## Common Environment

Use these for both single-node and multi-node validation:

```text
NIXL_PLUGIN_DIR=/path/to/nixl/build/src/plugins/ucx
LD_LIBRARY_PATH=/path/to/nixl/build/src/core:/path/to/nixl/build/src/infra:/path/to/nixl/build/src/utils/common:/path/to/nixl/build/src/utils/serdes:/path/to/nixl/build/src/utils/stream:/path/to/ucx/lib:/usr/local/cuda/lib64
NIXL_UCX_STAGING_CHUNK_SIZE=16777216
NIXL_UCX_STAGING_SLOTS=4
NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW=32
NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH=1
NIXL_UCX_STAGING_FORCE_PROGRESS_THREAD=1
NIXL_UCX_LOCAL_STAGING_FALLBACK=1
NIXL_UCX_STAGING_TARGET_H2D_WORKER=0
NIXL_UCX_STAGING_BATCH_FLUSH=0
```

Enable profile for smoke and shadow runs:

```text
NIXL_UCX_STAGING_PROFILE=1
NIXL_LOG_LEVEL=INFO
UCX_LOG_LEVEL=error
```

Disable profile for final throughput runs if log volume becomes intrusive.

## Single-Node Local PD

### Purpose

Validate local no-P2P SGLang PD on two GPUs in the same host. The payload path should not use UCX
RMA or network data movement.

Expected NIXL data path:

```text
prefill GPU -> source-owned shared pinned staging -> decode GPU
```

### Required Deployment Setup

The prefill and decode processes must see the same staging directory. Use one of:

```text
--ipc=host
```

or bind mount the same directory into both containers:

```text
/dev/shm/nixl
```

If different containers cannot share `/dev/shm`, use a bind-mounted directory, for example:

```text
/tmp/nixl-local-staging
```

and set:

```text
NIXL_UCX_LOCAL_STAGING_SHM_DIR=/tmp/nixl-local-staging
```

### Single-Node NIXL Env

```text
NIXL_UCX_VRAM_LOCAL_STAGING=1
NIXL_UCX_LOCAL_STAGING_SHM_DIR=/dev/shm/nixl
NIXL_UCX_LOCAL_STAGING_FALLBACK=1
```

`NIXL_UCX_VRAM_LOCAL_STAGING=1` automatically enables UCX VRAM staging if `NIXL_UCX_VRAM_STAGING`
was not set.

For container deployments with different hostnames, set the same explicit host id in both
processes:

```text
NIXL_UCX_LOCAL_STAGING_HOST_ID=<same-physical-host-id>
```

### Single-Node Preflight

Run these before SGLang:

```text
nvidia-smi
nvidia-smi topo -m
```

Confirm selected GPUs are idle enough for validation and whether P2P is unavailable:

```text
cudaDeviceCanAccessPeer(source_gpu, target_gpu) = 0
```

Run the local probe:

```text
local_vram_staging_probe \
  --mode single \
  --source-gpu <prefill_gpu> \
  --target-gpu <decode_gpu> \
  --bytes 64M \
  --iters 2 \
  --path /dev/shm/nixl-local-vram-staging-probe.bin
```

Expected:

```text
verification=passed
d2h_gib_per_sec ~= 24 GiB/s on RTX 4090 PCIe
h2d_gib_per_sec ~= 25 GiB/s on RTX 4090 PCIe
```

Then run the NIXL local staged smoke before SGLang:

```text
staged_vram_write_smoke \
  --mode target \
  --ip 127.0.0.1 \
  --port <port> \
  --bytes 16777216 \
  --concurrency 1 \
  --iters 1 \
  --descriptors 1 \
  --chunk-size 16777216 \
  --slots 4 \
  --staging 1 \
  --prepped 1 \
  --skip-desc-merge 1 \
  --poll-sleep-us 0 \
  --timing 1
```

Run initiator against the same port with the same parameters.

Expected initiator profile:

```text
local_shared_chunks > 0
local_shared_fallbacks = 0
rdma_write_posted = 0
flush_posted = 0
ack_received = chunks
```

Expected target:

```text
Target verification passed
attach_failures = 0
```

### Single-Node SGLang Smoke

Start a minimal PD run with one prefill process and one decode process on the selected local GPUs.
Use low request rate first.

Acceptance:

```text
NIXL staged profile appears in SGLang logs
local_shared_chunks > 0
rdma_write_posted = 0
flush_posted = 0
local_shared_fallbacks = 0 in normal path
TTFT/TPOT are stable
no target KV correctness errors
```

If `local_shared_chunks = 0`, check:

- both processes have `NIXL_UCX_VRAM_LOCAL_STAGING=1`;
- both processes see the same `NIXL_UCX_LOCAL_STAGING_SHM_DIR`;
- host id matches, or set `NIXL_UCX_LOCAL_STAGING_HOST_ID`;
- SGLang is actually selecting the NIXL UCX backend.

If `local_shared_fallbacks > 0`, check:

- target can open the source shared path;
- staging dir is shared across containers;
- `cudaHostRegister` succeeds on the target mapping;
- path is under `NIXL_UCX_LOCAL_STAGING_SHM_DIR`;
- no stale source process removed the shared file too early.

### Single-Node SGLang Validation Result

Validated on `sglang-rdma-0-41` with `Qwen3-8B-FP8`:

```text
prefill GPU: 4
decode GPU: 5
prefill: http://127.0.0.1:31100
decode:  http://127.0.0.1:31200
router:  http://127.0.0.1:31000
model:   /opt/users/models/Qwen3-8B-FP8
NIXL:    1.2.0 from /tmp/nixl-staged-src/build-staged-ucxhost
```

SGLang-side staging was explicitly disabled; only NIXL UCX VRAM staging was enabled:

```text
SGLANG_DISAGG_STAGING_BUFFER=0
SGLANG_MOONCAKE_HOST_STAGING=0
NIXL_UCX_VRAM_STAGING=1
NIXL_UCX_VRAM_LOCAL_STAGING=1
```

Do not force `UCX_TLS=sm,self` for this single-node SGLang smoke. In this environment it caused
UCX intra-agent endpoint creation to fail with `Destination is unreachable` because no usable AM
transport was selected. Leaving UCX transport selection at its default allowed the NIXL UCX backend
to initialize while the actual KV payload still used the local shared staging fast path.

When testing from an uninstalled NIXL build tree against the SGLang venv wheel, `NIXL_PLUGIN_DIR`
alone is not sufficient. The installed `nixl_cu12` Python extension has an RPATH to its bundled
`libnixl.so`; it can discover the build-tree UCX plugin but may fail `createBackend("UCX")` with
`NIXL_ERR_NOT_FOUND`. For build-tree validation, either install the rebuilt wheel or preload the
matching build-tree core:

```text
NIXL_PLUGIN_DIR=/tmp/nixl-staged-src/build-staged-ucxhost/src/plugins/ucx
LD_LIBRARY_PATH=/tmp/nixl-staged-src/build-staged-ucxhost/src/utils/common:\
/tmp/nixl-staged-src/build-staged-ucxhost/src/utils/serdes:\
/tmp/ucx-v1.21-host-install/lib:$LD_LIBRARY_PATH
LD_PRELOAD=/tmp/nixl-staged-src/build-staged-ucxhost/src/core/libnixl.so:$LD_PRELOAD
```

Router request:

```text
POST /generate
prompt_tokens=5
max_new_tokens=8
elapsed_sec=0.637
```

Prefill profile:

```text
local_shared_chunks=72
local_shared_bytes=737280
local_shared_ack_errors=0
local_shared_fallbacks=0
rdma_write_posted=0
flush_posted=0
ready_sent=72
ack_received=72
```

Decode profile:

```text
ready_count=64
errors=0
attach_cache_misses=64
attach_failures=0
```

This confirms the SGLang local PD smoke used source-owned shared pinned staging, not RDMA/RMA.

Follow-up after the local shared region cookie and reply-endpoint control AM patch, validated on
`sglang-rdma-0-26` with prefill GPU 0, decode GPU 1, and router `127.0.0.1:31000`:

```text
3 repeated greedy requests for "Hello, my name is" returned identical output_ids.
generated-shared-prefix, concurrency=16, num_prompts=64: 64/64 successful.
No NIXL_ERR, AssertionError, local shared ACK error, fallback, target error, attach failure, or
source metadata validation failure was observed.
```

Representative profile with `NIXL_LOG_LEVEL=INFO`:

```text
prefill:
local_shared_chunks=72
local_shared_bytes=737280
local_shared_ack_errors=0
local_shared_fallbacks=0
rdma_write_posted=0
flush_posted=0
ready_sent=72
ack_received=72

decode:
ready_count=192
errors=0
attach_cache_misses=72
attach_cache_hits=120
attach_failures=0
```

Follow-up validation with the same setup and unmodified SGLang `nixl/conn.py`:

```text
3 repeated greedy requests for "Hello, my name is" returned identical output_ids.
5 different prompts, each repeated twice with temperature=0, all matched exactly.
No Foreign traffic, AssertionError, NIXL_ERR, fallback, ACK error, target H2D error, or attach failure
was observed in the P/D/router logs.
```

Representative prefill profile after repeated requests:

```text
local_shared_chunks=72
local_shared_ack_errors=0
local_shared_fallbacks=0
rdma_write_posted=0
flush_posted=0
ready_sent=72
ack_received=72
```

Representative decode profile:

```text
ready_count=896
errors=0
attach_failures=0
```

### Single-Node SGLang Benchmark Result

The first attempt with `--dataset-name random` was not usable in this container because
`bench_serving` tried to download a ShareGPT file from HuggingFace and the container had no route to
that endpoint. The pressure test therefore used the local `generated-shared-prefix` dataset.

Benchmark command shape:

```text
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:31000 \
  --model /opt/users/models/Qwen3-8B-FP8 \
  --tokenizer /opt/users/models/Qwen3-8B-FP8 \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 4 \
  --gsp-system-prompt-len 512 \
  --gsp-question-len 64 \
  --gsp-output-len 16 \
  --gsp-range-ratio 1.0 \
  --request-rate inf
```

Results:

| max concurrency | requests | successful | input tokens | output tokens | req/s | output tok/s | P99 TTFT | P99 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 64 | not accurate due to `--gsp-fast-prepare` | 1024 | 31.27 | 500.27 | 628.95 ms | 14.91 ms |
| 32 | 128 | 128 | 77539 | 2048 | 40.93 | 654.85 | 798.36 ms | 23.80 ms |
| 64 | 256 | 256 | 154929 | 4096 | 46.08 | 737.26 | 1539.82 ms | 40.43 ms |

Stability check after the concurrency-64 run:

```text
Foreign traffic: 0
AssertionError: 0
NIXL_ERR: 0
local_shared_ack_errors: 0
local_shared_fallbacks: 0
target H2D errors: 0
attach_failures: 0
```

Representative NIXL profile remained on the local shared path:

```text
rdma_write_posted=0
flush_posted=0
local_shared_fallbacks=0
local_shared_ack_errors=0
ack_received = chunks
```

Decode-side cumulative profile reached about 40.2 GiB of local shared transfer:

```text
ready_count=78784
bytes=40226840576
errors=0
attach_failures=0
```

Implementation note: SGLang's NIXL PD path registers decode metadata with prefill, then prefill
issues `WRITE`; decode does not necessarily call `add_remote_agent(prefill)`. The staged backend
therefore must not require a pre-existing target-to-source NIXL connection for internal completion
messages. `STAGED_SLOT_REQ`, `STAGED_WRITE_READY`, and `STAGED_LOCAL_WRITE_READY` are sent with
`UCP_AM_SEND_FLAG_REPLY`, and the target replies to `SLOT_GRANT` or `ACK` through the UCX
`reply_ep` when present. For local shared READY, if source metadata was not preloaded on target,
the backend uses a restricted READY-carried validation: the path must be under
`NIXL_UCX_LOCAL_STAGING_SHM_DIR`, contain the source agent, region id, and region cookie when
present, and match the expected chunk slot offset and size bounds.

## Multi-Node PD

### Purpose

Validate SGLang PD across two hosts when consumer GPUs cannot use GPUDirect RDMA. The payload should
use host-pinned staging and RDMA for the host-to-host stage.

Expected NIXL data path:

```text
prefill GPU -> source pinned staging -> RDMA -> decode pinned staging -> decode GPU
```

### Multi-Node NIXL Env

Use these on both hosts:

```text
NIXL_UCX_VRAM_STAGING=1
NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH=1
NIXL_UCX_STAGING_CHUNK_SIZE=16777216
NIXL_UCX_STAGING_SLOTS=4
NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW=32
NIXL_UCX_STAGING_FORCE_PROGRESS_THREAD=1
NIXL_UCX_LOCAL_STAGING_FALLBACK=1
NIXL_UCX_STAGING_TARGET_H2D_WORKER=0
NIXL_UCX_STAGING_BATCH_FLUSH=0
```

Recommended UCX transport:

```text
UCX_TLS=rc,ud,self
```

If restricting data-plane HCAs, pass the fixed UCX backend parameter:

```text
ucx_devices=mlx5_4,mlx5_5
```

Do not change host routes. It is acceptable for metadata or handshake traffic to use the ordinary
management NIC while the UCX data plane uses RDMA HCAs.

### Multi-Node Preflight

On both hosts:

```text
nvidia-smi
ibdev2netdev
ibstat
```

Record HCA counters before the run. Use the counter files appropriate for the active HCA, for
example:

```text
/sys/class/infiniband/<mlx5_dev>/ports/1/counters/port_xmit_data
/sys/class/infiniband/<mlx5_dev>/ports/1/counters/port_rcv_data
```

Run the NIXL remote staged smoke first.

Expected initiator profile:

```text
local_shared_chunks = 0
slot_req_sent > 0
slot_grant_success > 0
rdma_write_posted > 0
flush_posted > 0
ack_received = chunks
```

Expected network validation:

```text
selected mlx5 counters increase roughly by payload size
no evidence that payload fell back to TCP
```

If RDMA counters do not move:

- confirm `UCX_TLS`;
- confirm `ucx_devices`;
- inspect UCX logs at a higher log level only for a short diagnostic run;
- do not change `ip route`;
- verify the selected HCA is connected and reachable.

### Multi-Node SGLang Smoke

Start one small prefill/decode PD pair across hosts with low request rate.

Acceptance:

```text
SGLang uses NIXL UCX backend
NIXL profile shows RDMA staged path
rdma_write_posted > 0
flush_posted > 0
local_shared_chunks = 0
mlx5 counter delta matches payload
TTFT/TPOT stable
no target KV correctness errors
```

If transfer fails with unsupported operation:

- check whether SGLang submitted `NIXL_READ`;
- current staged path supports WRITE only.

If transfer fails with descriptor mismatch:

- capture descriptor count and lengths;
- current path supports multi-descriptor WRITE, but local/remote descriptor counts and sizes must
  match.

### Multi-Node SGLang Validation Result

Validated on June 4, 2026 with one GPU per host:

```text
prefill host: sglang-rdma-0-26
prefill GPU:  0
decode host:  sglang-rdma-0-41
decode GPU:   6
router host:  sglang-rdma-0-26
router:       http://127.0.0.1:32000
model:        /opt/users/models/Qwen3-8B-FP8
NIXL build:   /tmp/nixl-staged-src/build-staged-ucxhost
```

The SGLang package loaded NIXL from the staged build by preloading the matching core library and
using the staged UCX plugin directory:

```text
NIXL_PLUGIN_DIR=/tmp/nixl-staged-src/build-staged-ucxhost/src/plugins/ucx
LD_PRELOAD=/tmp/nixl-staged-src/build-staged-ucxhost/src/core/libnixl.so:${LD_PRELOAD:-}
LD_LIBRARY_PATH=/tmp/nixl-staged-src/build-staged-ucxhost/src/utils/common:/tmp/nixl-staged-src/build-staged-ucxhost/src/utils/serdes:/tmp/ucx-v1.21-host-install/lib:${LD_LIBRARY_PATH:-}
```

Remote staged mode was enabled and local shared staging was explicitly disabled:

```text
NIXL_LOG_LEVEL=INFO
NIXL_UCX_VRAM_STAGING=1
NIXL_UCX_VRAM_LOCAL_STAGING=0
NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH=1
NIXL_UCX_STAGING_CHUNK_SIZE=16777216
NIXL_UCX_STAGING_SLOTS=4
NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW=32
NIXL_UCX_STAGING_TARGET_H2D_WORKER=0
NIXL_UCX_STAGING_BATCH_FLUSH=0
NIXL_UCX_STAGING_FORCE_PROGRESS_THREAD=1
NIXL_UCX_STAGING_PROFILE=1
SGLANG_DISAGG_STAGING_BUFFER=0
SGLANG_MOONCAKE_HOST_STAGING=0
```

Router bootstrap note: when the router is on the prefill host, do not pass the prefill URL as
`http://127.0.0.1:<port>` for a multi-host run. The decode process receives the prefill bootstrap
address from the router and must be able to connect to it from the decode host. The working command
used the externally reachable prefill address:

```text
sglang-router launch --host 0.0.0.0 --port 32000 \
  --pd-disaggregation \
  --prefill http://10.159.0.26:32100 32150 \
  --decode http://10.159.0.41:32200 \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --model-path /opt/users/models/Qwen3-8B-FP8 \
  --tokenizer-path /opt/users/models/Qwen3-8B-FP8 \
  --backend sglang
```

Using `http://127.0.0.1:32100 32150` caused decode on `sglang-rdma-0-41` to try
`127.0.0.1:32150` and fail bootstrap with `Connection refused`.

Small deterministic request through the router succeeded:

```text
request:      "Hello, my name is"
max_new_tokens: 16
status:       200 OK
elapsed_sec:  0.565
output:       " Alex, and I am a 22-year-old student. I am currently"
```

SGLang benchmark command:

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:32000 \
  --model /opt/users/models/Qwen3-8B-FP8 \
  --tokenizer /opt/users/models/Qwen3-8B-FP8 \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 4 \
  --gsp-system-prompt-len 512 \
  --gsp-question-len 64 \
  --gsp-output-len 16 \
  --gsp-range-ratio 1.0 \
  --num-prompts 64 \
  --max-concurrency 16 \
  --request-rate inf
```

Observed benchmark result:

```text
Successful requests:                  64
Benchmark duration:                   6.22 s
Request throughput:                   10.29 req/s
Input token throughput:               6292.16 tok/s
Output token throughput:              164.63 tok/s
Total token throughput:               6456.79 tok/s
Mean E2E latency:                     1436.28 ms
Median E2E latency:                   1387.05 ms
P90 E2E latency:                      1815.34 ms
P99 E2E latency:                      2166.53 ms
Mean TTFT:                            959.37 ms
Median TTFT:                          912.80 ms
P99 TTFT:                             1682.48 ms
Mean TPOT:                            31.79 ms
Median TPOT:                          31.74 ms
P99 TPOT:                             32.67 ms
```

The staged profiles confirmed that the payload path was remote staged RDMA, not local shared
staging and not SGLang's own host-staging path:

```text
prefill initiator:
  local_shared_chunks = 0
  local_shared_bytes = 0
  local_shared_fallbacks = 0
  rdma_write_posted > 0
  flush_posted > 0
  ready_sent == ack_received

decode target:
  ready_count increased
  target staged bytes ~= 5.85 GB during the concurrent run
  h2d_avg_us ~= 24
```

No new `NIXL_ERR`, staged fallback, H2D failure, or decode handshake error was observed after the
router address was corrected. The only decode handshake failure in the log was from the earlier
`127.0.0.1` bootstrap misconfiguration.

Cleanup was verified after the run:

```text
sglang-rdma-0-26 GPU0: 0 MiB after stopping the test prefill/router
sglang-rdma-0-41 GPU6: 0 MiB after stopping the test decode
```

Existing unrelated GPU users on `sglang-rdma-0-41` GPU0/GPU4 were not modified.

### Multi-Node Baseline Comparison

Validated on June 4, 2026 with the same host/GPU placement and benchmark shape:

```text
prefill host/GPU: sglang-rdma-0-26 GPU0
decode host/GPU:  sglang-rdma-0-41 GPU6
model:            /opt/users/models/Qwen3-8B-FP8
dataset:          generated-shared-prefix
num_prompts:      64
max_concurrency:  16
output_len:       16
```

The raw installed SGLang/NIXL environment did not start when SGLang-side staging was disabled and
UCX transport was left unrestricted. Installed NIXL `1.1.0` attempted CUDA memory registration on an
IB memory domain and failed before reaching TCP fallback:

```text
UCX ERROR ibv_reg_mr(... cuda ...) failed: Bad address
nixlBackendError: NIXL_ERR_BACKEND
```

For a runnable original-NIXL baseline, force UCX to the expected TCP/cuda-copy fallback:

```text
UCX_TLS=tcp,cuda_copy,self
SGLANG_DISAGG_STAGING_BUFFER=0
SGLANG_MOONCAKE_HOST_STAGING=0
```

The optimized staged run used the build-tree NIXL `1.2.0` core/plugin and remote VRAM staging:

```text
NIXL_PLUGIN_DIR=/tmp/nixl-staged-src/build-staged-ucxhost/src/plugins/ucx
LD_PRELOAD=/tmp/nixl-staged-src/build-staged-ucxhost/src/core/libnixl.so:${LD_PRELOAD:-}
NIXL_UCX_VRAM_STAGING=1
NIXL_UCX_VRAM_LOCAL_STAGING=0
NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH=1
NIXL_UCX_STAGING_CHUNK_SIZE=16777216
NIXL_UCX_STAGING_SLOTS=4
NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW=32
```

Observed results:

| Metric | Original NIXL TCP fallback | Optimized NIXL staged RDMA | Change |
| --- | ---: | ---: | ---: |
| Successful requests | 64 | 64 | same |
| Benchmark duration | 14.68 s | 6.27 s | 2.34x faster |
| Request throughput | 4.36 req/s | 10.21 req/s | 2.34x |
| Input token throughput | 2666.70 tok/s | 6245.02 tok/s | 2.34x |
| Output token throughput | 69.77 tok/s | 163.40 tok/s | 2.34x |
| Total token throughput | 2736.48 tok/s | 6408.42 tok/s | 2.34x |
| Mean E2E latency | 3281.78 ms | 1423.25 ms | 56.6% lower |
| Median E2E latency | 3524.31 ms | 1390.22 ms | 60.6% lower |
| P99 E2E latency | 3963.75 ms | 2072.18 ms | 47.7% lower |
| Mean TTFT | 2800.29 ms | 939.36 ms | 66.5% lower |
| Median TTFT | 3040.94 ms | 911.14 ms | 70.0% lower |
| P99 TTFT | 3482.10 ms | 1593.85 ms | 54.2% lower |
| Mean TPOT | 32.10 ms | 32.26 ms | effectively unchanged |
| Median TPOT | 32.11 ms | 32.05 ms | effectively unchanged |
| P99 TPOT | 33.52 ms | 34.21 ms | effectively unchanged |

Interpretation:

- The optimization mainly improves the prefill-to-decode KV transfer path, so TTFT and total
  throughput improve substantially.
- TPOT is nearly unchanged because the decode token loop is still dominated by the decode GPU/model
  execution, not the initial KV transfer.
- The original-NIXL comparison must explicitly force TCP fallback with `UCX_TLS=tcp,cuda_copy,self`
  in this environment; otherwise UCX attempts unsupported CUDA memory registration on the RDMA HCA
  and the server does not start.
- The optimized run's staged profile confirmed `local_shared_chunks=0`, `rdma_write_posted > 0`,
  `flush_posted > 0`, and `ready_sent == ack_received`.

### Multi-Node Long-Output QPS Matrix

Validated on June 4, 2026 with the same host/GPU placement:

```text
prefill host/GPU: sglang-rdma-0-26 GPU0
decode host/GPU:  sglang-rdma-0-41 GPU6
model:            /opt/users/models/Qwen3-8B-FP8
dataset:          generated-shared-prefix
input shape:      gsp_system_prompt_len=896, gsp_question_len=128
output lengths:   200, 300
request rates:    2, 4, inf
max_concurrency:  16
successful reqs:  64 per condition
```

Original NIXL was run as the TCP/cuda-copy fallback baseline:

```text
UCX_TLS=tcp,cuda_copy,self
SGLANG_DISAGG_STAGING_BUFFER=0
SGLANG_MOONCAKE_HOST_STAGING=0
```

Optimized NIXL used the staged RDMA path:

```text
NIXL_UCX_VRAM_STAGING=1
NIXL_UCX_VRAM_LOCAL_STAGING=0
NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH=1
NIXL_UCX_STAGING_CHUNK_SIZE=16777216
NIXL_UCX_STAGING_SLOTS=4
NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW=32
```

Observed comparison:

| Output | Request rate | Original req/s | Staged req/s | Original total tok/s | Staged total tok/s | Original mean TTFT | Staged mean TTFT | Original mean TPOT | Staged mean TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 2 | 1.79 | 1.82 | 2292.68 | 2335.78 | 1131.74 ms | 354.47 ms | 29.75 ms | 29.59 ms |
| 200 | 4 | 2.00 | 2.27 | 2569.90 | 2914.72 | 1149.59 ms | 571.59 ms | 29.67 ms | 29.50 ms |
| 200 | inf | 2.00 | 2.27 | 2559.91 | 2908.54 | 1347.48 ms | 845.02 ms | 29.90 ms | 29.33 ms |
| 300 | 2 | 1.46 | 1.57 | 2029.25 | 2176.45 | 921.01 ms | 413.31 ms | 29.53 ms | 29.12 ms |
| 300 | 4 | 1.46 | 1.57 | 2025.86 | 2184.00 | 1107.70 ms | 763.07 ms | 29.77 ms | 29.25 ms |
| 300 | inf | 1.41 | 1.51 | 1957.13 | 2096.48 | 1351.03 ms | 1304.06 ms | 30.92 ms | 29.31 ms |

Relative change:

| Output | Request rate | Req/s change | Total tok/s change | Mean TTFT change | Mean TPOT change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 2 | 1.02x | 1.02x | 68.7% lower | effectively unchanged |
| 200 | 4 | 1.14x | 1.13x | 50.3% lower | effectively unchanged |
| 200 | inf | 1.14x | 1.14x | 37.3% lower | effectively unchanged |
| 300 | 2 | 1.08x | 1.07x | 55.1% lower | effectively unchanged |
| 300 | 4 | 1.08x | 1.09x | 31.1% lower | effectively unchanged |
| 300 | inf | 1.07x | 1.07x | 3.5% lower | staged TPOT slightly lower |

Interpretation:

- With longer outputs, decode generation dominates more of the request lifetime, so total throughput
  improvement is smaller than in the short-output benchmark.
- The staged path still consistently improves TTFT, especially at moderate request rates, because
  the prefill-to-decode KV transfer finishes faster.
- TPOT remains nearly unchanged across the matrix. This is expected because TPOT is dominated by the
  decode GPU/model execution after the transferred KV is already resident on the decode GPU.
- At `output_len=300` and saturated `request_rate=inf`, TTFT improvement is small because the decode
  side is already the dominant bottleneck.
- The staged profile for this matrix confirmed remote staged RDMA: `local_shared_chunks=0`,
  `rdma_write_posted > 0`, `flush_posted > 0`, and `ready_sent == ack_received`. The decode target
  profile accumulated about 62.4 GB of staged payload during the run.

### Multi-Node 256-Request Clean QPS Matrix

The previous 64-request matrix was useful for trend checking but too small for stable comparison.
The 256-request follow-up used the same host/GPU placement and model, but bound the router to
`127.0.0.1` to avoid external HTTP scan traffic hitting the benchmark router:

```text
router host bind: 127.0.0.1
prefill host/GPU: sglang-rdma-0-26 GPU0
decode host/GPU:  sglang-rdma-0-41 GPU6
model:            /opt/users/models/Qwen3-8B-FP8
dataset:          generated-shared-prefix
input shape:      gsp_system_prompt_len=896, gsp_question_len=128
gsp groups:       4
prompts/group:    64
effective reqs:   256 per condition
max_concurrency:  16
```

Observed comparison:

| Output | Request rate | Original req/s | Staged req/s | Original total tok/s | Staged total tok/s | Original mean TTFT | Staged mean TTFT | Original mean TPOT | Staged mean TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 2 | 1.87 | 1.87 | 2400.48 | 2405.69 | 949.68 ms | 365.10 ms | 30.39 ms | 29.60 ms |
| 200 | 4 | 2.23 | 2.35 | 2864.77 | 3018.76 | 866.71 ms | 686.93 ms | 30.51 ms | 29.70 ms |
| 200 | inf | 2.22 | 2.33 | 2852.49 | 3002.47 | 863.11 ms | 773.96 ms | 30.95 ms | 29.76 ms |
| 300 | 2 | 1.58 | 1.68 | 2202.18 | 2336.35 | 713.46 ms | 483.00 ms | 30.34 ms | 29.37 ms |
| 300 | 4 | 1.59 | 1.66 | 2205.50 | 2303.07 | 729.29 ms | 639.91 ms | 30.36 ms | 29.49 ms |
| 300 | inf | failed | 1.66 | failed | 2308.53 | failed | 648.27 ms | failed | 29.53 ms |

Relative change for completed pairs:

| Output | Request rate | Req/s change | Total tok/s change | Mean TTFT change | Mean TPOT change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 2 | 1.00x | 1.00x | 61.6% lower | 2.6% lower |
| 200 | 4 | 1.05x | 1.05x | 20.7% lower | 2.7% lower |
| 200 | inf | 1.05x | 1.05x | 10.3% lower | 3.0% lower |
| 300 | 2 | 1.06x | 1.06x | 32.3% lower | 3.2% lower |
| 300 | 4 | 1.04x | 1.04x | 12.3% lower | 3.1% lower |
| 300 | inf | staged completed | staged completed | original failed | staged completed |

The original-NIXL TCP fallback baseline completed the first five conditions, but failed the saturated
`output_len=300, request_rate=inf` condition. Decode reported repeated transfer timeouts:

```text
Request <bootstrap_room> waiting_timeout
Decode transfer failed ... with exception NIXL KVReceiver Exception
```

The staged RDMA run completed all six 256-request conditions. This makes the longer-output result
more precise:

- At moderate load, staged RDMA still materially lowers TTFT because KV arrives at decode faster.
- At saturated load with long outputs, decode generation dominates, so req/s and total token
  throughput improve only modestly.
- TPOT remains close because it is dominated by the decode loop after KV transfer has completed.
- The original TCP fallback path can become unstable under saturated long-output pressure, while
  staged RDMA completed the same condition.

### Multi-Node Byte-Level Correctness Stress

SGLang output checks are useful, but they are not a bit-level proof that transferred KV tensors are
unchanged. The stronger backend-level check is `staged_vram_write_smoke`: the initiator fills source
VRAM with a deterministic byte pattern that encodes initiator id, transfer index, descriptor offset,
and byte offset; the target copies the final target VRAM region back to host and verifies every byte.

Validation run on 2026-06-05:

```text
initiator host/GPU: sglang-rdma-0-26 GPU0, and GPU1 for the two-initiator case
target host/GPU:    sglang-rdma-0-41 GPU6
backend:            UCX staged VRAM
UCX_TLS:            rc,ud,self
route changes:      none
```

Common staged settings:

```text
NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH=1
NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW=32
NIXL_UCX_STAGING_BATCH_FLUSH=0
NIXL_UCX_STAGING_TARGET_H2D_WORKER=0
```

Results:

| Case | Payload | Concurrency | Iterations | Descriptors/write | Initiators | Transfer-only bandwidth | Target byte verification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 1 GiB | 16 | 16 | 4 | 1 | 14.37 GiB/s | passed |
| stress | 8 GiB | 32 | 16 | 8 | 1 | 19.17 GiB/s | passed |
| multi-initiator ownership | 4 GiB total | 16 each | 16 each | 8 | 2 | 9.43 / 9.23 GiB/s per initiator | passed |

The two-initiator case is the important ownership check: both initiators wrote concurrently to the
same target GPU registration, and the target verified the merged target buffer by initiator-specific
patterns. This covers the main staged-slot corruption risks: remote slot lease collision, chunk
ordering errors, descriptor slicing errors, premature ACK, and cross-initiator overwrite.

This does not eliminate the need for SGLang-level validation. For production KV correctness, keep a
high-concurrency deterministic SGLang check as a separate acceptance test: run the same prompts with
`temperature=0`, compare output token ids against a low-concurrency reference, and treat any
divergence as requiring deeper KV instrumentation. A true bitwise KV proof inside SGLang would
require copying or checksumming the actual source and target KV tensor ranges around the NIXL
transfer.

### Multi-Node 1-Hour NIXL Stability Smoke

The regular `staged_vram_write_smoke` allocates a unique target offset for every transfer, so it is
not suitable for a long leak test: the target GPU buffer would grow with the iteration count. For
stability validation, a temporary `staged_vram_stability` binary was used instead. It keeps one UCX
backend and one registered VRAM region alive, repeatedly creates/posts/polls/releases WRITE
requests against a fixed 64 MiB target ring, and verifies the target GPU ring after every iteration
before allowing the next iteration to overwrite it.

Validation run on 2026-06-05:

```text
initiator host/GPU: sglang-rdma-0-26 GPU0
target host/GPU:    sglang-rdma-0-41 GPU6
duration:           3600 s
bytes/write:        4 MiB
concurrency:        16
descriptors/write:  4
registered region:  64 MiB on each side
verification:       target byte verification after every iteration
route changes:      none
```

Result:

| Metric | Value |
| --- | ---: |
| Completed iterations | 32234 |
| Completed WRITE transfers | 515744 |
| Verified payload | 2014.62 GiB |
| Runtime | 3600 s |
| Effective verified-loop bandwidth | 0.56 GiB/s |
| Target verification failures | 0 |
| Transfer timeouts/errors | 0 |

Memory observations:

| Side | Initial steady RSS | Final steady RSS before cleanup | RSS after cleanup | CUDA free during run | CUDA free after cleanup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initiator | 394232 KiB | 394604 KiB | 378452 KiB | 23622 MiB | 23686 MiB |
| Target | 300820 KiB | 301152 KiB | 284776 KiB | 23624 MiB | 23688 MiB |

Interpretation:

- No per-transfer or per-iteration linear RSS growth was observed over 515744 WRITE transfers.
- CUDA free memory stayed flat during the run and returned after deregistration/free.
- The small RSS steps were bounded: target increased by about 332 KiB early in the run and then
  stayed flat; initiator increased by about 372 KiB around the middle of the run and then stayed
  flat.
- Every iteration was byte-verified on the target GPU, so this run also covered long-duration
  repeated overwrite of the same KV-like target pages.

Open issue found during teardown:

```text
Communication thread has thrown an exception: std::bad_alloc
```

This warning appeared on both initiator and target after `STABILITY_DONE`, after deregistration and
GPU free had completed. It did not cause data corruption or a nonzero process exit in this run, but
it is a real stability follow-up for production: inspect `nixlAgent` comm-thread shutdown, queue
drain, and late metadata/notification handling during agent destruction.

## SGLang Path Audit Checklist

Before increasing QPS, confirm from logs or light instrumentation:

```text
operation = NIXL_WRITE
local descriptor type = VRAM_SEG
remote descriptor type = VRAM_SEG
descriptor counts match
descriptor lengths match
request handle is not reposted after completion
backend = UCX
```

Also record:

```text
model
GPU ids
hostnames
container ids
NIXL commit
SGLang commit/package version
UCX version
staged env block
chunk size
slot count
concurrency/QPS
```

## Performance Runs

After smoke passes, run controlled benchmarks.

Single-node local:

```text
baseline: previous SGLang transfer path
candidate: NIXL UCX local shared staging
metrics: TTFT, TPOT, total throughput, P50/P90/P99 latency, transfer profile
```

Multi-node:

```text
baseline: existing NIXL UCX path or TCP fallback path
candidate: NIXL UCX RDMA staged path
metrics: TTFT, TPOT, total throughput, P50/P90/P99 latency, RDMA counter delta, transfer profile
```

Use the same model, prompt set, output length, QPS scan, and GPU placement for baseline and
candidate.

## Rollback

Single-node rollback:

```text
unset NIXL_UCX_VRAM_LOCAL_STAGING
```

Multi-node rollback:

```text
unset NIXL_UCX_VRAM_STAGING
unset NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH
```

Full rollback:

```text
remove NIXL_PLUGIN_DIR override
restore previous SGLang/NIXL environment
```

## Debug Matrix

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| `local_shared_chunks = 0` on single node | host id mismatch or local staging disabled | set env on both processes and use explicit `NIXL_UCX_LOCAL_STAGING_HOST_ID` |
| local attach failure | staging dir not shared or path rejected | share IPC/bind mount and check `NIXL_UCX_LOCAL_STAGING_SHM_DIR` |
| local fallback count increases | target cannot map/register source shared file | inspect target log for path/cudaHostRegister error |
| RDMA counters do not move | UCX not using selected HCA | check `UCX_TLS`, `ucx_devices`, and HCA availability without route changes |
| staged transfer unsupported | SGLang used READ or non-VRAM descriptors | stop and inspect NIXL operation/descriptor path |
| target sees stale KV | app stream ordering issue or premature completion | verify NIXL completion/notification is used before decode consumes KV |
| high setup time but good transfer time | registration or benchmark setup included | compare transfer-loop timing, not shell real time |

## Promotion Criteria

Promote from smoke to small production shadow only when:

- single-node or multi-node NIXL smoke passes;
- SGLang small PD run passes correctness;
- transfer profile matches the intended path;
- rollback has been tested;
- no route changes were required;
- logs do not show repeated fallback, attach failure, or unsupported operation.
