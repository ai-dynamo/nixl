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
`NIXL_UCX_LOCAL_STAGING_SHM_DIR`, contain the source agent and region id, and match the expected
chunk slot offset and size bounds.

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
