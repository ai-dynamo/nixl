<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPUNETIO: two NIXLBench workers on one GPU

## Problem and remedy

Two GPUNETIO processes on one physical GPU have separate CUDA contexts.
Persistent progress kernels can interact with context scheduling so that a
transfer kernel waits milliseconds before it starts. NIXLBench includes this
delay in its timing: a low effective bandwidth need not mean slow NIC execution.

Use separate GPUs where possible, or use the single-GPU paired launcher below.
It automatically manages a private NVIDIA Multi-Process Service (MPS) session
for its two local workers, before either initializes CUDA. It does not modify
the GPUNETIO backend or change the normal nixlbench invocation.

MPS is a configuration remedy, not a bandwidth-formula correction. Results must
state whether it was used. Read the installed driver's
[MPS requirements](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html) first.

## Run the paired launcher

Requirements: Linux, Python 3 standard library, iproute2 (`ip -j`), `ps`, nvidia-smi, legacy-interface
nvidia-cuda-mps-control, an allocated idle GPU, a working GPUNETIO plugin and
a reachable etcd endpoint. Both clients and the daemon run as the same user.

Choose two local OOB interfaces with distinct, mutually reachable IPv4 addresses.
MPS does not fix a shared/wildcard fixed-port listener or incorrect RDMA/GID setup.
Configure any required GID through your benchmark/backend revision's supported
mechanism. Use the WRITE consistency fix from [#2284](https://github.com/ai-dynamo/nixl/pull/2284),
or an equivalent fix: it validates data, but does not remove scheduling delay.

From the NIXL repository root:

```bash
# Set these for your allocation. GPU_UUID must be the full UUID from nvidia-smi -L.
: "${GPU_UUID:?Set an allocated GPU UUID}"
: "${NIXLBENCH:?Set the nixlbench executable path}"
: "${ETCD_ENDPOINT:?Set a reachable etcd endpoint}"
: "${RDMA_DEVICE:?Set the RDMA device}"
: "${OOB0:?Set the first OOB interface}"
: "${OOB1:?Set the second, distinct-address OOB interface}"

python3 benchmark/nixlbench/scripts/run_gpunetio_single_gpu.py \
  --gpu "$GPU_UUID" --oob-interfaces "$OOB0" "$OOB1" \
  --mps auto --timeout 120 -- "$NIXLBENCH" \
  --etcd_endpoints="$ETCD_ENDPOINT" --device_list="$RDMA_DEVICE" \
  --initiator_seg_type=VRAM --target_seg_type=VRAM --op_type=WRITE \
  --total_buffer_size=67108864 --start_block_size=4096 --max_block_size=4096 \
  --start_batch_size=1 --max_batch_size=1 --pipeline_depth=1 \
  --num_threads=1 --warmup_iter=8 --num_iter=64
```

The launcher creates a fresh output directory (or takes a non-existing
`--output` path), launches both workers and sets the shared benchmark group.
It owns backend/runtime/topology/GPU/OOB/consistency options; overriding these
or using config files/raw CLI is outside this entry point. Other workload
flags are passed unchanged. It always enables consistency checking.

`--mps auto` is the default **for this single-GPU launcher only**. It starts a
private daemon and verifies that both exact worker PIDs join it. The selected
full GPU UUID is retained in the daemon and client environments; GPUNETIO uses
logical device 0. See NVIDIA's [UUID and pipe-directory rules](https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html).

The console identifies the MPS mode and reason. `result.json` records the GPU,
commands, PIDs, verified MPS membership, exit codes and outcome; worker logs
contain the benchmark's latency and bandwidth rows. These files may contain
your environment's private identifiers: redact them before publication.

## Control run and validation

Repeat the same command with `--mps off`. Keep binaries, GPU, NIC/OOB settings,
payload, iterations and validation identical. Run at least three fresh pairs
per arm with alternating order. Record all run values and software versions.
A workload that finishes before both MPS client PIDs are observed is rejected;
increase the iteration count in **both** arms if necessary. Test READ separately
by changing only `--op_type` in both arms.

No result is accepted if a worker fails, times out, logs a validation/CQE error,
or neither log contains an initiator benchmark row. MPS startup, membership or cleanup failure is
not silently converted into a successful non-MPS run.

CPU-only launcher checks (fake NVIDIA tools; no CUDA/MPS service is started):

```bash
python3 benchmark/nixlbench/tests/test_single_gpu_launcher.py
```

## Ownership and limitations

The launcher refuses an occupied GPU, a visible live MPS process or an inherited MPS pipe configuration;
it does not manage an existing scheduler-owned service. The caller must own the
GPU allocation: an empty process snapshot is not a resource reservation. Private
pipes isolate control, not hardware resources or fault effects. Run where the
process list exposes the host's MPS services; a container that hides them cannot
establish exclusivity. A launcher preflight cannot replace scheduler coordination.

It never changes compute mode/clocks and never stops other processes. On error
it asks its own workers to stop, with a bounded wait, before quitting its daemon.
If a worker will not exit, it retains the private daemon and reports failure
and recorded PIDs for operator recovery, rather than force-killing active CUDA
work. Do not start another run on that GPU until recovery is complete.
There is no automatic sudo, global MPS takeover or cross-node orchestration.
The shutdown/control commands use the
[legacy MPS interface](https://docs.nvidia.com/deploy/mps/common-tasks.html).

## Historical bounded evidence

H20, driver 550.127.08, CUDA 12.8, DOCA 3.1; #2052 backend at `41e93a3c`
plus benchmark GID forwarding and the WRITE checker repair. Same binary hashes
in both arms, one GPU, two processes, one descriptor/QP per connection, 4-KiB
WRITE, batch/pipeline depth 1, 64 measured iterations and 64-MiB buffers.
`--warmup_iter=8` produced 16 effective warmups in that runner.

| Same-GPU placement | Per-run average latency (us) | Median of run averages (us) | Median run P99 Tx (us) |
|---|---|---:|---:|
| MPS off | 4405.5 / 4403.6 / 4403.6 | 4403.6 | 4446 |
| MPS on | 26.1 / 27.0 / 26.2 | 26.2 | 23 |

All six pairs passed final destination-payload validation and exited successfully.
MPS runs verified both client PIDs and private-daemon cleanup. A separate READ
regression passed at 29.0 us average. Profiling located the dominant non-MPS
delay between CUDA launch return and transfer kernel start.

These archived measurements used the original diagnostic wrapper, not the new
launcher. They are not NIC peak-bandwidth or serving results. Current main
requires DOCA >=3.5; the historical measurements do not establish performance
or compatibility on that stack. Sustained multi-client throughput, fairness,
failure isolation and serving/training coexistence were not validated.
