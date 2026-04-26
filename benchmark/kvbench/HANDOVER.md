# KVBench Handover Documentation

**Branch**: `mtennenhaus/kvbench`
**Last Updated**: 2026-02

---

## Table of Contents

1. [Branch Changes Summary](#1-branch-changes-summary)
2. [Code Structure Changes](#2-code-structure-changes)
3. [Architecture Overview](#3-architecture-overview)
4. [Benchmarking Tactics](#4-benchmarking-tactics)
5. [KV Cache Workload Generation](#5-kv-cache-workload-generation)
6. [Running Benchmarks](#6-running-benchmarks)
7. [Using the Storage API](#7-using-the-storage-api)
8. [Storage I/O Optimization](#8-storage-io-optimization)
9. [Benchmark Results Summary](#9-benchmark-results-summary)
10. [Known Issues and Findings](#10-known-issues-and-findings)
11. [File Reference](#11-file-reference)

---

## 1. Branch Changes Summary

### Commits (local, on top of upstream merge-base)

The branch contains work in two areas: (a) storage I/O optimization for the
POSIX/GDS backends, and (b) KV cache workload generation for realistic LLM
inference benchmarks.

### Key Code Changes

#### Production Code

| File | Change |
|------|--------|
| `benchmark/kvbench/main.py` | Added `--storage-block-size`, `--storage-posix-api`, `--storage-num-handles` CLI options |
| `benchmark/kvbench/test/storage_backend.py` | Redesigned `FilesystemBackend`: chunked descriptors (`_create_chunked_descs`), multi-file sharding, parallel handle submission |
| `benchmark/kvbench/test/sequential_custom_traffic_perftest.py` | Multi-handle storage support, fixed rank index bug, storage iteration timing |
| `benchmark/kvbench/test/inference_workload_matgen.py` | Extended with `--storage-only`, `--read-only`, `--all-nodes-per-pattern`, `--mem-type` options; N:1 prefill:decode worker ratio support; skip parallelism assertions for storage-only mode |
| `src/bindings/python/nixl_bindings.cpp` | Added `parallelPostXferReqs()` and `parallelStorageTransfer()` C++ binding methods |
| `src/api/python/_api.py` | Python wrappers for the new bindings |

#### Diagnostic and Test Files

| File | Purpose |
|------|---------|
| `test/standalone_test.cpp` | Pure C++ NIXL benchmark proving 45 GB/s without Python overhead |
| `test/verify_binding.py` | Tests `parallelStorageTransfer` binding with iodepth sweeps |
| `test/mini_storage_bench.py` | Microbenchmark isolating POSIX backend: block sizes, handle counts, threading |
| `test/diagnose_io.py` | Layer-by-layer I/O diagnostic: raw OS, NIXL single, NIXL multi-handle |
| `STORAGE_OPTIMIZATION.md` | Root cause analysis and fix documentation for storage throughput |

#### Removed (obsolete)

| File | Reason |
|------|--------|
| `test/find_backend.py` | Superseded by current backend selection in main.py |
| `test/parallel_xfer.cpp` | Replaced by standalone_test.cpp |
| `test/run_comparison.sh` | Ad-hoc script, replaced by sbatch files |
| `test/run_final_compare.sh` | Ad-hoc script, replaced by sbatch files |
| `test/run_nixlbench_tests.sh` | Ad-hoc script, replaced by sbatch files |

---

## 2. Code Structure Changes

This section walks through the code changes from the perspective of someone
already familiar with the kvbench codebase (pre-storage-ops).

### What Changed and Why

Before this branch, kvbench's storage I/O used a simple 1-descriptor-per-read
model. A 256MB read created a single NIXL descriptor, which translated to a
single io_uring SQE with queue depth 1. On NFS (VAST), this limited throughput
to ~10 GB/s per node because the NFS client could not pipeline RPCs.

The branch introduces three layers of optimization, each with corresponding
code changes:

### 2.1 storage_backend.py -- Complete Redesign

**Before**: `FilesystemBackend.get_read_handle()` created one NIXL descriptor
for the entire read size. One descriptor = one io_uring SQE = iodepth 1.

**After**: Two new mechanisms:

**Block splitting** (`_create_chunked_descs`, line 334):
- When `block_size > 0`, a 256MB read becomes 256 x 1MB descriptors
- NIXL's POSIX backend creates io_uring with `queue_depth = descriptor_count`
- NFS can now pipeline 256 RPCs concurrently
- Impact: 10 GB/s -> 28 GB/s

```python
def _create_chunked_descs(self, fd, file_offset, total_size, buffer):
    # Splits total_size into block_size chunks
    # Returns (local_descs, file_descs) with N entries each
    for off in range(0, total_size, bs):
        file_tuples.append((file_offset + off, min(bs, total_size - off), fd))
        local_tuples.append((buf_addr + off, min(bs, total_size - off), dev_id))
```

**Multi-file sharding** (`prepare()` with `_num_handles > 1`, line 262):
- Creates N separate files per rank (one per handle/shard)
- Each file gets its own fd, enabling NFS connection-level parallelism
- `get_read_handles()` / `get_write_handles()` return N separate NIXL handles
- These are submitted concurrently via `parallelPostXferReqs` (C++ binding)
- Impact: 28 GB/s -> 45 GB/s

**Key constraint**: NIXL's `createXferReq` -> `populate()` requires all FILE
descriptors in a single request to belong to the same registered region.
Multiple fds from different files in one request causes `NIXL_ERR_NOT_FOUND`.
This is why multi-fd parallelism requires *separate* requests, submitted
concurrently from C++ threads.

The public API surface:

| Method | Single-handle (legacy) | Multi-handle (new) |
|--------|----------------------|-------------------|
| `prepare()` | Creates 1 file | Creates N shard files |
| `get_read_handle()` | Returns 1 handle | Returns 1 handle (unchanged) |
| `get_read_handles()` | N/A (new) | Returns N handles, one per shard fd |
| `get_write_handle()` | Returns 1 handle | Returns 1 handle (unchanged) |
| `get_write_handles()` | N/A (new) | Returns N handles, one per shard fd |

### 2.2 sequential_custom_traffic_perftest.py -- Multi-Handle Support

Changes are minimal and additive:

- **Constructor**: Accepts `storage_num_handles` parameter, passes it to
  `FilesystemBackend._num_handles`.
- **`_prepare_storage_xfer()`** (line 278): When `num_handles > 1`, calls
  `get_read_handles()` / `get_write_handles()` instead of the singular versions.
  Returns a list of `StorageXferHandle` objects.
- **Execution path**: When multiple handles are returned, they are submitted
  concurrently via `parallelPostXferReqs()` from the C++ bindings. This
  bypasses the Python GIL and posts each handle from a separate C++ thread.
- **Bug fix**: Fixed pre-existing `IndexError` in `_print_iteration_results()`
  where YAML config rank indices could exceed `world_size` when running with
  fewer ranks than the config specifies.

### 2.3 nixl_bindings.cpp -- Two New C++ Bindings

**`parallelPostXferReqs(reqhs, num_threads)`**:
- Takes a list of NIXL transfer request handles
- Spawns C++ threads (bypassing GIL) to post each handle concurrently
- Each thread: post request, poll until complete
- Used by kvbench when `storage_num_handles > 1`

**`parallelStorageTransfer(...)`**:
- Benchmark-oriented function replicating nixlbench's exact `execTransfer` pattern
- OMP parallel section, per-thread `posix_memalign`, `createXferReq` inside OMP
- Used only for standalone performance validation (not called by kvbench)

### 2.4 main.py -- Three New CLI Options

```
--storage-block-size 1M     # Split reads into 1MB descriptors (default: 0 = off)
--storage-posix-api uring   # io_uring instead of AIO (default: auto)
--storage-num-handles 16    # Multi-file parallelism (default: 1)
```

These flow through to:
```
main.py -> SequentialCTPerftest.__init__() -> FilesystemBackend.__init__()
                                           -> FilesystemBackend._num_handles
```

### 2.5 inference_workload_matgen.py -- Config Generator Extensions

New CLI options for storage-only benchmark configs:

| Option | Purpose |
|--------|---------|
| `--storage-only` | Skip RDMA matrix generation, produce storage-only YAML |
| `--read-only` | Zero out write sizes (read-only patterns) |
| `--all-nodes-per-pattern` | All nodes active in each pattern (vs round-robin) |
| `--mem-type cuda\|cpu` | Memory type for storage operations |
| `--iters N` | Iterations per traffic pattern |
| `--isolation-iters N` | Isolation iterations between patterns |

Logic changes:
- Parallelism assertions (prefill TP <= decode TP, etc.) are skipped when
  `num_decode_gpus == 0` (storage-only mode)
- N:1 prefill:decode worker ratio supported (e.g., 8 prefill workers to 4
  decode workers)
- Round-robin TP group assignment for storage-only patterns

---

## 3. Architecture Overview

### Benchmark Flow

```
main.py sequential-ct-perftest <config.yaml>
    |
    v
sequential_custom_traffic_perftest.py
    |
    +-- Reads YAML config (traffic patterns with storage read/write sizes per rank)
    +-- Sets up NIXL agent, registers memory, connects via ETCD
    +-- For each traffic pattern:
    |       1. Storage READ  (blocking) - each rank reads from storage
    |       2. Compute SLEEP (optional) - simulates prefill/decode compute
    |       3. Storage WRITE (async)    - each rank writes to storage
    |       4. RDMA Transfer (blocking) - all-to-all communication
    |       5. Wait for Write (blocking)
    |
    +-- Reports per-pattern timing: storage read/write ms, RDMA ms, bandwidth
```

### Storage Backend Stack

```
sequential_custom_traffic_perftest.py
    |
    v
storage_backend.py (FilesystemBackend)
    |-- _create_chunked_descs()  --> splits large reads into 1MB descriptors
    |-- prepare()                --> creates N shard files per rank
    |-- get_read_handles()       --> returns N transfer handles (one per shard)
    |
    v
NIXL Python bindings (nixl_bindings.cpp)
    |-- parallelPostXferReqs()   --> submits N handles concurrently from C++ threads
    |
    v
NIXL POSIX Backend (posix_backend.cpp)
    |-- io_uring queue_depth = descriptor_count
    |-- submits all SQEs in single io_uring_submit()
```

### KV Cache Config Generation

```
inference_workload_matgen.py generate
    |
    +-- Defines model (llama-70b: 80 layers, 8 KV heads, 128 head_dim)
    +-- Generates random ISL (Input Sequence Length) per request
    +-- Calculates KV cache size: ISL x bytes_per_token / TP
    +-- Applies hit_rate to determine storage read vs write
    +-- Distributes requests round-robin across TP groups
    +-- Outputs metadata.yaml with traffic patterns
```

---

## 4. Benchmarking Tactics

### 3.1 Storage-Only Benchmarks (Primary Focus)

Tests VAST/NFS storage throughput via NIXL POSIX or GDS backends without RDMA
transfers. Simulates the storage read portion of disaggregated
prefill-decode: each rank reads its KV cache shard from storage.

**Test matrix**:

| Variable | Values |
|----------|--------|
| Nodes | 1, 4, 8, 12 |
| TP | 4, 8 |
| Traffic patterns | 264, 512 |
| Read size range | 250MB - 1GB per rank |
| Backend | POSIX (O_DIRECT, io_uring), GDS |
| Memory type | CUDA, CPU |

**Key sbatch files** (in `test_plans/isr1_pre_test/`):

| File | Description |
|------|-------------|
| `run_8nodes_storage_tp8_264tp_vast.sbatch` | 8-node, TP=8, 264 patterns, GDS on VAST |
| `run_8nodes_storage_tp4_264tp_vast.sbatch` | 8-node, TP=4, 264 patterns, GDS on VAST |
| `run_8nodes_storage_tp8_512tp_vast.sbatch` | 8-node, TP=8, 512 patterns, GDS on VAST |
| `run_8nodes_storage_tp4_512tp_vast.sbatch` | 8-node, TP=4, 512 patterns, GDS on VAST |
| `run_12nodes_tp8_264tp_vast.sbatch` | 12-node, TP=8, 264 patterns |
| `run_12nodes_tp4_264tp_vast.sbatch` | 12-node, TP=4, 264 patterns |
| `run_storage_posix_large.sbatch` | 2-node POSIX O_DIRECT stress test |
| `run_storage_posix_optimized.sbatch` | Optimized POSIX (block splitting + uring) |

### 3.2 Saturation / Peak Throughput Tests

Simple fixed-size read patterns testing the maximum achievable storage
bandwidth per node. Vary ranks-per-node and total read size to find the
saturation point.

**Test matrix**:

| Variable | Values |
|----------|--------|
| Ranks per node | 1, 2, 4, 8, 16, 32 |
| Read size per rank | 128MB, 256MB, 512MB, 1GB |
| Nodes | 1, 2, 4, 6 |

**Config directory**: `test/saturation_v2/` (untracked, regenerate with `create_configs.py`)

**sbatch files** (in `test_plans/isr1_pre_test/`):

| File | Description |
|------|-------------|
| `run_sat_v2.sbatch` | Parameterized sbatch, called with env vars (SAT_NAME, SAT_NODES, SAT_CONFIG) |

### 3.3 RDMA-Only Benchmarks

Tests inter-node RDMA (UCX) throughput without storage. Measures the network
fabric bandwidth between nodes.

**sbatch files** (in `test_plans/isr1_pre_test/`):

| File | Description |
|------|-------------|
| `run_rdma.sbatch` | Pure RDMA transfers, CPU memory |

### 3.4 Combined RDMA + Storage

Full prefill-decode simulation: storage read, RDMA transfer, storage write.
Represents the complete KV cache disaggregation workflow.

**sbatch files** (in `test_plans/isr1_pre_test/`):

| File | Description |
|------|-------------|
| `run_rdma_storage.sbatch` | 2-node GDS_MT + RDMA combined |
| `run_12nodes_rdma_storage.sbatch` | 12-node combined test |
| `run_12nodes_decode_posix.sbatch` | 12-node prefill-decode with POSIX |

### 3.5 GDS-Specific Tests

Tests GPU Direct Storage (GDS) backend, where data moves directly between GPU
memory and storage without CPU staging.

**sbatch files** (in `test_plans/isr1_pre_test/`):

| File | Description |
|------|-------------|
| `run_storage_gds.sbatch` | GDS backend on VAST |
| `run_1node_gds_test.sbatch` | Single-node GDS validation |
| `run_1node_gds_minimal.sbatch` | Minimal GDS test for debugging |
| `run_4nodes_decode_gds.sbatch` | 4-node decode with GDS |
| `run_4nodes_decode_gds_fast.sbatch` | 4-node GDS (short iteration) |

### 3.6 Baseline Comparisons

FIO and nixlbench used as reference baselines for VAST throughput.

**sbatch files** (in `test_plans/isr1_pre_test/`):

| File | Description |
|------|-------------|
| `run_8nodes_fio_simple.sbatch` | 8-node FIO sequential read test |
| `run_8nodes_fio_read_vast.sbatch` | 8-node FIO read on VAST |
| `run_8nodes_fio_isolated_agg_vast.sbatch` | FIO with isolated aggregation |

**sbatch files** (in `test_plans/spcx_test/`):

| File | Description |
|------|-------------|
| `run_nixlbench_baseline.sbatch` | nixlbench reference (48 GB/s per node) |
| `run_vast_scaling.sbatch` | VAST scaling test |
| `run_all_saturation.sbatch` | Full saturation sweep |

---

## 5. KV Cache Workload Generation

### Model: Llama 3.1 70B

| Parameter | Value |
|-----------|-------|
| Layers | 80 |
| KV Heads | 8 (GQA) |
| Head Dim | 128 |
| Dtype | FP16 (2 bytes) |
| **KV Bytes/Token** | **320 KB** |

### Calculation

```
KV per token = 2 (K+V) x layers x kv_heads x head_dim x dtype_size
             = 2 x 80 x 8 x 128 x 2 = 327,680 bytes = 320 KB

Per rank with TP:
  TP=4: 320KB / 4 = 80 KB/token/rank
  TP=8: 320KB / 8 = 40 KB/token/rank

Storage read per rank = ISL x (bytes_per_token / TP) x (1 - hit_rate)
  TP=4, ISL=64K, hit=20%:  64000 x 80KB x 0.2 = 1 GB
  TP=8, ISL=128K, hit=20%: 128000 x 40KB x 0.2 = 1 GB
```

### Generated Config Summary

All configs: read-only, 20% hit rate, round-robin TP group distribution, max 1GB per rank.

**8-node configs** (in `test/decode_centric_12nodes/`):

| Directory | TP | Patterns | Max ISL | Max Read/Rank |
|-----------|----|----------|---------|---------------|
| `storage_70b_8nodes_tp4_264tp_read_only/` | 4 | 264 | 64K | 1 GB |
| `storage_70b_8nodes_tp8_264tp_read_only/` | 8 | 264 | 128K | 1 GB |
| `storage_70b_8nodes_tp4_512tp_read_only/` | 4 | 512 | 64K | 1 GB |
| `storage_70b_8nodes_tp8_512tp_read_only/` | 8 | 512 | 128K | 1 GB |

**12-node configs** (in `test/decode_centric_12nodes/`):

| Directory | TP | Patterns | Max ISL | Max Read/Rank |
|-----------|----|----------|---------|---------------|
| `storage_70b_12nodes_tp4_264tp_read_only/` | 4 | 264 | 64K | 1 GB |
| `storage_70b_12nodes_tp8_264tp_read_only/` | 8 | 264 | 128K | 1 GB |
| `storage_70b_12nodes_tp4_512tp_read_only/` | 4 | 512 | 64K | 1 GB |
| `storage_70b_12nodes_tp8_512tp_read_only/` | 8 | 512 | 128K | 1 GB |

### Regenerating Configs

The generated YAML configs are not tracked in git. Regenerate with:

```bash
source /.autodirect/mtrsysgwork/mtennenhaus/isr1_venv/bin/activate
cd benchmark/kvbench/test/decode_centric_12nodes

# 8-node TP=8, 264 patterns
python ../inference_workload_matgen.py generate \
  --model llama-70b \
  --prefill-tp 8 \
  --num-prefill-nodes 8 \
  --num-decode-nodes 0 \
  --num-user-requests 264 \
  --prefix-hit-rate 0.2 \
  --min-isl 16000 \
  --max-isl 128000 \
  --isl-mean 72000 \
  --isl-scale 40000 \
  --storage-only \
  --read-only \
  --results-dir storage_70b_8nodes_tp8_264tp_read_only

# 8-node TP=4, 264 patterns
python ../inference_workload_matgen.py generate \
  --model llama-70b \
  --prefill-tp 4 \
  --num-prefill-nodes 8 \
  --num-decode-nodes 0 \
  --num-user-requests 264 \
  --prefix-hit-rate 0.2 \
  --min-isl 16000 \
  --max-isl 64000 \
  --isl-mean 40000 \
  --isl-scale 20000 \
  --storage-only \
  --read-only \
  --results-dir storage_70b_8nodes_tp4_264tp_read_only
```

For 512 patterns, change `--num-user-requests 512`.
For 12 nodes, change `--num-prefill-nodes 12` and adjust the output directory name.

### ISL Distribution Parameters

The ISL (Input Sequence Length) follows a normal distribution clipped to [min, max]:

| TP | Min ISL | Max ISL | Mean | Scale | Rationale |
|----|---------|---------|------|-------|-----------|
| TP=4 | 16,000 | 64,000 | 40,000 | 20,000 | Max ISL capped at 64K so per-rank read stays under 1GB |
| TP=8 | 16,000 | 128,000 | 72,000 | 40,000 | Full 128K context; 1GB per rank at 20% hit rate |

### Available Predefined Models

```python
PREDEFINED_MODELS = {
    "llama-8b":   ModelConfig(hidden_size=4096,  num_layers=32,  num_heads=32,  num_kv_heads=8,  dtype_size=2),
    "llama-70b":  ModelConfig(hidden_size=8192,  num_layers=80,  num_heads=64,  num_kv_heads=8,  dtype_size=2),
    "llama-405b": ModelConfig(hidden_size=16384, num_layers=126, num_heads=128, num_kv_heads=8,  dtype_size=2),
    "deepseek-r1": ModelConfig(hidden_size=12288, num_layers=100, num_heads=96, num_kv_heads=12, dtype_size=2),
}
```

---

## 6. Running Benchmarks

### Prerequisites

1. **Container**: The benchmarks run inside a nixlbench container that includes NIXL and dependencies:
   ```
   /.autodirect/mswg2/E2E/Regression_logs/squash/nixlbench/gitlab-master.nvidia.com+eshukrun+warehouse+nixlbench+v0.1.0.dev.69b633aa.sqsh
   ```

2. **Python venv** (for config generation only):
   ```bash
   source /.autodirect/mtrsysgwork/mtennenhaus/isr1_venv/bin/activate
   ```

3. **ETCD**: Required for multi-node. Started as a Docker container on the master node.

4. **VAST storage**: Mount point at `/mnt/vast/` on ISR1-PRE nodes.

### Step-by-Step: Running a Storage Benchmark

1. **Generate configs** (if not already generated):
   ```bash
   cd benchmark/kvbench/test/decode_centric_12nodes
   python ../inference_workload_matgen.py generate \
     --model llama-70b --prefill-tp 8 --num-prefill-nodes 8 \
     --num-decode-nodes 0 --num-user-requests 264 \
     --prefix-hit-rate 0.2 --min-isl 16000 --max-isl 128000 \
     --isl-mean 72000 --isl-scale 40000 \
     --storage-only --read-only \
     --results-dir storage_70b_8nodes_tp8_264tp_read_only
   ```

2. **Submit sbatch**:
   ```bash
   sbatch benchmark/kvbench/test_plans/isr1_pre_test/run_8nodes_storage_tp8_264tp_vast.sbatch
   ```

3. **Monitor**:
   ```bash
   squeue -u $USER
   tail -f /auto/mtrsysgwork/mtennenhaus/kvbench_results/8n_st_tp8_264_pre_<JOB_ID>.out
   ```

4. **Results**: JSON output at the path specified by `--json-output-path` in the sbatch.

### Anatomy of an sbatch File

Every sbatch follows this pattern (using `run_8nodes_storage_tp8_264tp_vast.sbatch` as example):

```bash
# 1. SLURM headers
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

# 2. Environment
STORAGE_PATH="/mnt/vast/mtennenhaus/kvbench_storage_..."
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export NIXL_ETCD_ENDPOINTS="http://$MASTER_ADDR:2379"
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/unique_${SLURM_JOB_ID}"

# 3. Start ETCD on master node
srun -N1 -n1 -w $MASTER_ADDR docker run -d --name etcd_... etcd ...

# 4. Pre-pull container on all nodes
srun -n $SLURM_NNODES --ntasks-per-node=1 --container-image="$CONTAINER" true

# 5. Run benchmark
srun --mpi=pmix --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    bash -c "
        $BIND_SCRIPT $CONTAINER_PYTHON main.py \
            sequential-ct-perftest $CONFIG_FILE \
            --storage-backend GDS \
            --storage-path $STORAGE_PATH \
            --warmup-iters 1 \
            --isolation-iters 5 \
            --json-output-path $OUTPUT
    "

# 6. Cleanup ETCD
srun -N1 -n1 -w $MASTER_ADDR docker stop etcd_... ; docker rm etcd_...
```

Key elements:
- **`bind_8gpus.sh`**: Maps each rank to one GPU and one NIC (1:1 GPU-NIC affinity on ISR1-PRE)
- **`NIXL_ETCD_NAMESPACE`**: Must be unique per job to avoid metadata collisions
- **Container mounts**: Mount local code into container at `/workspace/nixl/benchmark/kvbench`
- **`--mpi=pmix`**: Required for multi-node MPI rank assignment via SLURM

### Key CLI Options

```bash
python main.py sequential-ct-perftest <config.yaml> \
    --storage-backend POSIX|GDS|GDS_MT    # Storage backend
    --storage-path /mnt/vast/...          # Where to read/write files
    --storage-direct-io                   # Enable O_DIRECT
    --storage-block-size 1M               # Split reads into 1MB chunks (perf optimization)
    --storage-posix-api uring             # Use io_uring instead of AIO
    --storage-num-handles 16              # Multi-file parallelism
    --warmup-iters N                      # Warmup iterations (not measured)
    --isolation-iters N                   # Gap iterations between measured patterns
    --json-output-path out.json           # JSON results file
```

---

## 7. Using the Storage API

This section explains how to use the new storage APIs, both as a CLI user
and as a developer extending the code.

### 7.1 CLI Quick Start

**Fastest POSIX reads on NFS/VAST** (45 GB/s per node):

```bash
python main.py sequential-ct-perftest config.yaml \
    --storage-backend POSIX \
    --storage-path /mnt/vast/my_storage \
    --storage-direct-io \
    --storage-block-size 1M \
    --storage-posix-api uring \
    --storage-num-handles 16 \
    --warmup-iters 1 \
    --isolation-iters 5 \
    --json-output-path results.json
```

**GDS reads (GPU Direct Storage)**:

```bash
python main.py sequential-ct-perftest config.yaml \
    --storage-backend GDS \
    --storage-path /mnt/vast/my_storage \
    --storage-direct-io \
    --json-output-path results.json
```

Note: GDS does not benefit from `--storage-block-size` or `--storage-num-handles`
because it uses its own internal batching.

**Legacy mode** (backwards compatible, ~10 GB/s):

```bash
python main.py sequential-ct-perftest config.yaml \
    --storage-backend POSIX \
    --storage-path /mnt/vast/my_storage
```

### 7.2 When to Use Each Option

| Scenario | block-size | posix-api | num-handles | Expected BW |
|----------|-----------|-----------|-------------|-------------|
| Quick test / debugging | 0 (default) | auto | 1 | ~10 GB/s |
| Moderate performance | 1M | auto | 1 | ~28 GB/s |
| Maximum POSIX throughput | 1M | uring | 16 | ~45 GB/s |
| GDS (GPU memory) | N/A | N/A | 1 | Depends on GDS version |

### 7.3 Storage Backend Python API

If you are writing new code that uses the storage backend directly:

```python
from nixl._api import nixl_agent
from test.storage_backend import FilesystemBackend

# 1. Create agent and backend
agent = nixl_agent("my_agent")
backend = FilesystemBackend(
    agent=agent,
    base_path="/mnt/vast/storage",
    nixl_backend="POSIX",
    use_direct_io=True,
    block_size=1048576,          # 1MB chunks -> high io_uring queue depth
    backend_params={"use_uring": "true"},
)
backend._num_handles = 16        # Enable multi-file sharding

# 2. Prepare storage (creates files, registers with NIXL)
handle = backend.prepare(tp_idx=0, rank=0, read_size=256*1024*1024, write_size=0)

# 3a. Single-handle read (legacy, simple)
import torch
buf = torch.empty(256*1024*1024, dtype=torch.uint8, device="cuda")
xfer = backend.get_read_handle(handle, buf)
agent.post_xfer_req(xfer)       # Submit
# ... poll until complete ...

# 3b. Multi-handle read (high throughput)
xfers = backend.get_read_handles(handle, buf, num_handles=16)
agent.parallelPostXferReqs(xfers, num_threads=16)  # Concurrent C++ submission

# 4. Cleanup
backend.close()
```

### 7.4 YAML Config Format

Traffic patterns are defined in `metadata.yaml`. Each pattern specifies
per-rank storage read/write sizes:

```yaml
traffic_patterns:
- sleep_before_launch_sec: 1.5
  metadata: {isl: 40000}
  mem_type: cuda
  storage:
    read: [625.5M, 625.5M, 625.5M, 625.5M, '0', '0', '0', '0']
    write: ['0', '0', '0', '0', '0', '0', '0', '0']
```

Fields:
- `sleep_before_launch_sec`: Inter-request arrival delay
- `metadata.isl`: Input sequence length that generated this pattern
- `mem_type`: Buffer memory type (`cuda` or `cpu`)
- `storage.read`: Per-rank read sizes (supports K/M/G suffixes, `'0'` = no read)
- `storage.write`: Per-rank write sizes (optional, omitted for read-only configs)

The read array has one entry per GPU rank. In a round-robin config with TP=4
on 64 GPUs, only 4 ranks are active per pattern (the rest are `'0'`).

### 7.5 Understanding Traffic Pattern Distribution

With **round-robin** distribution (default for storage-only), patterns cycle
through TP groups:

```
64 GPUs, TP=4 -> 16 TP groups (replicas)

Pattern 0:  ranks 0-3 active    (TP group 0)
Pattern 1:  ranks 4-7 active    (TP group 1)
Pattern 2:  ranks 8-11 active   (TP group 2)
...
Pattern 15: ranks 60-63 active  (TP group 15)
Pattern 16: ranks 0-3 active    (wraps around)
```

This simulates realistic inference where different replicas handle different
user requests at different times. The benchmark measures how storage handles
this interleaved access pattern.

---

## 8. Storage I/O Optimization

See [STORAGE_OPTIMIZATION.md](STORAGE_OPTIMIZATION.md) for the full investigation.

### Summary

kvbench storage I/O went from **~10 GB/s** to **~45 GB/s** per node (vs nixlbench's 48 GB/s baseline) through three fixes:

| Optimization | Impact | CLI Flag |
|-------------|--------|----------|
| Split reads into 1MB descriptors | 10 -> 28 GB/s | `--storage-block-size 1M` |
| Use io_uring instead of AIO | Marginal | `--storage-posix-api uring` |
| Multi-file sharding (16 fds) | 28 -> 45 GB/s | `--storage-num-handles 16` |

Root cause: NIXL POSIX backend sets `io_uring queue_depth = descriptor_count`. One large descriptor = iodepth 1 = no NFS pipelining. Splitting into many small descriptors enables concurrent I/O.

---

## 9. Benchmark Results Summary

### 9.1 Storage I/O Performance (VAST, ISR1-PRE)

These are the results from the POSIX backend optimization work on ISR1-PRE
with VAST NFS storage. See `STORAGE_OPTIMIZATION.md` for the full investigation.

| Configuration | Per-Node BW | Notes |
|---------------|-------------|-------|
| kvbench baseline (iodepth=1, AIO, 1 file) | ~10 GB/s | Original behavior |
| + `--storage-block-size 1M` | ~28 GB/s | High iodepth, single fd |
| + `--storage-posix-api uring` | ~28 GB/s | Marginal improvement over AIO |
| + `--storage-num-handles 16` | ~35 GB/s | Multi-fd NFS parallelism |
| kvbench workload aggregate (4 ranks) | **45-47 GB/s** | All optimizations combined |
| nixlbench reference (C++) | 48.5 GB/s | Standalone C++ binary |
| Standalone C++ test (no Python) | 45.2 GB/s | Proves NIXL API parity |

**How to reproduce**: The optimization investigation used scripts in
`test_plans/spcx_test/`. The final optimized runs use:

| Sbatch | What it tests |
|--------|---------------|
| `spcx_test/run_nixlbench_baseline.sbatch` | nixlbench 48 GB/s reference |
| `spcx_test/run_mini_bench.sbatch` | Block size / iodepth sweeps |
| `spcx_test/run_multi_handle_test.sbatch` | Multi-file sharding tests |
| `spcx_test/run_diagnose.sbatch` | Layer-by-layer I/O diagnostics |
| `spcx_test/run_vast_scaling.sbatch` | VAST scaling across nodes |
| `isr1_pre_test/run_storage_posix_optimized.sbatch` | Final optimized POSIX config |
| `isr1_pre_test/run_storage_posix_1node_test.sbatch` | Quick 1-node validation |

### 9.2 RDMA Performance (GAIA and ISR1-PRE)

**GAIA cluster** (Lustre, CPU memory only -- nvidia_peermem broken):

| Nodes | Memory | RDMA BW | Status |
|-------|--------|---------|--------|
| 1 (8 ranks) | CPU | 30-35 GB/s | Functional |
| 2-8 (16-64 ranks) | CPU | 37-42 GB/s | Functional |
| Any | CUDA | N/A | nvidia_peermem broken |

**ISR1-PRE cluster** (CUDA memory works):

| Nodes | Memory | RDMA BW | Status |
|-------|--------|---------|--------|
| 2 (16 ranks) | CUDA | 47.6 GB/s | Functional (exec-cetera) |

**How to reproduce**:

| Sbatch | Cluster | What it tests |
|--------|---------|---------------|
| `scripts/test_simple_2rank.sbatch` | GAIA | Minimal 2-rank RDMA validation |
| `scripts/test_simple_8rank.sbatch` | GAIA | 8-rank single-node RDMA |
| `scripts/test_simple_16rank.sbatch` | GAIA | 16-rank 2-node RDMA |
| `isr1_pre_test/run_rdma.sbatch` | ISR1-PRE | Pure RDMA transfers |
| `isr1_all_test/run_2nodes_8gpu.sbatch` | ISR1 | 2-node 8-GPU RDMA |

### 9.3 Storage + RDMA Combined (GAIA, Lustre)

POSIX backend with CPU memory, 1-8 nodes:

| Nodes | I/O Size | Read BW | Write BW | RDMA BW |
|-------|----------|---------|----------|---------|
| 1 | 64M | 14.7 GB/s | 8.2 GB/s | 38.7 GB/s |
| 1 | 1G | 15.7 GB/s | 4.7 GB/s | 35.5 GB/s |
| 2 | 64M | 15.5 GB/s | 6.7 GB/s | 40.6 GB/s |
| 2 | 1G | 19.0 GB/s | 4.8 GB/s | 39.6 GB/s |
| 4 | 256M | 20.0 GB/s | 6.4 GB/s | 41.5 GB/s |
| 8 | 256M | 17.9 GB/s | 4.9 GB/s | 39.7 GB/s |

RDMA: 90-100% of theoretical 42 GB/s max. Storage reads benefit from page
cache and parallelism.

**How to reproduce**:

| Sbatch | Cluster | What it tests |
|--------|---------|---------------|
| `scripts/comprehensive_test.sbatch` | GAIA | Full suite: all backends, I/O sizes, 1-8 nodes |
| `scripts/test_storage_8rank.sbatch` | GAIA | 8-rank POSIX storage + RDMA |
| `isr1_pre_test/run_rdma_storage.sbatch` | ISR1-PRE | 2-node GDS_MT + RDMA on VAST |
| `isr1_pre_test/run_12nodes_rdma_storage.sbatch` | ISR1-PRE | 12-node combined |
| `isr1_pre_test/run_12nodes_decode_posix.sbatch` | ISR1-PRE | 12-node prefill-decode POSIX |

### 9.4 Storage-Only GDS (GAIA, Lustre)

KVBench GDS vs GDSIO baseline (both GPU memory):

| I/O Size | GDSIO Read | KVBench Read | GDSIO Write | KVBench Write |
|----------|-----------|--------------|-------------|---------------|
| 1M | 0.36 GiB/s | 0.58 GiB/s (+61%) | 0.46 GiB/s | 0.22 GiB/s (-52%) |
| 16M | 2.17 GiB/s | 2.20 GiB/s (+1%) | 2.04 GiB/s | 0.90 GiB/s (-56%) |
| 64M | 4.12 GiB/s | 4.67 GiB/s (+13%) | 3.83 GiB/s | 1.89 GiB/s (-51%) |

Reads: no overhead. Writes: ~50% slower (file management overhead).

**How to reproduce**:

| Sbatch | Cluster | What it tests |
|--------|---------|---------------|
| `scripts/test_storage_8rank_gds.sbatch` | GAIA | 8-rank GDS storage-only |
| `isr1_pre_test/run_storage_gds.sbatch` | ISR1-PRE | GDS on VAST |
| `isr1_pre_test/run_1node_gds_test.sbatch` | ISR1-PRE | Single-node GDS validation |
| `isr1_pre_test/run_4nodes_decode_gds.sbatch` | ISR1-PRE | 4-node decode with GDS |

GDSIO baseline was run manually (not via sbatch):
```bash
gdsio -f /mnt/lustre/test.bin -d 0 -s 64M -i 64M -x 0 -I 0 -T 8  # Read
gdsio -f /mnt/lustre/test.bin -d 0 -s 64M -i 64M -x 1 -I 1 -T 8  # Write
```

### 9.5 KV Cache Workload Tests (ISR1-PRE, VAST)

Storage-only with realistic Llama 70B KV cache read patterns:

| Sbatch | Nodes | TP | Patterns | Config |
|--------|-------|----|----------|--------|
| `isr1_pre_test/run_8nodes_storage_tp8_264tp_vast.sbatch` | 8 | 8 | 264 | `storage_70b_8nodes_tp8_264tp_read_only/` |
| `isr1_pre_test/run_8nodes_storage_tp4_264tp_vast.sbatch` | 8 | 4 | 264 | `storage_70b_8nodes_tp4_264tp_read_only/` |
| `isr1_pre_test/run_8nodes_storage_tp8_512tp_vast.sbatch` | 8 | 8 | 512 | `storage_70b_8nodes_tp8_512tp_read_only/` |
| `isr1_pre_test/run_8nodes_storage_tp4_512tp_vast.sbatch` | 8 | 4 | 512 | `storage_70b_8nodes_tp4_512tp_read_only/` |
| `isr1_pre_test/run_12nodes_tp8_264tp_vast.sbatch` | 12 | 8 | 264 | `storage_70b_12nodes_tp8_264tp_read_only/` |
| `isr1_pre_test/run_12nodes_tp4_264tp_vast.sbatch` | 12 | 4 | 264 | `storage_70b_12nodes_tp4_264tp_read_only/` |

### 9.6 Saturation / Peak Throughput (ISR1-PRE, VAST)

Fixed-size reads to find per-node bandwidth ceiling:

| Sbatch | How to run |
|--------|-----------|
| `isr1_pre_test/run_sat_v2.sbatch` | Parameterized -- set env vars before sbatch |

```bash
export SAT_NAME="peak_1n8r_256m"
export SAT_NODES=1
export SAT_NTASKS_PER_NODE=8
export SAT_CONFIG="/path/to/saturation_v2/peak_1n8r_256m/metadata.yaml"
export SAT_NODELIST="hgx-isr1-pre-02"
sbatch --nodes=$SAT_NODES --ntasks-per-node=$SAT_NTASKS_PER_NODE \
       --nodelist=$SAT_NODELIST --job-name=$SAT_NAME \
       test_plans/isr1_pre_test/run_sat_v2.sbatch
```

Configs in `test/saturation_v2/` (regenerate with `create_configs.py`).

### 9.7 FIO Baselines (ISR1-PRE, VAST)

| Sbatch | What it tests |
|--------|---------------|
| `isr1_pre_test/run_8nodes_fio_simple.sbatch` | 8-node sequential read, simple config |
| `isr1_pre_test/run_8nodes_fio_read_vast.sbatch` | 8-node FIO read on VAST |
| `isr1_pre_test/run_8nodes_fio_isolated_agg_vast.sbatch` | FIO with per-node isolation |

### 9.8 Backend Comparison (GAIA, 1 node)

| I/O Size | Best Read Backend | Best Write Backend | Notes |
|----------|------------------|--------------------|-------|
| 1-16M | POSIX (CPU) | POSIX (CPU) | Page cache benefits |
| 64M+ | GDS (CUDA) | GDS (CUDA) | Direct GPU-storage faster |

### 9.9 Cluster Comparison

| Feature | GAIA | ISR1-PRE |
|---------|------|----------|
| nvidia_peermem | Loaded but broken | Working |
| GPU-Storage (GDS) | Works | Works |
| GPU-GPU RDMA (NIXL) | Fails | 47.6 GB/s |
| CPU-CPU RDMA | 35-42 GB/s | ~47 GB/s |
| Storage | Lustre (~5 GiB/s) | VAST (~48 GB/s/node) |

### 9.10 Detailed Results Files

For raw data and full test logs, see the untracked files in `test_plans/`:

| File | Contents |
|------|----------|
| `test_plans/RESULTS_SUMMARY.md` | Full GAIA test results with all configurations |
| `test_plans/STORAGE_SUMMARY.md` | Storage backend comparison (GDS vs POSIX vs FIO) |
| `test_plans/TEST_LOG.md` | Chronological test execution log with job IDs |
| `test_plans/GPUDIRECT_RDMA_TECHNICAL.md` | GPUDirect RDMA technical deep-dive |

---

## 10. Known Issues and Findings

### GPU RDMA on Certain Clusters

On GAIA cluster, `nvidia_peermem` is loaded but GPU memory registration for
InfiniBand fails with "Local protection error". This means:
- RDMA with `mem_type: cuda` fails
- CPU memory RDMA works fine (35-42 GB/s)
- GDS (GPU Direct Storage) works independently of peermem

On ISR1-PRE, GPU RDMA works inside containers with UCX.

### GDS Backend Limitations

- `GDS` backend fails for I/O > 1MB ("Error in setting up Batch")
- `GDS_MT` (multi-threaded) works for larger I/O sizes
- Recommendation: use `GDS` for the container-based runs on ISR1-PRE, which handles batching internally

### ETCD Namespace Collisions

Multi-node runs that share an ETCD instance can collide if they use the same
namespace. Always set a unique namespace:
```bash
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/unique_${SLURM_JOB_ID}"
```

### NFS Performance Characteristics

- Per-fd throughput ceiling: ~5-6 GB/s (depends on nconnect)
- Need multiple fds (files) to saturate node bandwidth
- `O_DIRECT` is required for consistent performance (avoids page cache pollution)
- `spread_reads` mount option helps distribute I/O across NFS connections

---

## 11. File Reference

### Repository Structure

```
benchmark/kvbench/
    main.py                                  # CLI entry point
    STORAGE_OPTIMIZATION.md                  # Storage I/O investigation and fixes
    HANDOVER.md                              # This file
    test/
        sequential_custom_traffic_perftest.py  # Main benchmark runner
        storage_backend.py                     # Storage backend (POSIX/GDS)
        traffic_pattern.py                     # Traffic pattern parser
        inference_workload_matgen.py           # KV cache config generator
        standalone_test.cpp                    # C++ baseline benchmark
        mini_storage_bench.py                  # Storage microbenchmark
        diagnose_io.py                         # I/O diagnostics
        verify_binding.py                      # Binding validation
        saturation_v2/                         # Peak throughput configs (untracked)
            create_configs.py                  # Config generator for saturation tests
        decode_centric_12nodes/                # KV cache workload configs (untracked)
            CONFIG_GENERATION_SUMMARY.md       # Generation parameters and commands
    test_plans/                                # Sbatch files and test configs (untracked)
        isr1_pre_test/                         # ISR1-PRE cluster sbatch files
            bind_8gpus.sh                      # GPU-NIC affinity script
            run_8nodes_storage_tp8_264tp_vast.sbatch
            run_8nodes_storage_tp4_264tp_vast.sbatch
            ...
        isr1_all_test/                         # Full test suite (GAIA cluster)
        spcx_test/                             # Storage optimization investigation sbatch
        scripts/                               # Generic/early test scripts
        RESULTS_SUMMARY.md                     # GAIA cluster test results
        TEST_LOG.md                            # Detailed test execution log
        STORAGE_SUMMARY.md                     # Storage test findings
```

### Untracked Directories

These directories contain generated configs and test artifacts. They are not
tracked in git because they are large and can be regenerated:

- `test/decode_centric_12nodes/storage_70b_*/` - KV cache configs (see Section 4 to regenerate)
- `test/saturation_v2/peak_*/` - Saturation configs (run `test/saturation_v2/create_configs.py`)
- `test_plans/` - All sbatch files and cluster-specific configs (115 files)

### Key Sbatch Files Quick Reference

| Test Type | Sbatch | Nodes | Backend |
|-----------|--------|-------|---------|
| Storage TP=8 264tp | `isr1_pre_test/run_8nodes_storage_tp8_264tp_vast.sbatch` | 8 | GDS |
| Storage TP=4 264tp | `isr1_pre_test/run_8nodes_storage_tp4_264tp_vast.sbatch` | 8 | GDS |
| Storage TP=8 512tp | `isr1_pre_test/run_8nodes_storage_tp8_512tp_vast.sbatch` | 8 | GDS |
| Storage TP=4 512tp | `isr1_pre_test/run_8nodes_storage_tp4_512tp_vast.sbatch` | 8 | GDS |
| POSIX optimized | `isr1_pre_test/run_storage_posix_optimized.sbatch` | 8 | POSIX |
| RDMA + Storage | `isr1_pre_test/run_rdma_storage.sbatch` | 2 | GDS_MT |
| RDMA only | `isr1_pre_test/run_rdma.sbatch` | 2 | N/A |
| 12-node storage | `isr1_pre_test/run_12nodes_tp8_264tp_vast.sbatch` | 12 | GDS |
| Saturation sweep | `isr1_pre_test/run_sat_v2.sbatch` | Param | POSIX |
| FIO baseline | `isr1_pre_test/run_8nodes_fio_simple.sbatch` | 8 | N/A |
| nixlbench baseline | `spcx_test/run_nixlbench_baseline.sbatch` | 1 | POSIX |
