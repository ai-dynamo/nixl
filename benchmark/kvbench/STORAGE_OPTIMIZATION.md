# kvbench Storage I/O Optimization

## Summary

kvbench's storage I/O was achieving ~10 GB/s per node on VAST. nixlbench achieves ~48 GB/s on the same hardware. Through systematic investigation of NIXL's POSIX backend internals, we identified the root causes and implemented optimizations that bring kvbench to **45-47 GB/s** aggregate per node.

## Root Cause Analysis

Three factors limited kvbench's storage throughput:

### 1. iodepth = 1

kvbench created ONE large descriptor per storage read (e.g., one 256MB descriptor). NIXL's POSIX backend sets `io_uring queue_depth = descriptor_count` ([posix_backend.cpp:167](../../src/plugins/posix/posix_backend.cpp)). One descriptor means one outstanding I/O at a time — the NFS client can't pipeline RPCs.

### 2. Single file per rank

kvbench used one file per rank. NFS distributes I/O across TCP connections per file descriptor. With `nconnect` and `spread_reads` mount options, each fd maps to a subset of connections. One fd = limited NFS parallelism.

### 3. AIO instead of URING

The default POSIX backend used Linux AIO (`libaio`). io_uring provides lower submission overhead for high-iodepth workloads.

## nixlbench Reference Configuration (48 GB/s)

The following nixlbench configuration achieves ~48 GB/s per node:

```bash
nixlbench --backend POSIX \
    --filepath /mnt/vast/data \
    --posix_api_type URING \
    --storage_enable_direct \
    --max_batch_size=256 \       # 256 descriptors per request = iodepth 256
    --start_batch_size=256 \
    --max_block_size=1048576 \   # 1MB per descriptor
    --start_block_size=1048576 \
    --op_type READ \
    --num_files=16 \             # 16 files, round-robin across fds
    --num_threads=16 \
    --total_buffer_size=858993459200 \
    --num_iter=3125 \
    --warmup_iter=16
```

Key parameters:
- `batch_size=256`: Each `createXferReq` gets 256 descriptor pairs, giving io_uring 256 SQEs per submission
- `block_size=1MB`: Each descriptor is 1MB (matches NFS rsize)
- `num_files=16`: Descriptors distributed round-robin across 16 fds in `exchangeIOV`
- `URING`: Uses io_uring instead of libaio

## Architecture of Changes

### Before: ~10 GB/s per node

A rank needs to read 256MB from VAST. kvbench creates **one descriptor** for the full 256MB:

```
rank 0 reads 256MB:

    storage_backend.get_read_handle()
        → 1 descriptor: (offset=0, size=256MB, fd=file_0)
        → createXferReq with 1 descriptor pair
        → POSIX backend creates io_uring with queue_depth=1
        → io_uring_submit: 1 SQE (one 256MB read)
        → NFS sends 1 RPC, waits, sends next, waits...
        → ~10 GB/s
```

The bottleneck: io_uring has only 1 outstanding I/O. The NFS client can only use 1 TCP connection at a time.

### After: ~28 GB/s per rank, ~38+ GB/s per node

Same read, but now split into **N x 1MB descriptors** (N = read_size / 1MB):

```
rank 0 reads 256MB (with --storage-block-size 1M):

    storage_backend.get_read_handle()  (via _create_chunked_descs)
        → 256 descriptors: (0,1MB,fd), (1MB,1MB,fd), (2MB,1MB,fd), ...
        → createXferReq with 256 descriptor pairs
        → POSIX backend creates io_uring with queue_depth=256
        → io_uring_submit: 256 SQEs (256 concurrent 1MB reads)
        → NFS pipelines 256 RPCs across multiple TCP connections
        → ~11 GB/s per rank

rank 0 reads 1GB (with --storage-block-size 1M):

    → 1024 descriptors (1024 x 1MB), queue_depth=1024
    → ~11 GB/s per rank (same per-rank BW, just takes longer)
```

The descriptor count scales with read size: a 2GB read creates 2048 descriptors.
Larger reads don't get faster per-rank — they get higher iodepth but hit the
same NFS per-fd throughput ceiling. Multiple ranks running in parallel are
needed to saturate the node's 2x 25 GB/s NICs.

The key insight: NIXL's POSIX backend sets **io_uring queue depth = number of descriptors**. More descriptors = more concurrent I/O = higher per-rank throughput (up to ~11 GB/s with O_DIRECT).

## Changes by File

### Core Production Changes

#### `src/bindings/python/nixl_bindings.cpp`

Two new C++ binding methods added to the NIXL Python bindings:

- **`parallelPostXferReqs(reqhs, num_threads)`**: Submits a list of NIXL transfer request handles concurrently from C++ threads, bypassing the Python GIL. Each thread posts one request and polls until completion. Used when `num_handles > 1` to submit multiple shard reads in parallel.

- **`parallelStorageTransfer(file_desc_lists, agent_name, backends, num_iters, warmup_iters, num_threads)`**: Benchmark-oriented function replicating nixlbench's exact `execTransfer` pattern — OMP parallel section, per-thread buffer allocation via `posix_memalign`, `createXferReq` inside OMP, separate warmup and timed passes. Used for standalone performance validation, not called by kvbench directly.

#### `benchmark/kvbench/test/storage_backend.py`

`FilesystemBackend` redesigned with two performance optimizations:

- **Block splitting** (`_create_chunked_descs`): When `block_size > 0`, splits a single large read into many 1MB descriptors within one NIXL request. This raises io_uring queue depth from 1 to N (e.g., 256 for a 256MB read). The NFS client can then pipeline N concurrent RPCs. Primary optimization: ~10 GB/s to ~28 GB/s.

- **Multi-file sharding** (`prepare()` with `_num_handles > 1`): Creates N shard files per rank, each with its own fd. `get_read_handles()` / `get_write_handles()` return N separate NIXL transfer handles (one per shard). Submitted concurrently via `parallelPostXferReqs`. Each fd gets its own NFS connection, enabling NFS-level parallelism.

- Extracted `_open_and_register_file()` helper for centralized fd management.

#### `benchmark/kvbench/test/sequential_custom_traffic_perftest.py`

- Added `storage_num_handles` parameter (flows from CLI to storage backend).
- `_prepare_storage_xfer()` supports multi-handle mode: when `num_handles > 1`, returns a list of `StorageXferHandle` objects for concurrent submission.
- Fixed pre-existing `IndexError` bug in `_print_iteration_results()` where rank indices from the YAML config could exceed `world_size` when running with fewer ranks than the config specifies.

#### `benchmark/kvbench/main.py`

- Added `--storage-num-handles` CLI option (default: 1). Controls how many concurrent transfer handles per storage operation.

### Diagnostic and Test Files

| File | Purpose |
|------|---------|
| `test/standalone_test.cpp` | Pure C++ program calling same NIXL APIs, proves 45 GB/s without Python (vs 39 GB/s with Python = 13% overhead) |
| `test/verify_binding.py` | Tests `parallelStorageTransfer` binding with iodepth sweeps and fd distribution patterns |
| `test/mini_storage_bench.py` | Microbenchmark isolating POSIX backend behavior: block sizes, handle counts, threading |
| `test/diagnose_io.py` | Layer-by-layer I/O diagnostic: raw OS, NIXL single, NIXL multi-handle, NIXL parallel |
| `test/run_nixlbench_tests.sh` | nixlbench parameter sweeps (AIO vs URING, O_DIRECT, thread/file counts) |
| `test/run_final_compare.sh` | Side-by-side nixlbench vs our binding on same node |
| `test/run_standalone_compare.sh` | Three-way comparison: nixlbench vs standalone C++ vs Python binding |
| `test_plans/isr1_pre_test/run_storage_posix_optimized.sbatch` | 8-node optimized POSIX sbatch |
| `test_plans/isr1_pre_test/run_storage_posix_1node_test.sbatch` | 1-node quick validation sbatch |

## Usage

### Optimized POSIX storage (recommended for VAST)

```bash
python3 -m main sequential-ct-perftest config.yaml \
    --storage-backend POSIX \
    --storage-path /mnt/vast/storage \
    --storage-direct-io \
    --storage-block-size 1M \
    --storage-posix-api uring \
    --storage-num-handles 16
```

### Legacy mode (default, backwards compatible)

```bash
python3 -m main sequential-ct-perftest config.yaml \
    --storage-backend POSIX \
    --storage-path /mnt/vast/storage
```

### GDS mode (unchanged)

```bash
python3 -m main sequential-ct-perftest config.yaml \
    --storage-backend GDS \
    --storage-path /mnt/vast/storage \
    --storage-direct-io
```

## Performance Results

| Configuration | Per-Node BW | Notes |
|---|---|---|
| kvbench baseline (iodepth=1, AIO, 1 file) | ~10 GB/s | Original behavior |
| + `--storage-block-size 1M` | ~28 GB/s | High iodepth, single fd |
| + `--storage-posix-api uring` | ~28 GB/s | Marginal improvement over AIO |
| + `--storage-num-handles 16` | ~35 GB/s | Multi-fd NFS parallelism |
| kvbench workload aggregate (4 ranks) | **45-47 GB/s** | All optimizations combined |
| nixlbench reference | 48.5 GB/s | Standalone C++ binary |
| Standalone C++ test (no Python) | 45.2 GB/s | Proves NIXL API parity |

## Key Technical Findings

### NIXL POSIX Backend Internals

1. **Queue depth = descriptor count**: Each `createXferReq` creates an io_uring ring sized to the number of descriptors. One descriptor = iodepth 1. The fix: split into many 1MB descriptors.

2. **Single-fd-per-request constraint**: NIXL's `createXferReq` → `populate()` requires all FILE descriptors in a single request to belong to the same registered region. Multiple fds from different files in one request causes `NIXL_ERR_NOT_FOUND`. Multi-fd parallelism requires separate requests submitted concurrently.

3. **`registerMem` is a no-op for POSIX**: The POSIX backend's `registerMem` only checks type support — buffer size has zero effect on performance. Only descriptor count in `createXferReq` matters.

4. **io_uring batch submission**: All descriptors in a request are submitted in a single `io_uring_submit()` call ([uring_queue.cpp:114](../../src/plugins/posix/uring_queue.cpp)). No special flags (SQPOLL, IOPOLL) are used.

### Python Process Overhead

Standalone C++ achieves 45.2 GB/s vs Python binding's 39.1 GB/s (~13% overhead). This comes from the Python process environment (memory layout, NUMA, signal handling). For kvbench's workload pattern (one storage read per TP iteration, not a tight benchmark loop), this overhead is negligible — the aggregate workload BW reaches 45-47 GB/s.

### nixlbench's Round-Robin Pattern

nixlbench's `exchangeIOV` distributes each thread's 256 descriptors round-robin across all 16 file descriptors. This puts 16 different fds in a single `createXferReq`, which works because nixlbench registers all files in a single batch. kvbench registers files individually, so multi-fd requests fail. The workaround: use separate requests per fd, submitted concurrently via `parallelPostXferReqs`.
