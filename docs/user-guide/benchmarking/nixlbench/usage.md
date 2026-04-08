---
title: NIXLBench Usage and Troubleshooting
description: How to run NIXLBench benchmarks and troubleshoot common issues.
---

This page covers running NIXLBench benchmarks end-to-end, including etcd coordination, the four communication patterns, storage backend examples, and essential CLI options. For installation prerequisites, see [Building NIXLBench](./build.md).

## etcd Coordination

NIXLBench workers coordinate through [etcd](../../etcd-metadata-exchange.md) for discovery and synchronization. Network backends ([UCX](../../backends/ucx.md), [DOCA GPUNetIO](../../backends/gpunetio.md), [Mooncake](../../backends/mooncake.md), [Libfabric](../../backends/libfabric.md)) and all multi-node setups require etcd. Storage backends ([GPUDirect Storage](../../backends/gds.md), [GPUDirect Storage MT](../../backends/gds-mt.md), [POSIX](../../backends/posix.md), [HF3FS](../../backends/hf3fs.md), [OBJ](../../backends/obj.md), [GUSLI](../../backends/gusli.md)) can run without etcd when launched as a single instance.

Start an etcd server with Docker:

```bash
docker run -d --name etcd-server \
  -p 2379:2379 -p 2380:2380 \
  quay.io/coreos/etcd:v3.5.18 \
  /usr/local/bin/etcd \
  --data-dir=/etcd-data \
  --listen-client-urls=http://0.0.0.0:2379 \
  --advertise-client-urls=http://0.0.0.0:2379 \
  --listen-peer-urls=http://0.0.0.0:2380 \
  --initial-advertise-peer-urls=http://0.0.0.0:2380 \
  --initial-cluster=default=http://0.0.0.0:2380
```

<Warning>
All NIXLBench workers in a benchmark group must connect to etcd within 60 seconds of the first worker joining. Workers that miss this window cause the barrier to fail and the benchmark to abort. For etcd setup and configuration details, see [Metadata Exchange with etcd](../../etcd-metadata-exchange.md).
</Warning>

<Note>
Support for running NIXLBench without etcd when using multiple workers is currently in progress. For now, etcd is required for all multi-worker benchmarks.
</Note>

## Communication Patterns

NIXLBench supports four communication patterns selected with the `--scheme` flag. All examples below use the UCX backend with VRAM transfers.

### Pairwise

Pairwise is the default pattern. Data transfers between matched pairs of initiators and targets, making it ideal for point-to-point throughput measurement.

On host 1 (initiator):

```bash
nixlbench --etcd_endpoints http://etcd-server:2379 \
  --backend UCX \
  --initiator_seg_type VRAM \
  --target_seg_type VRAM \
  --scheme pairwise
```

On host 2 (target):

```bash
nixlbench --etcd_endpoints http://etcd-server:2379 \
  --backend UCX \
  --initiator_seg_type VRAM \
  --target_seg_type VRAM \
  --scheme pairwise
```

### Many-to-One

Multiple initiators send data to a single target. This pattern measures how a target handles concurrent incoming transfers from several sources.

```bash
nixlbench --etcd_endpoints http://etcd-server:2379 \
  --backend UCX \
  --initiator_seg_type VRAM \
  --target_seg_type VRAM \
  --scheme manytoone
```

Launch one target worker and multiple initiator workers, all pointing to the same etcd server.

### One-to-Many

A single initiator sends data to multiple targets. This pattern is useful for measuring fan-out performance such as broadcast or scatter workloads.

```bash
nixlbench --etcd_endpoints http://etcd-server:2379 \
  --backend UCX \
  --initiator_seg_type VRAM \
  --target_seg_type VRAM \
  --scheme onetomany
```

Launch one initiator worker and multiple target workers, all pointing to the same etcd server.

### TP (Tensor Parallel)

All-to-all exchange where every worker communicates with every other worker. This pattern simulates tensor-parallel distributed training workloads.

```bash
nixlbench --etcd_endpoints http://etcd-server:2379 \
  --backend UCX \
  --initiator_seg_type VRAM \
  --target_seg_type VRAM \
  --scheme tp
```

Launch two or more workers, all pointing to the same etcd server. Each worker acts as both initiator and target.

## Storage Backend Examples

Storage backends benchmark file and object I/O operations. They can run without etcd when launched as a single instance.

### GPUDirect Storage (GDS)

Run a single-instance GDS benchmark with direct I/O. For backend-specific flags, see the [GPUDirect Storage](../../backends/gds.md) page.

```bash
nixlbench --backend GDS --filepath /mnt/storage/testfile --storage_enable_direct
```

### OBJ (S3)

Run an S3 object storage benchmark using CLI flags for credentials. For backend-specific flags, see the [OBJ](../../backends/obj.md) page.

```bash
nixlbench --backend OBJ \
  --obj_access_key $AWS_ACCESS_KEY_ID \
  --obj_secret_key $AWS_SECRET_ACCESS_KEY \
  --obj_region us-east-1 \
  --obj_bucket_name my-bucket
```

For backend-specific options not listed on this page, see the corresponding backend page in the [User Guide](../../backends/index.md).

## CLI Options

### Core Configuration

| Flag | Description | Default |
|------|-------------|---------|
| `--config_file` | Configuration file in TOML format | None |
| `--runtime_type` | Runtime coordination type | `ETCD` |
| `--worker_type` | Worker transfer engine (`nixl`, `nvshmem`) | `nixl` |
| `--backend` | Communication backend (UCX, GDS, GDS_MT, POSIX, GPUNETIO, Mooncake, HF3FS, OBJ, GUSLI) | `UCX` |
| `--benchmark_group` | Group name for parallel runs | `default` |
| `--etcd_endpoints` | etcd server URL for coordination | `http://localhost:2379` |

### Memory and Transfer Configuration

| Flag | Description | Default |
|------|-------------|---------|
| `--initiator_seg_type` | Initiator memory segment type (DRAM, VRAM) | `DRAM` |
| `--target_seg_type` | Target memory segment type (DRAM, VRAM) | `DRAM` |
| `--scheme` | Communication pattern (pairwise, manytoone, onetomany, tp) | `pairwise` |
| `--mode` | Process mode: SG (single GPU) or MG (multi GPU) | `SG` |
| `--op_type` | Operation type (READ, WRITE) | `WRITE` |
| `--check_consistency` | Enable data consistency checking | disabled |
| `--total_buffer_size` | Total buffer size per process | `8GiB` |
| `--start_block_size` | Starting block size | `4KiB` |
| `--max_block_size` | Maximum block size | `64MiB` |
| `--start_batch_size` | Starting batch size | `1` |
| `--max_batch_size` | Maximum batch size | `1` |
| `--recreate_xfer` | Recreate transfer handle per iteration | disabled |

NIXLBench accepts a TOML configuration file via `--config_file` where CLI parameter names are used as keys in the global scope. For example, `backend="UCX"` in the config file is equivalent to `--backend UCX` on the command line. When a parameter appears in both the config file and on the command line, the command-line value takes precedence.

The `--worker_type nvshmem` option selects the NVSHMEM worker for GPU-only VRAM-to-VRAM transfers. NVSHMEM workers require `--initiator_seg_type VRAM` and `--target_seg_type VRAM`.

## Reading Benchmark Output

After each run, NIXLBench prints a results table with one row per block-size and batch-size combination. Block sizes sweep from `--start_block_size` to `--max_block_size`, doubling each step. For each block size, batch sizes sweep from `--start_batch_size` to `--max_batch_size` in the same way.

| Column | Unit | Description |
|--------|------|-------------|
| Block Size (B) | Bytes | Transfer block size for this row |
| Batch Size | Count | Number of transfers per batch |
| B/W (GB/Sec) | GB/s | Per-worker throughput (total data transferred divided by elapsed time) |
| Aggregate B/W (GB/Sec) | GB/s | Sum of all workers' throughput (multi-worker pairwise only) |
| Network Util (%) | Percent | Aggregate bandwidth as a percentage of theoretical peak (multi-worker pairwise only) |
| Avg Lat. (us) | Microseconds | Average latency per individual transfer operation |
| Avg Prep (us) | Microseconds | Average time for the prepare phase (buffer registration and handle setup) |
| P99 Prep (us) | Microseconds | 99th percentile prepare phase duration |
| Avg Post (us) | Microseconds | Average time for the post phase (completion checking and cleanup) |
| P99 Post (us) | Microseconds | 99th percentile post phase duration |
| Avg Tx (us) | Microseconds | Average time for the transfer phase (actual data movement) |
| P99 Tx (us) | Microseconds | 99th percentile transfer phase duration |

<Note>
When running pairwise benchmarks with more than two workers, NIXLBench adds the Aggregate B/W and Network Util columns. These columns do not appear for other communication patterns or two-worker pairwise runs.
</Note>

The latency columns break down each transfer into three phases. Prep measures buffer registration and transfer handle setup overhead. Tx measures the actual data movement time. Post measures completion checking and cleanup. Together, Avg Lat. reflects the end-to-end latency across all three phases.

## Troubleshooting

### etcd Connection Failures

**Symptoms:** Workers fail to join the benchmark group, barrier timeout errors, or "connection refused" messages.

**Resolution:**

Verify etcd is running and reachable:

```bash
ETCDCTL_API=3 etcdctl endpoint health --endpoints=http://etcd-server:2379
```

If a previous NIXLBench run failed or was interrupted, stale keys may prevent new runs. Clean them up:

```bash
ETCDCTL_API=3 etcdctl del "xferbench" --prefix=true
```

Confirm that all workers use the same `--etcd_endpoints` value and that the etcd server is accessible from every host in the benchmark.

### Build Failures

**Symptoms:** Compilation errors during the native build, missing header files, or linker errors.

**Resolution:**

For UCX builds missing RDMA libraries:

```bash
sudo apt-get reinstall -y libibverbs-dev librdmacm-dev rdma-core
```

For etcd-cpp-api build errors related to cpprestsdk or protobuf:

```bash
sudo apt-get install -y libcpprest-dev
sudo apt-get install -y libprotobuf-dev protobuf-compiler
```

For Docker build failures, clear the cache and rebuild:

```bash
docker system prune -a
```

For full build instructions, see [Building NIXLBench](./build.md).

### CUDA / GPU Not Found

**Symptoms:** CUDA-related errors at launch, `nvcc` not found, or GPU detection failures.

**Resolution:**

Ensure CUDA is in your environment:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Verify the installation:

```bash
nvcc --version
nvidia-smi
```

### Backend Library Missing

**Symptoms:** "library not found" errors at runtime when launching NIXLBench.

**Resolution:**

Update the shared library cache:

```bash
sudo ldconfig
```

Check that all required libraries are resolved:

```bash
ldd /usr/local/nixlbench/bin/nixlbench
```

If libraries are installed in non-standard paths, add them to `LD_LIBRARY_PATH` before running NIXLBench.
