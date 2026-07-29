# NIXL KVBench
A comprehensive utility for generating NIXL Bench commands that test KVCache transfer across various LLM architectures and access patterns, plus custom traffic performance testing for asymmetric traffic flows.

## Table of Contents
- [Overview](#overview)
  - [Quickstart (end-to-end)](#quickstart-end-to-end)
- [Supported LLM Architectures](#supported-llm-architectures)
- [Building](#building)
  - [Docker](#docker)
  - [Python](#python)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
- [Command Line Arguments](#command-line-arguments)
  - [Common Arguments](#common-arguments)
  - [CLI Override Arguments](#cli-override-arguments)
  - [Plan Command Arguments](#plan-command-arguments)
  - [Shared Benchmark Arguments](#shared-benchmark-arguments)
  - [CTP Command Arguments](#ctp-command-arguments)
- [Command Descriptions](#command-descriptions)
  - [KVBench Commands](#kvbench-commands)
  - [CTP Commands](#ctp-commands)
- [Examples](#examples)
  - [KVBench Examples](#kvbench-examples)
  - [CTP Examples](#ctp-examples)
- [Developer Guides](#developer-guides)

## Overview

NIXL KVBench provides two main categories of functionality:

1. **KVBench Commands**: Test KV cache transfers across various LLM architectures with different access patterns (block and layer approaches)
2. **CTP Commands**: Custom Traffic Performance Testing for measuring asymmetric traffic patterns using transfer matrices

### Design Philosophy

KVBench is a **generic multi-rank benchmark framework** that separates concerns:

| Component | Role | Example |
|-----------|------|---------|
| **Workload Generators** | Create YAML configs for specific applications | `matgen` (LLM workloads) |
| **KVBench Framework** | Execute any valid YAML with RDMA + Storage + Compute | `sequential-ct-perftest` |
| **Storage Backends** | Pluggable I/O implementations | POSIX, GDS, GDS_MT |

This design allows:
- **Portable configs**: YAML specifies "what" (bytes), not "where" (paths)
- **Multiple workload types**: LLM inference, database patterns, custom scenarios
- **Backend flexibility**: Choose storage backend at runtime via CLI

KVBench simulates real application workloads combining:
- Storage I/O (read/write per rank)
- Network transfers (RDMA)
- Compute simulation (sleep), before and after the RDMA transfer

### Execution Flow Per Traffic Pattern

```text
Each Rank (per traffic pattern):
  ┌─────────────────────────────────────┐
  │ 1. STORAGE READ (blocking)          │  Read the cached KV prefix from file
  │ 2. PREFILL COMPUTE (sleep)          │  Compute the prefix that was not cached
  │ 3. RDMA TRANSFER (blocking)         │  Send the KV cache to the decode ranks
  │ 4. DECODE COMPUTE (sleep)           │  Decode step on the received KV cache
  │ 5. STORAGE WRITE (blocking)         │  Write the new KV cache to file
  └─────────────────────────────────────┘
```

Each operation is timed independently.

### Quickstart (end-to-end)

Five steps: start etcd, export the etcd variables, generate a workload, launch the ranks, read the table.

#### Step 1 — Start an etcd server

`sequential-ct-perftest` uses etcd to coordinate the ranks (barrier, allgather, allgather of results). It does **not** use the `--etcd-endpoints` flag from [Shared Benchmark Arguments](#shared-benchmark-arguments) — that flag belongs to the `plan` and `profile` commands, which drive `nixlbench`. The CTP commands read environment variables instead.

Start one etcd server that every node can reach:

```bash
docker run -d --name kvbench-etcd --network host quay.io/coreos/etcd:v3.5.17 \
    /usr/local/bin/etcd \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://0.0.0.0:2379
```

#### Step 2 — Export the etcd variables

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `NIXL_ETCD_ENDPOINTS` | `http://localhost:2379` | etcd server, format `[http://]host[:port]`. `localhost` only works for a single-node run. |
| `NIXL_ETCD_NAMESPACE` | `/nixl/kvbench` | Key prefix for this run. Rank 0 deletes the whole prefix at startup, so two runs that share a prefix erase each other. |

```bash
export NIXL_ETCD_ENDPOINTS="http://<etcd-host>:2379"
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/$(uuidgen)"
```

Every rank must see the **same** endpoint and the **same** namespace. Export both before the launcher, never inside the per-rank command: `$(uuidgen)` evaluated once per rank would give each rank a different prefix, and the ranks would never find each other.

If `NIXL_ETCD_NAMESPACE` is not set, kvbench appends `run-<token>` to the default prefix, where the token comes from `SLURM_JOB_ID`, `SLURM_JOBID` or `PMIX_NAMESPACE`. If none of them exist it logs a warning and every run shares `/nixl/kvbench`. Setting the variable yourself is safer.

Rank and world size also come from the environment: `SLURM_PROCID`/`SLURM_NTASKS`, `OMPI_COMM_WORLD_RANK`/`OMPI_COMM_WORLD_SIZE`, or `RANK`/`WORLD_SIZE`. One of those pairs must be set, otherwise the run fails at startup.

#### Step 3 — Generate the workload

`inference_workload_matgen.py` writes `metadata.yaml` and one `tps/tp_<idx>.tp` file per traffic pattern into `--results-dir`:

```bash
python test/inference_workload_matgen.py generate \
    --model llama-405b \
    --num-prefill-nodes 1 \
    --num-decode-nodes 1 \
    --prefill-tp 8 \
    --decode-tp 8 \
    --num-user-requests 10 \
    --prefix-hit-rate 0.75 \
    --results-dir ./workload
```

`metadata.yaml` is the file you pass to the runner. It holds `iters`, `isolation_iters` and the `traffic_patterns` list. Each pattern points at its `tp_file` and carries `sleep_before_launch_sec` (prefill compute), optionally `decode_compute_sec` (decode compute), plus free-form `metadata`. The `.tp` file holds the `[rdma]` matrix and the per-rank `[read]` / `[write]` sizes. Full field list: [Configuration Files](#configuration-files) and [docs/ct-perftest.md](docs/ct-perftest.md).

The generated files fix the number of ranks: `(--num-prefill-nodes + --num-decode-nodes) * --ppn`, with `--ppn` defaulting to 8. The command above gives 16 ranks, so the run must start exactly 16 processes.

#### Step 4 — Launch

The config path is the positional argument of the command. Please read the note about `CUDA_VISIBLE_DEVICES` in the [CT Perftest section](#ct-perftest): one process per GPU, and each process must pin its own device.

**Single node**, 8 ranks. Regenerate step 3 with `--num-prefill-nodes 1 --num-decode-nodes 0 --storage-only` to get 8 ranks and no RDMA:

```bash
export NIXL_ETCD_ENDPOINTS="http://localhost:2379"
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/$(uuidgen)"
export WORLD_SIZE=8
for r in $(seq 0 7); do
  RANK=$r CUDA_VISIBLE_DEVICES=$r \
    python main.py sequential-ct-perftest ./workload/metadata.yaml \
      --storage-backend POSIX \
      --storage-path /tmp/kvbench_storage \
      --json-output-path ./results.json &
done
wait
```

**Multi-node with Slurm**, 2 nodes x 8 ranks = the 16 ranks of step 3:

```bash
export NIXL_ETCD_ENDPOINTS="http://<etcd-host>:2379"
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/$(uuidgen)"

srun --partition=<partition> \
     --nodes=2 \
     --ntasks-per-node=8 \
     --export=ALL \
     bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID && \
       python main.py sequential-ct-perftest ./workload/metadata.yaml \
         --storage-backend POSIX \
         --storage-path /mnt/shared/kvbench_storage \
         --json-output-path ./results.json'
```

- Use single quotes around the `bash -c` body. `$SLURM_LOCALID` must expand on each rank, not in the submitting shell.
- `--export=ALL` passes both `NIXL_ETCD_*` variables to every rank.
- Each rank uses its own file `<storage-path>/tp_<pattern>/rank_<rank>.bin`, so point `--storage-path` at the filesystem you want to measure.
- More options: [Running CTP Tests](#running-ctp-tests).

#### Step 5 — Read the output

Rank 0 prints one table per iteration. There are 18 columns in three groups: RDMA, storage read, storage write. Sizes are GB, latencies are ms, bandwidth is GB/s. A `-` means the pattern does not use that operation.

| Column | Meaning |
| ------ | ------- |
| `RDMA (GB)` | Total bytes moved by the RDMA matrix of this pattern. |
| `RDMA (ms)` | Workload latency: last rank end minus first rank start, all ranks running together. |
| `Iso p50` / `Iso p90` | Isolated RDMA latency, p50 / p90 over `--isolation-iters` iterations, worst rank. |
| `RDMA BW` | Workload bandwidth: mean over the sender ranks of (bytes that rank sent / that rank's elapsed time). |
| `Iso BW` | Isolated RDMA bandwidth of the slowest sender (min across ranks = the bottleneck). |
| `Read (GB)` | Total bytes read from storage by all ranks in this pattern. |
| `Read (ms)` | Workload read latency of the slowest rank, all ranks reading together. |
| `Rd p50` / `Rd p90` | Isolated read latency, p50 / p90 over `--isolation-iters` iterations. |
| `Read BW` | Workload read bandwidth: `Read (GB)` / `Read (ms)`, so all ranks summed. |
| `Iso Rd BW` | Isolated read bandwidth of the slowest rank (min across ranks). |
| `Write (GB)` | Total bytes written to storage by all ranks in this pattern. |
| `Write (ms)` | Workload write latency of the slowest rank, all ranks writing together. |
| `Wr p50` / `Wr p90` | Isolated write latency, p50 / p90 over `--isolation-iters` iterations. |
| `Write BW` | Workload write bandwidth: `Write (GB)` / `Write (ms)`, so all ranks summed. |
| `Iso Wr BW` | Isolated write bandwidth of the slowest rank (min across ranks). |

**Isolated vs workload.** Isolated numbers are measured with nothing else running:

- Isolated RDMA runs one traffic pattern alone — only its senders, no storage I/O in flight, no other pattern at the same time.
- Isolated storage is stricter: only the first rank that has the operation runs it, every other rank waits. It is one client against the storage target, with no contention at all.

Workload numbers are measured with all ranks running the full pipeline together (read, RDMA, write). The gap between the two is the cost of contention — on the fabric for the RDMA columns, on the storage target for the read and write columns. That gap is the point of this benchmark. `Iso p50` close to `RDMA (ms)` means the fabric absorbs the load; a large gap means it does not.

Compare latency against latency: `RDMA (ms)` vs `Iso p50`, `Read (ms)` vs `Rd p50`, `Write (ms)` vs `Wr p50`. Those are the same measurement under different load. Be careful with the bandwidth columns, they have different scopes: `Read BW` and `Write BW` sum all ranks, while `Iso Rd BW` and `Iso Wr BW` are single-rank rates.

Add `--json-output-path ./results.json` for machine-readable output (rank 0 writes the file). It has two top-level keys, `metadata` and `iterations_results`. Each pattern entry carries `size`, `latency`, `mean_bw`, `num_senders`, `storage_read_max_ms`, `storage_write_max_ms`, `storage_read_size_gb`, `storage_write_size_gb`, and the `isolated_{rdma,read,write}_{p50,p90,p99,min,max}_ms` keys.

## Supported LLM Architectures
- DeepSeek R1
- LLama 3.1
- and more

## Building

### Docker
```bash
git clone https://github.com/ai-dynamo/nixl.git
export NIXL_SRC=/path/to/nixl/
cd nixl/benchmark/nixlbench/contrib
./build.sh --nixl $NIXL_SRC
```

### Python
```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl/benchmark/kvbench
python3 -m venv venv
source venv/bin/activate
pip install uv
uv sync --active
```

## Usage

### Basic Usage
```bash
python main.py --help
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  KVBench - NIXL Performance Testing CLI

Options:
  --debug / --no-debug  Enable debug logging
  --help                Show this message and exit.

Commands:
  ct-perftest             Run custom traffic performance test using...
  kvcache                 Display kvcache information
  plan                    Display the recommended configuration for...
  profile                 Run nixlbench
  sequential-ct-perftest  Run sequential custom traffic performance test...
```

## Command Line Arguments

### Common Arguments

These arguments are shared across KVBench commands (plan, kvcache, profile, io-size):

| Argument | Description |
| -------- | ----------- |
| `--model` | Path to a model architecture config YAML file |
| `--model_config` | Path to a model config YAML file |
| `--model_configs` | Path to multiple model config YAML files (supports glob patterns) |

### CLI Override Arguments

These arguments can be used to override values in model config files:

| Argument | Description |
| -------- | ----------- |
| `--pp` | Pipeline parallelism size |
| `--tp` | Tensor parallelism size |
| `--isl` | Input sequence length |
| `--osl` | Output sequence length |
| `--num_requests` | Number of requests |
| `--page_size` | Page size |
| `--access_pattern` | Access pattern [block, layer] |

### Plan Command Arguments

Specific to the `plan` command:

| Argument | Description |
| -------- | ----------- |
| `--format` | Output format of the nixl command [text, json, csv] (default: text) |

### Shared Benchmark Arguments

These arguments are used by both `plan` and `profile` commands:

| Argument | Description |
| -------- | ----------- |
| `--source` | Source of the nixl descriptors [file, memory, gpu] (default: file) |
| `--destination` | Destination of the nixl descriptors [file, memory, gpu] (default: memory) |
| `--backend` | Communication backend [UCX, GDS, GDS_MT, POSIX, GPUNETIO, Mooncake, HF3FS, OBJ] (default: UCX) |
| `--worker_type` | Worker to use to transfer data [nixl, nvshmem] (default: nixl) |
| `--initiator_seg_type` | Memory segment type for initiator [DRAM, VRAM] (default: DRAM) |
| `--target_seg_type` | Memory segment type for target [DRAM, VRAM] (default: DRAM) |
| `--scheme` | Communication scheme [pairwise, manytoone, onetomany, tp] (default: pairwise) |
| `--mode` | Process mode [SG (Single GPU per proc), MG (Multi GPU per proc)] (default: SG) |
| `--op_type` | Operation type [READ, WRITE] (default: WRITE) |
| `--check_consistency` | Enable consistency checking |
| `--total_buffer_size` | Total buffer size in bytes (default: 8GiB) |
| `--recreate_xfer` | Recreate xfer for every iteration (default: false for all backends, true for GUSLI) |
| `--start_block_size` | Starting block size in bytes (default: 4KiB) |
| `--max_block_size` | Maximum block size in bytes (default: 64MiB) |
| `--start_batch_size` | Starting batch size (default: 1) |
| `--max_batch_size` | Maximum batch size (default: 1) |
| `--num_iter` | Number of iterations (default: 1000) |
| `--warmup_iter` | Number of warmup iterations (default: 100) |
| `--num_threads` | Number of threads used by benchmark (default: 1) |
| `--num_initiator_dev` | Number of devices in initiator processes (default: 1) |
| `--num_target_dev` | Number of devices in target processes (default: 1) |
| `--enable_pt` | Enable progress thread |
| `--progress_threads` |  Number of progress threads (default: 0) |
| `--device_list` | Comma-separated device names (default: all) |
| `--runtime_type` | Type of runtime to use [ETCD] (default: ETCD) |
| `--etcd-endpoints` | ETCD server URL for coordination (default: http://localhost:2379) |
| `--storage_enable_direct` | Enable direct I/O for storage operations |
| `--filepath` | File path for storage operations |
| `--enable_vmm` | Enable VMM memory allocation when DRAM is requested |

### CTP Command Arguments

Specific to CTP (Custom Traffic Performance) commands:

| Argument | Description |
| -------- | ----------- |
| `config_file` | Path to YAML configuration file (required) |
| `--verify-buffers / --no-verify-buffers` | Verify buffer contents after transfer (default: False) |
| `--print-recv-buffers / --no-print-recv-buffers` | Print received buffer contents (default: False) |
| `--json-output-path` | Path to save JSON output (sequential-ct-perftest only) |
| `--storage-backend` | Storage backend: POSIX, GDS, GDS_MT (default: POSIX) |
| `--storage-path` | Base path for storage files (default: `<config_dir>/storage`) |
| `--storage-direct-io / --no-storage-direct-io` | Enable O_DIRECT for storage I/O (auto-enabled for GDS and GDS_MT) |
| `--storage-block-size` | Split storage I/O into fixed-size blocks (e.g. `1M`) so io_uring/AIO can pipeline requests. `0` = no splitting. |
| `--storage-posix-api` | POSIX async API: `auto`, `aio`, or `uring`. Use `uring` for highest NFS/VAST throughput. |
| `--storage-num-handles` | Number of concurrent transfer handles per storage op. Recommended: `8` together with `--storage-block-size 1M --storage-posix-api uring`. |
| `--warmup-iters` | Warmup iterations per traffic pattern (default: 30) |
| `--isolation-iters` | Isolation-benchmark iterations per traffic pattern (default: 10) |

## Command Descriptions

### KVBench Commands

#### Plan Command

The `plan` command generates and displays recommended `nixlbench` command configurations based on your model architecture and parameters. It helps you prepare optimal benchmark settings without running the benchmark itself.

```bash
python main.py plan --model ./examples/model_deepseek_r1.yaml --model_config "./examples/block-tp1-pp8.yaml" --backend POSIX
```

#### Profile Command

The `profile` command actually runs the benchmark with the specified configuration using `nixlbench`, collecting performance data across various KV cache operations and access patterns.

```bash
python main.py profile --model ./examples/model_deepseek_r1.yaml --model_config "./examples/block-tp1-pp8.yaml" --backend POSIX
```

#### KVCache Command

The `kvcache` command analyzes and displays detailed information about the KV cache for a specified model configuration, including model type, sequence lengths, batch sizes, and I/O sizes.

```bash
python main.py kvcache --model ./examples/model_deepseek_r1.yaml --model_config "./examples/block-tp1-pp8.yaml" --isl 10000 --page_size 512
Model          ISL    Num Requests    Batch Size  IO Size      TP    PP    Page Size  Access
-----------  -----  --------------  ------------  ---------  ----  ----  -----------  --------
DEEPSEEK_R1  10000              10          1490  2.25 MB       1     8          512  block
```

### CTP Commands

#### Sequential CT Perftest

Benchmark the performance of a continuum of traffic patterns one after the other. Before running each pattern, all ranks do a barrier, optionally sleep for a given amount of time, then run the pattern and measure execution time.

**Reports**: Sequential CT Perftest reports the total latency per matrix execution, along with their latency when run isolated, which can be used to evaluate how well the network reacts to congestion.

```
  Transfer size (GB)    Latency (ms)    Isolated Latency (ms)    Num Senders
--------------------  --------------  -----------------------  -------------
         4.945               35.047                   35.421              4
         3.230               21.152                   21.800              4
         1.104                8.222                    8.280              4
         ...                 ...                         ...             ...
         0.129                2.147                    2.386              4
```

#### CT Perftest {#ct-perftest}

Benchmark the performance of one traffic pattern. The pattern is run in multiple iterations and then metrics are reported. Useful for optimizing specific patterns.

**Reports**: CT Perftest reports total latency (time elapsed between the first rank started until the last rank finished), average time per iteration, total size sent over the network, and average bandwidth by rank.

**Important note**: GPU memory is allocated with pytorch on the GPU specified by the `CUDA_VISIBLE_DEVICE` environment variable, make sure that each process sets this variable to the right device.

## Examples

### KVBench Examples

#### DeepSeek R1 with Block Access (TP=1, PP=16)
```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp16.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
================================================================================
Model Config: ./examples/block-tp1-pp16.yaml
ISL: 10000 tokens
Page Size: 256
Requests: 10
TP: 1
PP: 16
================================================================================
nixlbench \
    --backend GDS \
    --max_batch_size 5958 \
    --max_block_size 589824 \
    --start_batch_size 5958 \
    --start_block_size 589824 \
    --target_seg_type VRAM
```

#### DeepSeek R1 with Layer Access (TP=1, PP=16)
```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/layer-tp1-pp16.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
================================================================================
Model Config: ./examples/layer-tp1-pp16.yaml
ISL: 10000 tokens
Page Size: 256
Requests: 10
TP: 1
PP: 16
================================================================================
nixlbench \
    --backend GDS \
    --max_batch_size 23829 \
    --max_block_size 147456 \
    --start_batch_size 23829 \
    --start_block_size 147456 \
    --target_seg_type VRAM
```

#### Overriding Model Configuration with CLI Args
```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp8.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379" \
  --pp 32 \
  --num_requests 100
================================================================================
Model Config: ./examples/block-tp1-pp8.yaml
ISL: 10000 tokens
Page Size: 256
Requests: 100
TP: 1
PP: 32
================================================================================
nixlbench \
    --backend GDS \
    --max_batch_size 119141 \
    --max_block_size 294912 \
    --start_batch_size 119141 \
    --start_block_size 294912 \
    --target_seg_type VRAM
```

### CTP Examples

#### Configuration Files

CTP tests are defined using YAML configuration files.

**CT Perftest Configuration** (single traffic pattern):
```yaml
iters: 50
warmup_iters: 10
traffic_pattern:
  matrix_file: "/path/to/matrix.txt"
  shards: 1
  mem_type: "cuda"
  xfer_op: "WRITE"
```

**Sequential CT Perftest Configuration** (multiple traffic patterns):
```yaml
traffic_patterns:
  # RDMA only (no storage)
  - matrix_file: /path/to/matrices/matrix_0.txt
    mem_type: cuda
    sleep_before_launch_sec: 0.01
    metadata:
      description: "RDMA transfer only"

  # RDMA + Storage (75% cache hit)
  - matrix_file: /path/to/matrices/matrix_1.txt
    mem_type: cpu
    sleep_before_launch_sec: 0.005   # prefill compute, after read, before RDMA
    decode_compute_sec: 0.002        # decode compute, after RDMA, before write
    storage:
      read:  [1572864, 1572864, 0, 0]  # Ranks 0,1 read 1.5MB each
      write: [524288, 524288, 0, 0]    # Ranks 0,1 write 0.5MB each
    metadata:
      prefix_hit_rate: 0.75

  # Storage only (no RDMA) - matrix_file is optional
  - mem_type: cpu
    storage:
      read:  [1M, 1M, 1M, 1M, 1M, 1M, 1M, 1M]   # All 8 ranks read 1MB
      write: [1M, 1M, 1M, 1M, 1M, 1M, 1M, 1M]   # All 8 ranks write 1MB
```

**Traffic Pattern Parameters**:
- `tp_file`: Unified TP file with `[rdma]`, `[read]`, `[write]` sections — preferred when both RDMA and storage are exercised by the same pattern. Mutually exclusive with `matrix_file`/`matrix` and the inline `storage:` block.
- `matrix_file`: File containing the RDMA transfer matrix (legacy, optional — omit for storage-only).
- `matrix`: Inline RDMA matrix as 2D array (alternative to `matrix_file`).
- `mem_type`: Memory type — `"cuda"`, `"vram"`, `"cpu"`, `"dram"` (default: `"cuda"`).
- `sleep_before_launch_sec`: Prefill compute simulation, in seconds. Sleeps after the storage read and before the RDMA transfer (default: 0). `sequential-ct-perftest` only.
- `decode_compute_sec`: Decode compute simulation, in seconds. Sleeps after the RDMA transfer and before the storage write (default: 0). `sequential-ct-perftest` only. Replaces the old `sleep_after_launch_sec`, which is still accepted but deprecated.
- `storage`: Per-rank storage requirements when using the legacy split format.
- `storage.read`: Array of read sizes per rank (index = rank, use 0 to skip).
- `storage.write`: Array of write sizes per rank (index = rank, use 0 to skip).
- `metadata`: Arbitrary metadata (informational, not used by kvbench).
- `shards`: Number of chunks to shard the buffer into (default: 1).

**RDMA Matrix File Format**:
Matrix cells are separated by whitespace and contain either bytes as integers or sizes with a `K`/`M`/`G` suffix.

Example matrix file:
```
0 0 1M 1M
0 0 1M 1M
0 0 0 5K
0 0 0 5K
```

**Unified TP File Format** (combined RDMA + Storage):

A single text file describes the RDMA matrix, the per-rank storage reads, and the per-rank storage writes for one traffic pattern. Each section is optional, so the same file format covers RDMA-only, storage-only, and mixed patterns. Files without any `[section]` header are parsed as a legacy RDMA-only matrix (backward compatible).

```text
[rdma]
0 710M 0 0
0 0 0 0

[read]
710M 710M 0 0

[write]
710M 710M 0 0
```

Reference it from YAML with `tp_file`:

```yaml
traffic_patterns:
  - tp_file: tps/tp_0.tp
    mem_type: cpu
```

`inference_workload_matgen.py` emits this unified `tp_file` format exclusively (one `tps/tp_<idx>.tp` per pattern); it does not also write the legacy `matrix_file` + inline `storage:` block, since the runner treats `tp_file` and the legacy keys as mutually exclusive.

#### Generate Traffic Pattern Matrices

Optionally, generate matrices using the inference workload matrix generation tool:

```bash
python test/inference_workload_matgen.py generate \
    --num-user-requests 10 \
    --num-prefill-nodes 16 \
    --num-decode-nodes 16 \
    --prefill-tp 8 \
    --prefill-pp 1 \
    --prefill-cp 1 \
    --decode-tp 8 \
    --decode-pp 1 \
    --decode-cp 1 \
    --results-dir $PWD/matrices \
    --model llama-405b
```

#### Running CTP Tests

Please read the important note about setting `CUDA_VISIBLE_DEVICES` in [CT Perftest section](#ct-perftest).

**Sequential CT Perftest**:
```bash
# Basic usage
python main.py sequential-ct-perftest ./config.yaml

# With options
python main.py sequential-ct-perftest ./config.yaml \
    --verify-buffers \
    --json-output-path ./results.json

# With storage simulation (use matgen with --prefix-hit-rate)
# matgen requires --num-prefill-nodes and --num-decode-nodes.
python test/inference_workload_matgen.py generate \
    --model llama-405b --prefix-hit-rate 0.75 \
    --num-prefill-nodes 1 --num-decode-nodes 1 \
    --prefill-tp 8 --decode-tp 8 \
    --results-dir ./workload
python main.py sequential-ct-perftest ./workload/metadata.yaml \
    --storage-backend POSIX \
    --storage-path /tmp/kvbench_storage

# Tuned for NFS/VAST POSIX throughput
python main.py sequential-ct-perftest ./workload/metadata.yaml \
    --storage-backend POSIX \
    --storage-path /mnt/vast/kvbench \
    --storage-posix-api uring \
    --storage-block-size 1M \
    --storage-num-handles 8

# With debug logging
python main.py --debug sequential-ct-perftest ./config.yaml \
    --verify-buffers \
    --json-output-path ./results.json

# With Slurm (see the Quickstart for the etcd setup this needs)
srun --partition=<partition> --nodes=2 --ntasks-per-node=8 --export=ALL \
  bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID && \
    python main.py sequential-ct-perftest ./config.yaml \
      --verify-buffers \
      --json-output-path ./results.json'
```

Multi-rank runs need `NIXL_ETCD_ENDPOINTS` and `NIXL_ETCD_NAMESPACE` exported before `srun`. See [Quickstart (end-to-end)](#quickstart-end-to-end).

**CT Perftest**:
```bash
# Basic usage
python main.py ct-perftest ./config.yaml

# With options
python main.py ct-perftest ./config.yaml \
    --verify-buffers \
    --print-recv-buffers

# With debug logging
python main.py --debug ct-perftest ./config.yaml \
    --verify-buffers
```

## Implementation Details

### CTP Implementation
- Custom traffic patterns defined using transfer matrices where cell [i,j] defines message size from rank i to rank j
- `benchmark.kvbench.test.custom_traffic_perftest.py` implements CT Perftest and TrafficPattern dataclass
- `benchmark.kvbench.test.sequential_custom_traffic_perftest.py` implements Sequential CT Perftest
- Utilizes common utilities for distributed testing support
- `benchmark.kvbench.test.traffic_pattern.py` defines abstraction for traffic patterns

### Known Issues
- The nixl xfer preparation currently takes significant time (the `_prepare_tp()` method)

### Next Steps
- [ ] Support more memory types beyond CUDA
- [ ] Optimize transfer preparation performance

## Developer Guides

- [Tutorial with GDS](docs/tutorial-gds.md) - Quick tutorial for running NIXLBench with GDS
- [Creating a Model Configuration](docs/creating-a-model-config.md) - Guide for creating model configuration files
- [Adding a New Model Architecture](docs/adding-a-new-model-architecture.md) - Instructions for extending KVBench with new model architectures
