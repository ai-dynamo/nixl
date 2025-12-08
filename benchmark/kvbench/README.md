# NIXL KVBench
A comprehensive utility for generating NIXL Bench commands that test KVCache transfer across various LLM architectures and access patterns, plus custom traffic performance testing for asymmetric traffic flows.

## Table of Contents
- [Overview](#overview)
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
- matrix_file: /path/to/matrices/matrix_0.txt
  metadata:
    isl: 38328
  sleep_after_launch_sec: 16.480753057792
- matrix_file: /path/to/matrices/matrix_1.txt
  metadata:
    isl: 25034
  sleep_after_launch_sec: 71.875102179328
```

**Traffic Pattern Parameters**:
- `matrix_file`: File containing the transfer matrix (required)
- `shards`: Number of chunks to shard the buffer into (default: 1)
- `mem_type`: Memory type, currently supports "cuda" (default: "cuda") (Use `CUDA_VISIBLE_DEVICES` to control the GPU device)
- `xfer_op`: Transfer operation, "READ" or "WRITE" (default: "WRITE")
- `sleep_after_launch_sec`: Seconds to sleep before running pattern (default: 0)

**Matrix File Format**:
Matrix cells are separated by whitespaces and contain either bytes as integers or standard units (K, M, G).

Example matrix file:
```
0 0 1M 1M
0 0 1M 1M
0 0 0 5K
0 0 0 5K
```

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

# With debug logging
python main.py --debug sequential-ct-perftest ./config.yaml \
    --verify-buffers \
    --json-output-path ./results.json

# With Slurm
srun <params> bash -c "
  CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  python main.py sequential-ct-perftest ./config.yaml \
    --verify-buffers \
    --json-output-path ./results.json
"
```

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
For more detailed information, please refer to the following documentation:
- [Tutorial with GDS](docs/tutorial-gds.md) - Quick tutorial for running NIXLBench with GDS
- [Creating a Model Configuration](docs/creating-a-model-config.md) - Guide for creating model configuration files
- [Adding a New Model Architecture](docs/adding-a-new-model-architecture.md) - Instructions for extending KVBench with new model architectures
