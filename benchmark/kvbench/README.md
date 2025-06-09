# NIXL KVBench
A utility for generating NIXL Bench commands that test KVCache transfer across various LLM architectures and access patterns (including block and layer approaches).

## Table of Contents
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
- [Command Descriptions](#command-descriptions)
  - [Plan Command](#plan-command)
  - [Profile Command](#profile-command)
  - [KVCache Command](#kvcache-command)
  - [IOSize Command](#iosize-command)
- [Examples](#example-deepseek-r1-with-block-access-tp1-pp16)
  - [DeepSeek R1 with Block Access](#example-deepseek-r1-with-block-access-tp1-pp16)
  - [DeepSeek R1 with Layer Access](#example-deepseek-r1-with-layer-access-tp1-pp16)
  - [Overriding Model Configuration](#example-overriding-model-configuration-with-cli-args)
- [Developer Guides](#developer-guides)

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
usage: main.py [-h] {plan,profile,kvcache,iosize} ...

KVBench

positional arguments:
  {plan,profile,kvcache,iosize}
                        Available commands
    plan                Display the recommended configuration for nixlbench
    profile             Run nixlbench
    kvcache             Display kvcache information
    iosize              Display io size information

options:
  -h, --help            show this help message and exit
```

## Command Line Arguments

KVBench supports various argument groups that apply to different commands:

### Common Arguments

These arguments are shared across all KVBench commands (plan, kvcache, profile):

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
| `--backend` | Communication backend [UCX, UCX_MO, GDS] (default: UCX) |
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
| `--device_list` | Comma-separated device names (default: all) |
| `--runtime_type` | Type of runtime to use [ETCD] (default: ETCD) |
| `--etcd-endpoints` | ETCD server URL for coordination (default: http://localhost:2379) |
| `--storage_enable_direct` | Enable direct I/O for GDS operations |
| `--gds_filepath` | File path for GDS operations |
| `--enable_vmm` | Enable VMM memory allocation when DRAM is requested |

## Command Descriptions

KVBench provides four main commands:

### Plan Command

The `plan` command generates and displays recommended `nixlbench` command configurations based on your model architecture and parameters. It helps you prepare optimal benchmark settings without running the benchmark itself.

```bash
python main.py plan --model ./examples/model_deepseek_r1.yaml --model_config "./examples/block-tp1-pp8.yaml"
```

### Profile Command

The `profile` command actually runs the benchmark with the specified configuration using `nixlbench`, collecting performance data across various KV cache operations and access patterns.

```bash
python main.py profile --model ./examples/model_deepseek_r1.yaml --model_config "./examples/block-tp1-pp8.yaml"
```

### KVCache Command

The `kvcache` command analyzes and displays detailed information about the KV cache for a specified model configuration, including model type, sequence lengths, batch sizes, and I/O sizes.

```bash
python main.py kvcache --model ./examples/model_deepseek_r1.yaml --model_config "./examples/block-tp1-pp8.yaml"
Model                  : DEEPSEEK_R1
Input Sequence Length  : 10000
Batch Size             : 298
IO Size                : 1.12 MB
```

### IOSize Command

The `iosize` command displays information about the I/O size requirements for a specified model configuration, helping you understand memory usage and data transfer needs.

```bash
python main.py iosize --model ./examples/model_deepseek_r1.yaml --model_config "./examples/block-tp1-pp8.yaml"
IO Size per GPU: 4608
```

## Example: DeepSeek R1 with Block Access (TP=1, PP=16)
```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp16.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoint "http://localhost:2379"
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

## Example: DeepSeek R1 with Layer Access (TP=1, PP=16)
```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/layer-tp1-pp16.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoint "http://localhost:2379"
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

## Example: Overriding model configuration with cli args
```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp8.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoint "http://localhost:2379" \
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

## Developer Guides
For more detailed information, please refer to the following documentation:
- [Tutorial with GDS](docs/tutorial-gds.md) - Quick tutorial for running NIXLBench with GDS
- [Creating a Model Configuration](docs/creating-a-model-config.md) - Guide for creating model configuration files
- [Adding a New Model Architecture](docs/adding-a-new-model-architecture.md) - Instructions for extending KVBench with new model architectures
