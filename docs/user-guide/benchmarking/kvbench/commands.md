---
title: KVBench Commands and Examples
description: KVBench command reference, model configuration guide, and LLM example configurations.
---

This page covers the KVBench command reference, model configuration schemas, and end-to-end LLM examples. For installation and build instructions, see [Building KVBench](./build.md). KVBench's `profile` command invokes [NIXLBench](../nixlbench/index.md) as a subprocess to run the actual transfer benchmarks.

## Command Reference

### plan

The `plan` command generates and displays recommended `nixlbench` command configurations based on your model architecture and parameters. It computes KV cache transfer sizes and produces the exact NIXLBench invocation without running the benchmark itself.

```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp8.yaml \
  --backend GDS \
  --source gpu
```

Use `--format` to control output format: `text` (default), `json`, or `csv`. The `--model_configs` flag accepts glob patterns to plan multiple configurations in a single invocation.

### profile

The `profile` command runs NIXLBench with the planned configuration, collecting performance data across KV cache operations and access patterns. It computes the same parameters as `plan` and then executes `nixlbench` as a subprocess.

```bash
python main.py profile \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp8.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
```

### kvcache

The `kvcache` command analyzes and displays detailed information about the KV cache for a specified model configuration, including model type, sequence lengths, batch sizes, and I/O sizes.

```bash
python main.py kvcache \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp8.yaml \
  --isl 10000 \
  --page_size 512
```

Output:

```
Model          ISL    Num Requests    Batch Size  IO Size      TP    PP    Page Size  Access
-----------  -----  --------------  ------------  ---------  ----  ----  -----------  --------
DEEPSEEK_R1  10000              10          1490  2.25 MB       1     8          512  block
```

### ct-perftest

The `ct-perftest` command benchmarks the performance of a single custom traffic pattern. The pattern runs in multiple iterations and then metrics are reported. This is useful for optimizing specific traffic patterns.

```bash
python main.py ct-perftest ./config.yaml --verify-buffers
```

**Reports:** Total latency (time elapsed between the first rank starting and the last rank finishing), average time per iteration, total size sent over the network, and average bandwidth by rank.

<Note>
GPU memory is allocated with PyTorch on the GPU specified by the `CUDA_VISIBLE_DEVICES` environment variable. Make sure each process sets this variable to the correct device.
</Note>

### sequential-ct-perftest

The `sequential-ct-perftest` command benchmarks the performance of a series of traffic patterns executed one after the other. Before running each pattern, all ranks perform a barrier, optionally sleep for a configured duration, then run the pattern and measure execution time.

```bash
python main.py sequential-ct-perftest ./config.yaml \
  --verify-buffers \
  --json-output-path ./results.json
```

**Reports:** Total latency per matrix execution, along with isolated latency (latency when the pattern is run alone), which can be used to evaluate how well the network reacts to congestion.

```
  Transfer size (GB)    Latency (ms)    Isolated Latency (ms)    Num Senders
--------------------  --------------  -----------------------  -------------
         4.945               35.047                   35.421              4
         3.230               21.152                   21.800              4
         1.104                8.222                    8.280              4
         ...                 ...                         ...             ...
         0.129                2.147                    2.386              4
```

## Command Line Arguments

### Common Arguments

These arguments are shared across KVBench commands (`plan`, `kvcache`, `profile`):

| Argument | Description |
| -------- | ----------- |
| `--model` | Path to a model architecture config YAML file |
| `--model_config` | Path to a single model config YAML file |
| `--model_configs` | Path to multiple model config YAML files (supports glob patterns like `configs/*.yaml`) |

### CLI Override Arguments

These arguments override values specified in model config files:

| Argument | Description |
| -------- | ----------- |
| `--pp` | Pipeline parallelism size |
| `--tp` | Tensor parallelism size |
| `--isl` | Input sequence length |
| `--osl` | Output sequence length |
| `--num_requests` | Number of requests |
| `--page_size` | Page size |
| `--access_pattern` | Access pattern (`block` or `layer`) |

### Plan Command Arguments

Specific to the `plan` command:

| Argument | Description |
| -------- | ----------- |
| `--format` | Output format of the nixlbench command: `text`, `json`, or `csv` (default: `text`) |

### Shared Benchmark Arguments

These arguments are used by both `plan` and `profile` commands and are passed through to NIXLBench:

| Argument | Description |
| -------- | ----------- |
| `--source` | Source of the NIXL descriptors: `file`, `memory`, or `gpu` (default: `file`) |
| `--destination` | Destination of the NIXL descriptors: `file`, `memory`, or `gpu` (default: `memory`) |
| `--backend` | Communication backend: [UCX](../../backends/ucx.md), [GDS](../../backends/gds.md), [GDS_MT](../../backends/gds-mt.md), [POSIX](../../backends/posix.md), [GPUNETIO](../../backends/gpunetio.md), [Mooncake](../../backends/mooncake.md), [HF3FS](../../backends/hf3fs.md), [OBJ](../../backends/obj.md) (default: `UCX`) |
| `--worker_type` | Worker to use to transfer data: `nixl` or `nvshmem` (default: `nixl`) |
| `--initiator_seg_type` | Memory segment type for initiator: `DRAM`, `VRAM`, `FILE`, or `OBJ` (default: `DRAM`) |
| `--target_seg_type` | Memory segment type for target: `DRAM`, `VRAM`, `FILE`, or `OBJ` (default: `DRAM`) |
| `--scheme` | Communication scheme: `pairwise`, `manytoone`, `onetomany`, or `tp` (default: `pairwise`) |
| `--mode` | Process mode: `SG` (single GPU per process) or `MG` (multi GPU per process) (default: `SG`) |
| `--op_type` | Operation type: `READ` or `WRITE` (default: `WRITE`) |
| `--check_consistency` | Enable consistency checking |
| `--total_buffer_size` | Total buffer size in bytes (default: 8 GiB) |
| `--recreate_xfer` | Recreate transfer handle for every iteration (default: `false` for all backends, `true` for GUSLI) |
| `--start_block_size` | Starting block size in bytes (default: 4 KiB) |
| `--max_block_size` | Maximum block size in bytes (default: 64 MiB) |
| `--start_batch_size` | Starting batch size (default: `1`) |
| `--max_batch_size` | Maximum batch size (default: `1`) |
| `--num_iter` | Number of iterations (default: `1000`) |
| `--warmup_iter` | Number of warmup iterations (default: `100`) |
| `--num_threads` | Number of threads used by benchmark (default: `1`) |
| `--num_initiator_dev` | Number of devices in initiator processes (default: `1`) |
| `--num_target_dev` | Number of devices in target processes (default: `1`) |
| `--enable_pt` | Enable progress thread |
| `--progress_threads` | Number of progress threads (default: `0`) |
| `--device_list` | Comma-separated device names (default: all) |
| `--runtime_type` | Type of runtime to use: `ETCD` (default: `ETCD`) |
| `--etcd-endpoints` | [etcd](../../etcd-metadata-exchange.md) server URL for coordination (default: `http://localhost:2379`) |
| `--storage_enable_direct` | Enable direct I/O for storage operations |
| `--filepath` | File path for storage operations |
| `--enable_vmm` | Enable VMM memory allocation when DRAM is requested |

<Note>
KVBench uses `--etcd-endpoints` (hyphens). NIXLBench uses `--etcd_endpoints` (underscores). Both forms are accepted by the CLI, but this documentation follows each tool's convention.
</Note>

### CTP Command Arguments

Specific to CTP (Custom Traffic Performance) commands (`ct-perftest` and `sequential-ct-perftest`):

| Argument | Description |
| -------- | ----------- |
| `config_file` | Path to YAML configuration file (required, positional argument) |
| `--verify-buffers` / `--no-verify-buffers` | Verify buffer contents after transfer (default: `false`) |
| `--print-recv-buffers` / `--no-print-recv-buffers` | Print received buffer contents (default: `false`) |
| `--json-output-path` | Path to save JSON output (`sequential-ct-perftest` only) |

## Model Configuration Guide

KVBench uses two YAML configuration files: a **model architecture** file describing the LLM structure, and a **model config** file specifying parallelism, runtime, and system settings. Both files are passed to KVBench commands via the `--model` and `--model_config` flags respectively.

### Model Architecture YAML

The model architecture file defines the structural parameters of an LLM. Different attention mechanisms require different fields.

**Common fields** shared by all architectures:

| Field | Description |
| ----- | ----------- |
| `model_name` | Model identifier (e.g., `DEEPSEEK_R1`, `LLAMA3.1_70B`) |
| `num_layers` | Number of transformer layers |
| `query_head_dimension` | Dimension of each query head |
| `num_model_params` | Total model parameter count |

**MLA fields** (Multi-Latent Attention, e.g., DeepSeek R1):

| Field | Description |
| ----- | ----------- |
| `num_query_heads` | Number of query attention heads |
| `embedding_dimension` | Model embedding dimension |
| `rope_mla_dimension` | RoPE dimension for MLA |
| `mla_latent_vector_dimension` | Latent vector dimension for MLA compression |

**MHA/GQA fields** (Multi-Head / Grouped-Query Attention, e.g., Llama 3.1):

| Field | Description |
| ----- | ----------- |
| `num_query_heads_with_mha` | Number of query heads (MHA variant) |
| `gqa_num_queries_in_group` | Number of queries per KV head group (GQA) |

**DeepSeek R1 example** (`model_deepseek_r1.yaml`):

```yaml
model_name: 'DEEPSEEK_R1'        # Model identifier
num_layers: 61                    # 61 transformer layers
num_query_heads: 128              # 128 query attention heads
query_head_dimension: 128         # 128-dim per query head
embedding_dimension: 7168         # Model embedding dimension
rope_mla_dimension: 64            # RoPE dimension for MLA
mla_latent_vector_dimension: 512  # Latent vector dimension for MLA compression
num_model_params: 671000000000    # 671B parameters
```

**Llama 3.1 70B example** (`model_llama_3_1_70b.yaml`):

```yaml
model_name: 'LLAMA3.1_70B'       # Model identifier
num_layers: 80                    # 80 transformer layers
num_query_heads_with_mha: 64      # 64 query heads (MHA)
query_head_dimension: 128         # 128-dim per query head
gqa_num_queries_in_group: 8       # 8 queries per KV head group (GQA)
num_model_params: 70000000000     # 70B parameters
```

### Model Config YAML

The model config file has three sections: `strategy`, `runtime`, and `system`.

**Strategy fields:**

| Field | Description |
| ----- | ----------- |
| `tp_size` | Tensor parallelism size -- number of GPUs for tensor-parallel execution (default: `1`) |
| `pp_size` | Pipeline parallelism size -- number of GPUs for pipeline-parallel execution (default: `1`) |
| `model_quant_mode` | Model weight quantization mode, e.g., `fp8`, `fp16`, `int8` (default: `"fp8"`) |
| `kvcache_quant_mode` | KV cache quantization mode, e.g., `fp8`, `fp16`, `int8` (default: `"fp8"`) |

**Runtime fields:**

| Field | Description |
| ----- | ----------- |
| `isl` | Input sequence length in tokens (default: `1`) |
| `osl` | Output sequence length in tokens (default: `1`) |
| `num_requests` | Number of inference requests (default: `1`) |

**System fields:**

| Field | Description |
| ----- | ----------- |
| `hardware` | Hardware platform (e.g., `"H100"`, `"A100"`) |
| `backend` | Inference backend engine (e.g., `"SGLANG"`) |
| `access_pattern` | KV cache access pattern: `"block"` or `"layer"` |
| `page_size` | Page size for access pattern (default: `1`) |
| `source` | Source descriptor type |
| `destination` | Destination descriptor type |

**Block access example** (`block-tp1-pp8.yaml`):

```yaml
strategy:
  tp_size: 1                    # Tensor parallelism -- 1 GPU for tensor-parallel
  pp_size: 8                    # Pipeline parallelism -- 8 GPUs for pipeline-parallel
  model_quant_mode: "fp8"       # Model weight quantization
  kvcache_quant_mode: "fp8"     # KV cache quantization

runtime:
  isl: 1000                     # Input sequence length (tokens)
  osl: 100                      # Output sequence length (tokens)
  num_requests: 10              # Number of inference requests

system:
  hardware: "H100"              # Hardware platform
  backend: "SGLANG"             # Inference backend engine
  access_pattern: "block"       # KV cache access pattern
  page_size: 16                 # Page size for block access
```

<Note>
Block access groups KV cache entries into fixed-size pages. Layer access transfers KV cache one transformer layer at a time. Block access typically produces fewer, larger transfers; layer access produces more, smaller transfers.
</Note>

## LLM Examples

End-to-end examples showing model architecture YAML, model config YAML, and the `plan` and `profile` commands. These examples can be copy-pasted and run directly from the KVBench directory.

### DeepSeek R1

#### Block Access (TP=1, PP=16)

**Model architecture** (`model_deepseek_r1.yaml`):

```yaml
model_name: 'DEEPSEEK_R1'
num_layers: 61
num_query_heads: 128
query_head_dimension: 128
embedding_dimension: 7168
rope_mla_dimension: 64
mla_latent_vector_dimension: 512
num_model_params: 671000000000
```

**Model config** (`block-tp1-pp16.yaml`):

```yaml
strategy:
  tp_size: 1
  pp_size: 16
  model_quant_mode: "fp8"
  kvcache_quant_mode: "fp8"

runtime:
  isl: 1000
  osl: 100
  num_requests: 10

system:
  hardware: "H100"
  backend: "SGLANG"
  access_pattern: "block"
  page_size: 16
```

**Plan command:**

```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp16.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
```

**Output:**

```
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

**Profile command:**

```bash
python main.py profile \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp16.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
```

#### Layer Access (TP=1, PP=16)

Uses the same model architecture YAML as above (`model_deepseek_r1.yaml`).

**Model config** (`layer-tp1-pp16.yaml`):

```yaml
strategy:
  tp_size: 1
  pp_size: 16
  model_quant_mode: "fp8"
  kvcache_quant_mode: "fp8"

runtime:
  isl: 1000
  osl: 100
  num_requests: 10

system:
  hardware: "H100"
  backend: "SGLANG"
  access_pattern: "layer"
  page_size: 16
```

**Plan command:**

```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/layer-tp1-pp16.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
```

**Output:**

```
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

With layer access, the batch size increases and block size decreases compared to block access, reflecting the per-layer transfer granularity.

### Llama 3.1 70B

#### Block Access (TP=1, PP=8)

**Model architecture** (`model_llama_3_1_70b.yaml`):

```yaml
model_name: 'LLAMA3.1_70B'
num_layers: 80
num_query_heads_with_mha: 64
query_head_dimension: 128
gqa_num_queries_in_group: 8
num_model_params: 70000000000
```

**Model config** (`block-tp1-pp8.yaml`):

```yaml
strategy:
  tp_size: 1
  pp_size: 8
  model_quant_mode: "fp8"
  kvcache_quant_mode: "fp8"

runtime:
  isl: 1000
  osl: 100
  num_requests: 10

system:
  hardware: "H100"
  backend: "SGLANG"
  access_pattern: "block"
  page_size: 16
```

**Plan command:**

```bash
python main.py plan \
  --model ./examples/model_llama_3_1_70b.yaml \
  --model_config ./examples/block-tp1-pp8.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
```

<Note>
The output follows the same format as the DeepSeek R1 example above, with values computed from the Llama 3.1 70B architecture.
</Note>

**Profile command:**

```bash
python main.py profile \
  --model ./examples/model_llama_3_1_70b.yaml \
  --model_config ./examples/block-tp1-pp8.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
```
