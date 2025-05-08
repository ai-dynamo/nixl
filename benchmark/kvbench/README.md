# NIXL KVBench
A tool that helps generate NIXL Benchmark commands for common LLM architectures and access patterns for KVCache transfer. 

## Supported LLM Architectures
- DeepSeek R1
- LLama 3.1 
- and more

## Building

```bash
# cd nixl/benchmark/kvbench
uv sync
```

## Usage 

### Basic Usage
```bash
python main.py --help
usage: main.py [-h] {plan,kvcache,profile} ...

KVBench

positional arguments:
  {plan,kvcache,profile}
                        Available commands
    plan                Display the recommended configuration for nixlbench
    kvcache             Display kvcache information
    profile             Run nixlbench

options:
  -h, --help            show this help message and exit
```

### Display KVCache Information
```bash
python main.py kvcache --model ./examples/model_deepseek_r1.yaml --model_config ./examples/latency-chat-sglang.yaml
KV Cache Information:
---------------------------------------------------
KV Cache Size Per Token        : 34.31 KB
Batch Size                     : 16
Page Size (batch)              : 549.0 KB
---------------------------------------------------
Input Sequence Length (ISL)    : 8192
KV Cache Size For ISL          : 274.5 MB
---------------------------------------------------
Output Sequence Length (OSL)   : 2048
KV Cache Size For OSL          : 68.62 MB
---------------------------------------------------
Total Sequence Length (ISL+OSL): 10240
Total KV Cache Size (ISL+OSL)  : 343.12 MB
```

### Display NIXL Bench Commands
```bash
python main.py plan  --model ./examples/model_deepseek_r1.yaml --model_config ./examples/latency-chat-sglang.yaml 
================================================================================
NIXL BENCHMARK COMMAND FOR ISL (INPUT SEQUENCE)
ISL: 8192 tokens
================================================================================
nixlbench \
    --max_batch_size 512 \
    --max_block_size 562176 \
    --start_batch_size 512 \
    --start_block_size 562176
================================================================================
NIXL BENCHMARK COMMAND FOR OSL (OUTPUT SEQUENCE)
OSL: 2048 tokens
================================================================================
nixlbench \
    --max_batch_size 128 \
    --max_block_size 562176 \
    --start_batch_size 128 \
    --start_block_size 562176

NOTE: Use the appropriate command based on whether you're benchmarking
      input sequence (prefill) or output sequence (generation) performance.
```