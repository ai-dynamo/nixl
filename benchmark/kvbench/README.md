# NIXL KVBench
A utility for generating NIXL Bench commands that test KVCache transfer across various LLM architectures and access patterns (including block and layer approaches).

## Supported LLM Architectures
- DeepSeek R1
- LLama 3.1 
- and more

## Building

### Docker
```bash
# cd nixl/benchmark/nixlbench
export NIXL_SRC=/path/to/nixl/
cd nixlbench/contrib
./build.sh --nixl $NIXL_SRC
```

### Python
```bash
cd kvbench
python3 -m venv venv
source venv/bin/activate
pip install uv
uv sync --active
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
python main.py kvcache --model ./examples/model_deepseek_r1.yaml --model_config "./examples/block-tp1-pp8.yaml" 
Model                  : DEEPSEEK_R1
Input Sequence Length  : 10000
Batch Size             : 298
IO Size                : 1.12 MB
```

### Display NIXL Bench Commands
```bash
python main.py plan --model ./examples/model_deepseek_r1.yaml --model_configs "./examples/block-tp1-pp16.yaml" --backend GDS --source gpu --etcd-endpoint "http://10.185.99.120:3379"
================================================================================
Model Config: ./examples/block-tp1-pp16.yaml
ISL: 10000 tokens
================================================================================
nixlbench \
    --backend GDS \
    --etcd_endpoints http://10.185.99.120:3379 \
    --initiator_seg_type VRAM \
    --max_batch_size 149 \
    --max_block_size 589824 \
    --start_batch_size 149 \
    --start_block_size 589824
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
================================================================================
nixlbench \
    --backend GDS \
    --initiator_seg_type VRAM \
    --max_batch_size 5958 \
    --max_block_size 589824 \
    --start_batch_size 5958 \
    --start_block_size 589824
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
================================================================================
nixlbench \
    --backend GDS \
    --initiator_seg_type VRAM \
    --max_batch_size 23829 \
    --max_block_size 147456 \
    --start_batch_size 23829 \
    --start_block_size 147456
```