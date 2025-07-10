# NIXL KVCache GDS Profiling Tutorial

This tutorial explains how to use KVBench to profile KV cache transfers between file storage and GPU memory using NIXLBench

## Prerequisites

- NIXL repository cloned and properly set up
- CUDA environment configured with GPU Direct Storage support
- An ETCD server running (default: http://localhost:2379)
- A directory with write permissions for GDS operations

## Understanding the Model Configuration

Before we start, let's understand the model configuration parameters in our YAML files. For example, in `block-tp1-pp16.yaml`:

```yaml
strategy:
  tp_size: 1                # Tensor Parallelism size
  pp_size: 16               # Pipeline Parallelism size
  model_quant_mode: "fp8"   # Model weights quantization mode
  kvcache_quant_mode: "fp8" # KV cache quantization mode

runtime:
  isl: 1000             # Input sequence length
  osl: 100              # Output sequence length
  num_requests: 10      # Number of requests to process

system:
  hardware: "H100"
  backend: "SGLANG"
  access_pattern: "block"   # Access pattern: "block" or "layer"
  page_size: 16             # Page size for KV cache
```

## Profiling

### Step 1: Perform a WRITE operation first (GPU to Storage)

```bash
python main.py profile \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp16.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379" \
  --filepath /path/to/your/directory \
  --num_requests 1
```

This command:
- Uses the DeepSeek model configuration
- Sets up GDS as the backend
- Sets the source as GPU memory (`--source gpu`)
- Specifies the directory for GDS files (`--filepath`)
- Specifies the number of concurrent user requests to simulate (`--num_requests`)

### Step 2: Perform a READ operation (Storage to GPU)

After the write operation completes successfully, you can perform the read operation:

```bash
python main.py profile \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp16.yaml \
  --backend GDS \
  --source file \
  --etcd-endpoints "http://localhost:2379" \
  --filepath /path/to/your/directory \
  --num_requests 1
```

This command is similar to the WRITE operation but changes:
- Source to file storage (`--source file`)
- Destination to GPU memory (`--destination gpu`)

## Understanding Output

The profiling results will show:
- Transfer bandwidth between storage and GPU memory
- Operation latency
- Throughput metrics for different block sizes

## Troubleshooting

- If you encounter "File not found" errors during READ operations, ensure you've performed a WRITE operation first
- Check that the directory specified in `--filepath` exists and has proper permissions
- Verify that your system has GPU Direct Storage support enabled
- Make sure the ETCD server is running at the specified endpoint

## Next Steps

- Experiment with different model architectures
- Profile with different quantization modes
- Try different access patterns (block vs. layer)
- Increase ISL or number of concurrent requests
