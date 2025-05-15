# Creating a Model Configuration

This guide explains how to create a model configuration for KVBench.

## Configuration Structure

A model configuration is defined in a YAML file with three main sections:

1. `strategy`: Defines model parallelism and quantization settings
2. `runtime`: Specifies batch size and sequence length parameters
3. `system`: Configures hardware and backend specifications

## Required Parameters

### Strategy Configuration

```yaml
strategy:
  tp_size: 1                # Tensor parallelism size
  pp_size: 1                # Pipeline parallelism size
  model_quant_mode: "fp8"   # Model weight quantization mode
  kvcache_quant_mode: "fp8" # KV cache quantization mode
```

### Runtime Configuration

```yaml
runtime:
  isl: 8192                 # Input sequence length
  osl: 2048                 # Output sequence length
  num_requests: 10          # Number of requests to process
```

### System Configuration

```yaml
system:
  hardware: "H100"          # Hardware platform (e.g., "H100", "A100")
  backend: "SGLANG"         # Inference backend to use
  access_pattern: "block"   # KV cache access pattern ("block" or "layer")
  page_size: 16             # Page size for access pattern
```

## Example Configurations

### Block Access Pattern

This configuration uses a block access pattern, where KV cache is accessed in blocks:

```yaml
strategy:
  tp_size: 1
  pp_size: 1
  model_quant_mode: "fp8"
  kvcache_quant_mode: "fp8"

runtime:
  isl: 8192
  osl: 2048
  num_requests: 10
system:
  hardware: "H100"
  backend: "SGLANG"
  access_pattern: "block"
  page_size: 16
```

### Layer Access Pattern

This configuration uses a layer access pattern, where KV cache is accessed layer by layer:

```yaml
strategy:
  tp_size: 1
  pp_size: 1
  model_quant_mode: "fp8"
  kvcache_quant_mode: "fp8"
runtime:
  isl: 8192
  osl: 2048
  num_requests: 10
system:
  hardware: "H100"
  backend: "SGLANG"
  access_pattern: "layer"
  page_size: 16
```

## Using Configurations in Code

You can load a model configuration using the `ModelConfig` class:

```python
from models.model_config import ModelConfig

# Load from a single YAML file
config = ModelConfig.from_yaml("path/to/config.yaml")

# Load and merge multiple YAML files (later files override earlier ones)
config = ModelConfig.from_yaml_files(["base_config.yaml", "override_config.yaml"])

# Create from a dictionary
config_dict = {
    'strategy': {'tp_size': 2, 'pp_size': 2},
    'runtime': {'isl': 4096},
    'system': {'hardware': 'A100'}
}
config = ModelConfig.from_dict(config_dict)

# Update an existing configuration
updated_config = config.update({
    'strategy': {'tp_size': 4}
})

# Save configuration to a YAML file
config.to_yaml("new_config.yaml")
```

## Parameter Descriptions

### Strategy Configuration

- `tp_size`: Tensor parallelism size - number of GPUs for tensor-parallel execution
- `pp_size`: Pipeline parallelism size - number of GPUs for pipeline-parallel execution
- `model_quant_mode`: Quantization mode for model weights (e.g., "fp8", "fp16", "int8")
- `kvcache_quant_mode`: Quantization mode for KV cache (e.g., "fp8", "fp16", "int8")

### Runtime Configuration

- `isl`: Input sequence length - maximum length of input sequences
- `osl`: Output sequence length - maximum length of generated sequences
- `num_requests`: Number of inference requests to process

### System Configuration

- `hardware`: Hardware platform (e.g., "H100", "A100")
- `backend`: Inference backend engine (e.g., "SGLANG", "TensorRT")
- `access_pattern`: KV cache access pattern ("block" or "layer")
- `page_size`: Page size when using block access pattern

## Best Practices

1. Start with an existing configuration example and modify it for your needs
2. Use descriptive file names that indicate the strategy and system being used
3. Organize configurations into directories by model architecture or hardware target
4. Store common base configurations separately and override them for specific scenarios
