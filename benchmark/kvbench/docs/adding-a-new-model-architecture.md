# Adding a new model architecture

Adding a new model architecture to the KV benchmark system involves the following steps:

## 1. Create a new model class

Create a new Python file in the `models/` directory. Name it after your model architecture (e.g., `my_model.py`).

Your model class must inherit from `BaseModelArch` and implement all required abstract methods:

```python
from models.models import BaseModelArch
from models.model_config import ModelConfig
from typing import Any, Dict
from models.utils import get_precision_size

class MyModel(BaseModelArch):
    def __init__(self, model: str,
                 # Add other parameters specific to your model architecture
                 model_config: ModelConfig = None):
        self.model = model
        self.model_config = model_config
        # Initialize other model-specific attributes

    def get_kv_size_per_token(self, token_count: int=1) -> int:
        # Calculate and return the KV cache size per token
        # Example implementation:
        return int(
            # Model-specific KV cache size calculation
        )

    def get_io_size(self, page_size: int = 1) -> int:
        # Calculate and return the IO size
        # Example implementation based on your model's architecture

    def to_dict(self) -> Dict[str, Any]:
        # Convert model attributes to a dictionary
        return {
            'model': self.model.lower(),
            # Add other attributes
        }
```

## 2. Register your model in the factory method

Update the `from_yaml` class method in `models/models.py` to include your new model architecture:

```python
@classmethod
def from_yaml(cls, yaml_path: str, model_config: ModelConfig = None) -> 'BaseModelArch':
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        filtered_dict = {k: v for k, v in config.items() if v is not None}
        model_name = filtered_dict.get('model')

        # Add your model to the factory method
        if "llama3.1" in model_name.lower():
            from models.llama3_1 import Llama3_1
            model = Llama3_1(**filtered_dict)
        elif "deepseek_r1" in model_name.lower():
            from models.deepseek_r1 import DeepSeekR1
            model = DeepSeekR1(**filtered_dict)
        elif "my_model" in model_name.lower():  # Add your model here
            from models.my_model import MyModel
            model = MyModel(**filtered_dict)
        else:
            raise ValueError(f"Model name {model_name} not supported")

        # Set model_config if provided
        if model_config is not None:
            model.set_model_config(model_config)

        return model
```

## 3. Create a YAML configuration file

Create a YAML configuration file in the appropriate directory (e.g., `configs/models/my_model.yaml`):

```yaml
model: my_model
# Add other parameters required by your model's __init__ method
# Example:
num_layers: 32
# Other model-specific parameters
```

## 4. Implement the required methods

### 4.1 `get_kv_size_per_token()`

This method should calculate the key-value cache size for your model architecture per token:

```python
def get_kv_size_per_token(self, token_count: int=1) -> int:
    # Example implementation based on Llama 3.1:
    return int(
        self.num_layers * (self.num_heads / self.group_size) * \
        self.head_dim * 2 * get_precision_size(self.model_config.model.model_quant_mode) * token_count
    )
```

### 4.2 `get_io_size()`

Calculate the I/O operations size for your model:

```python
def get_io_size(self, page_size: int = 1) -> int:
    kv_size = self.get_kv_size_per_token()
    # Calculate based on your model's architecture
    # See llama3_1.py for reference implementation
```

### 4.3 `to_dict()`

Serialize your model's attributes to a dictionary:

```python
def to_dict(self) -> Dict[str, Any]:
    return {
        'model': self.model.lower(),
        # Add all relevant model parameters here
    }
```

## 5. Testing your implementation

1. Create a model configuration file for your model
2. Initialize your model using the `BaseModelArch.from_yaml()` method
3. Verify the KV cache size calculations are correct
4. Run the benchmark with your new model architecture

## Example

For a complete implementation example, refer to the `Llama3_1` class in `models/llama3_1.py`.
