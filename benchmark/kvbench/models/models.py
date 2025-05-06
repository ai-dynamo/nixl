from abc import ABC, abstractmethod
from typing import Any, Dict
import yaml

class BaseModelArch(ABC):
    """
    Abstract base class defining the interface for model architectures.
    
    All model architectures should inherit from this class and implement
    the required abstract methods.
    """
    
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize a model architecture instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_kv_size_per_token(self) -> int:
        """
        Get the key-value cache size for this model architecture.
        
        Returns:
            int: The size of the key-value cache.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
   
    @abstractmethod
    def get_io_size(self) -> int:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model architecture to a dictionary representation.
        
        Returns:
            Dict[str, Any]: A dictionary containing the model's configuration.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseModelArch':
        """
        Create a model architecture instance from a YAML configuration file.
        
        Args:
            yaml_path (str): Path to the YAML configuration file.
            
        Returns:
            BaseModelArch: An instance of the appropriate model architecture class.
            
        Raises:
            ValueError: If the specified model name is not supported.
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            filtered_dict = {k: v for k, v in config.items() if v is not None}
            model_name = filtered_dict.get('model')
            if "llama3.1" in model_name.lower():
                from models.llama3_1 import Llama3_1
                return Llama3_1(**filtered_dict)    
            elif "deepseek_r1" in model_name.lower():
                from models.deepseek_r1 import DeepSeekR1
                return DeepSeekR1(**filtered_dict)
            else:
                raise ValueError(f"Model name {model_name} not supported")