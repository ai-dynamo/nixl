from models.models import BaseModelArch
from typing import Any, Dict
import yaml

class Llama3_1(BaseModelArch):
    """
    Implementation of the Llama 3.1 model architecture.
    
    This class represents the Llama 3.1 model and provides methods
    to access its parameters and configuration.
    """
    
    def __init__(self, model: str, num_layers: int,
                num_query_heads_with_mha: int,
                query_head_dimension: int,
                gqa_num_queries_in_group: int,
                num_model_params: int):
        """
        Initialize a Llama 3.1 model architecture.
        
        Args:
            model (str): The model identifier.
            num_layers (int): Number of transformer layers.
            num_query_heads_with_mha (int): Number of query heads with multi-head attention.
            query_head_dimension (int): Dimension of each query head.
            gqa_num_queries_in_group (int): Number of queries in a group for grouped-query attention.
            num_model_params (int): Total number of model parameters.
        """
        self.model = model
        self.num_layers = num_layers
        self.num_query_heads_with_mha = num_query_heads_with_mha
        self.query_head_dimension = query_head_dimension
        self.gqa_num_queries_in_group = gqa_num_queries_in_group
        self.num_model_params = num_model_params
        self.model_dimension = self.num_query_heads_with_mha * self.query_head_dimension

    def get_kv_size(self) -> int:
        """
        Get the key-value cache size for the Llama 3.1 model.
        
        Returns:
            int: The size of the key-value cache, currently hardcoded to 1.
        """
        return 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Llama 3.1 model configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: A dictionary containing all model configuration parameters.
        """
        return {
            'model': self.model.lower(),
            'num_layers': self.num_layers,
            'num_query_heads_with_mha': self.num_query_heads_with_mha,
            'query_head_dimension': self.query_head_dimension,
            'gqa_num_queries_in_group': self.gqa_num_queries_in_group,
            'num_model_params': self.num_model_params,
            'model_dimension': self.model_dimension
        }

    def __str__(self) -> str:
        """
        Get a string representation of the Llama 3.1 model.
        
        Returns:
            str: YAML formatted string of the model configuration.
        """
        return yaml.dump(self.to_dict())