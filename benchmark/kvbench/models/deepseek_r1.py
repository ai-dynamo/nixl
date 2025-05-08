from models.models import BaseModelArch
from models.utils import get_precision_size
from models.model_config import ModelConfig
from typing import Any, Dict
import yaml

class DeepSeekR1(BaseModelArch):
    """
    Implementation of the DeepSeek-R1 model architecture.
    
    This class represents the DeepSeek-R1 model and provides methods
    to access its parameters and configuration.
    """
    
    def __init__(self, model: str, num_layers: int,
                num_query_heads: int,
                query_head_dimension: int,
                embedding_dimension: int,
                rope_mla_dimension: int,
                mla_latent_vector_dimension: int,
                num_model_params: int,
                model_config: ModelConfig = None):
        """
        Initialize a DeepSeek-R1 model architecture.
        
        Args:
            model (str): The model identifier.
            num_layers (int): Number of transformer layers.
            num_query_heads (int): Number of query heads.
            query_head_dimension (int): Dimension of each query head.
            embedding_dimension (int): Dimension of token embeddings.
            rope_mla_dimension (int): Dimension for rotary position embedding in MLA.
            mla_latent_vector_dimension (int): Dimension of the latent vectors in multi-linear attention.
            num_model_params (int): Total number of model parameters.
        """
        
        self.model = model
        self.model_config = model_config
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.query_head_dimension = query_head_dimension
        self.embedding_dimension = embedding_dimension
        self.rope_mla_dimension = rope_mla_dimension
        self.mla_latent_vector_dimension = mla_latent_vector_dimension
        self.num_model_params = num_model_params

    def get_kv_size_per_token(self, token_count: int = 1) -> int:
        """
        Get the key-value cache size for the DeepSeek-R1 model.
        
        Returns:
            int: The size of the key-value cache, currently hardcoded to 1.
        """

        #         ( rope_mla_dimension + mla mla_latent_vector_dimension )
        # * quantization * num_layers
        return int((self.rope_mla_dimension + self.mla_latent_vector_dimension) * 
                get_precision_size(self.model_config.model.kvcache_quant_mode) * 
                self.num_layers)*token_count
    
    def get_io_size(self, sequence_length: int) -> int:
        """
        Get the input/output size for the Llama 3.1 model.
        
        Returns:
            int: The size of the input/output cache, currently hardcoded to 1.
        """
        return int((sequence_length/self.model_config.runtime.batch_size))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DeepSeek-R1 model configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: A dictionary containing all model configuration parameters.
        """
        return {
            'model': self.model.lower(),
            'num_layers': self.num_layers,
            'num_query_heads': self.num_query_heads,
            'query_head_dimension': self.query_head_dimension,
            'embedding_dimension': self.embedding_dimension,
            'rope_mla_dimension': self.rope_mla_dimension,
            'mla_latent_vector_dimension': self.mla_latent_vector_dimension,
            'num_model_params': self.num_model_params
        }

    def __str__(self) -> str:
        """
        Get a string representation of the DeepSeek-R1 model.
        
        Returns:
            str: YAML formatted string of the model configuration.
        """
        return yaml.dump(self.to_dict())