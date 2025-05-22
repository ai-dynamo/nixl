# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from models.models import BaseModelArch
from models.utils import get_precision_size
from models.model_config import ModelConfig
from typing import Any, Dict
import math
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
    
    def get_io_size(self, page_size: int = 1) -> int:
        """
        Calculates the size (bytes) of an IO request for the DeepSeek-R1 model.
 
        Returns:
            int: The number of bytes in an IO request.
        """

        kv_size = self.get_kv_size_per_token()

        if kv_size <= 0:
            raise ValueError("Invalid KV Size: 0")

        # we need the size of kv per token per attention layer 
        kv_size = (kv_size / self.num_layers)
        io_size = kv_size / self.model_config.model.tp_size
        if self.model_config.system.access_pattern == 'block':
            io_size = io_size * math.ceil(self.num_layers/self.model_config.model.pp_size)
        return int(io_size * page_size)

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
