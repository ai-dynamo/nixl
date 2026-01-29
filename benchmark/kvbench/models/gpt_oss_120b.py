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

from typing import Any, Dict, Optional

import yaml  # type: ignore
from models.model_config import ModelConfig
from models.models import BaseModelArch
from models.utils import get_precision_size


class GptOss120B(BaseModelArch):
    """
    Implementation of the GPT-OSS-120B model architecture.

    This class represents the GPT-OSS-120B model and provides methods
    to access its parameters and configuration.
    """

    def __init__(
        self,
        model_name: str,
        num_layers: int,
        num_query_heads_with_mha: int,
        query_head_dimension: int,
        gqa_num_queries_in_group: int,
        num_model_params: int,
        model_config: Optional[ModelConfig] = None,
    ):
        """
        Initialize a GPT-OSS-120B model architecture.

        Args:
            model_name (str): The model identifier.
            num_layers (int): Number of transformer layers.
            num_query_heads_with_mha (int): Number of query heads with multi-head attention.
            query_head_dimension (int): Dimension of each query head.
            gqa_num_queries_in_group (int): Number of queries in a group for grouped-query attention.
            num_model_params (int): Total number of model parameters.
        """
        self.model_name = model_name
        self.model_config = model_config or ModelConfig()
        self.num_layers = num_layers
        self.num_query_heads_with_mha = num_query_heads_with_mha
        self.query_head_dimension = query_head_dimension
        self.gqa_num_queries_in_group = gqa_num_queries_in_group
        self.num_model_params = num_model_params
        self.model_dimension = self.num_query_heads_with_mha * self.query_head_dimension

    def get_kv_size_per_token(self, token_count: int = 1) -> int:
        """
        Get the key-value cache size for the GPT-OSS-120B model (per token).

        Returns:
            int: The size of the key-value cache.
        """
        return int(
            self.num_layers
            * (self.num_query_heads_with_mha / self.gqa_num_queries_in_group)
            * self.query_head_dimension
            * 2
            * get_precision_size(self.model_config.model.model_quant_mode)
            * token_count
        )

    def get_io_size(self, page_size: int = 1) -> int:
        """
        Calculates the number of IOPs for one token per GPU

        Returns:
            int: The number of IOPs.
        """

        kv_size = self.get_kv_size_per_token()
        # we need the size of kv per token per attention layer
        kv_size = int(kv_size / self.num_layers)
        if kv_size <= 0:
            raise ValueError("Invalid KV Size: 0")
        io_size = int(kv_size / self.model_config.model.tp_size)
        if self.model_config.system.access_pattern == "block":
            io_size = int(io_size * (self.num_layers / self.model_config.model.pp_size))
        return int(io_size * page_size)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the GPT-OSS-120B model configuration to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing all model configuration parameters.
        """
        return {
            "model_name": self.model_name.lower(),
            "num_layers": self.num_layers,
            "num_query_heads_with_mha": self.num_query_heads_with_mha,
            "query_head_dimension": self.query_head_dimension,
            "gqa_num_queries_in_group": self.gqa_num_queries_in_group,
            "num_model_params": self.num_model_params,
            "model_dimension": self.model_dimension,
        }

    def __str__(self) -> str:
        """
        Get a string representation of the GPT-OSS-120B model.

        Returns:
            str: YAML formatted string of the model configuration.
        """
        return yaml.dump(self.to_dict())
