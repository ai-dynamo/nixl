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

import math
from models.models import BaseModelArch
from models.model_config import ModelConfig

def get_precision_size(precision: str) -> int:
    if precision == "fp8":
        return 1
    elif precision == "int8":
        return 1
    elif precision == "fp16":
        return 2
    elif precision == "bfloat16":
        return 2
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
def get_batch_size(model: BaseModelArch, model_config: ModelConfig, io_size: int):
    return math.ceil(
        (model.get_kv_size_per_token(model_config.runtime.isl)/io_size)
        * model_config.runtime.num_requests
    )