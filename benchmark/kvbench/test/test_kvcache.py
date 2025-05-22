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

import os
import sys
from pathlib import Path

import pytest  # type: ignore

p = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, p + "/../")


# Define fixtures for file paths
@pytest.fixture
def examples_dir():
    return Path(p).parent / "examples"


@pytest.mark.parametrize(
    "model_arch, model_config, expected_kv_size",
    [
        ("model_llama_3_1_8b.yaml", "block-tp1-pp1.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "block-tp1-pp2.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "block-tp1-pp4.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "block-tp1-pp8.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "block-tp8-pp2.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "block-tp8-pp4.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "block-tp8-pp8.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "block-tp8-pp16.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "block-tp8-pp32.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "layer-tp1-pp1.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "layer-tp1-pp2.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "layer-tp1-pp4.yaml", (1, 65536)),
        ("model_llama_3_1_8b.yaml", "layer-tp1-pp8.yaml", (1, 65536)),
        ("model_llama_3_1_70b.yaml", "block-tp1-pp1.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "block-tp1-pp2.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "block-tp1-pp4.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "block-tp1-pp8.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "block-tp8-pp2.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "block-tp8-pp4.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "block-tp8-pp8.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "block-tp8-pp16.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "block-tp8-pp32.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "layer-tp1-pp1.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "layer-tp1-pp2.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "layer-tp1-pp4.yaml", (1, 163840)),
        ("model_llama_3_1_70b.yaml", "layer-tp1-pp8.yaml", (1, 163840)),
        ("model_deepseek_r1.yaml", "block-tp8-pp2.yaml", (1, 35136)),
        ("model_deepseek_r1.yaml", "block-tp8-pp4.yaml", (1, 35136)),
        ("model_deepseek_r1.yaml", "block-tp8-pp8.yaml", (1, 35136)),
        ("model_deepseek_r1.yaml", "block-tp8-pp16.yaml", (1, 35136)),
        ("model_deepseek_r1.yaml", "block-tp8-pp32.yaml", (1, 35136)),
    ],
)
def test_kvcache_size(examples_dir, model_arch, model_config, expected_kv_size):
    from models.models import BaseModelArch, ModelConfig

    model = BaseModelArch.from_yaml(examples_dir / model_arch)
    config = ModelConfig.from_yaml(examples_dir / model_config)
    model.set_model_config(config)

    assert model is not None
    assert model.get_kv_size_per_token(expected_kv_size[0]) == expected_kv_size[1]


@pytest.mark.parametrize(
    "model_arch, model_config, expected_io_size",
    [
        ("model_llama_3_1_70b.yaml", "block-tp8-pp2.yaml", 10240),
        ("model_llama_3_1_70b.yaml", "block-tp8-pp4.yaml", 5120),
        ("model_llama_3_1_70b.yaml", "block-tp8-pp8.yaml", 2560),
        ("model_llama_3_1_70b.yaml", "block-tp8-pp16.yaml", 1280),
        ("model_llama_3_1_70b.yaml", "block-tp8-pp32.yaml", 640),
        ("model_llama_3_1_70b.yaml", "layer-tp8-pp2.yaml", 256),
        ("model_llama_3_1_70b.yaml", "layer-tp8-pp4.yaml", 256),
        ("model_llama_3_1_70b.yaml", "layer-tp8-pp8.yaml", 256),
        ("model_llama_3_1_70b.yaml", "layer-tp8-pp16.yaml", 256),
        ("model_llama_3_1_70b.yaml", "layer-tp8-pp32.yaml", 256),
    ],
)
def test_io_size(examples_dir, model_arch, model_config, expected_io_size):
    from models.models import BaseModelArch, ModelConfig

    model = BaseModelArch.from_yaml(examples_dir / model_arch)
    config = ModelConfig.from_yaml(examples_dir / model_config)
    model.set_model_config(config)

    assert model is not None
    assert model.get_io_size() == expected_io_size
