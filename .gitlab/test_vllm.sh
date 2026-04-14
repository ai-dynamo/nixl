#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -ex

echo "==== vLLM/SGLang NixL GPU sanity check ===="

nvidia-smi -L

python3 -c "
import nixl
print('nixl version:', nixl.__version__)
"

python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('CUDA devices:', torch.cuda.device_count())
t = torch.zeros(1024, device='cuda')
print('GPU tensor allocated OK')
"

python3 -c "
import nixl
from nixl import nixlAgent

agent = nixlAgent('vllm_sanity_test')
print('NixL agent created OK')
del agent
print('NixL agent destroyed OK')
"

echo "==== vLLM/SGLang NixL GPU sanity check PASSED ===="
