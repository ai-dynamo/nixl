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

import pytest
import yaml

from . import inference_workload_matgen as matgen


def test_generated_workload_respects_batch_memory_budget(tmp_path):
    # Each fixed-size request needs 48 bytes of KV. Exactly two fit.
    task = matgen.TaskConfig(batch_size=8, min_isl=3, max_isl=3, max_batch_mem=96)
    model = matgen.ModelConfig(hidden_size=4, num_layers=1)
    worker = matgen.WorkerConfig(tp=1)
    matgen.main(3, task, 1, 1, worker, worker, model, results_dir=tmp_path)
    patterns = yaml.safe_load((tmp_path / "metadata.yaml").read_text())[
        "traffic_patterns"
    ]
    batches = [p["metadata"]["isl"] for p in patterns]
    if batches != [6, 3]:
        pytest.fail(f"Expected batches of six and three tokens, got {batches}")

    task.max_batch_mem = 47
    with pytest.raises(ValueError, match="exceeds the batch memory budget"):
        matgen.main(1, task, 1, 1, worker, worker, model, results_dir=tmp_path)
