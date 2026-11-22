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


def test_prefill_delay_adds_independent_requests(tmp_path, monkeypatch):
    requests = iter([matgen.UserRequest(2), matgen.UserRequest(3)])
    monkeypatch.setattr(matgen.UserRequest, "rand", lambda *args: next(requests))
    matgen.main(
        num_user_requests=2,
        task_config=matgen.TaskConfig(batch_size=2),
        num_prefill_gpus=1,
        num_decode_gpus=1,
        prefill_worker_config=matgen.WorkerConfig(tp=1),
        decode_worker_config=matgen.WorkerConfig(tp=1),
        model_config=matgen.ModelConfig(hidden_size=4, num_layers=1),
        results_dir=tmp_path,
        flops_per_gpu=32,
    )
    workload = yaml.safe_load((tmp_path / "metadata.yaml").read_text())
    (pattern,) = workload["traffic_patterns"]
    # The two individual prefill estimates are 18 and 28.5 seconds.
    if pattern["sleep_before_launch_sec"] != 46.5:
        pytest.fail(f"Unexpected prefill delay: {pattern['sleep_before_launch_sec']}")
