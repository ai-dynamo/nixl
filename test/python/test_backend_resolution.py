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

from nixl._api import nixl_agent


class _RecordingBinding:
    """Records the backend-handle lists the agent passes to the binding."""

    def __init__(self):
        self.registered = []
        self.deregistered = []

    def registerMem(self, descs, handle_list):
        self.registered.append(list(handle_list))

    def deregisterMem(self, descs, handle_list):
        self.deregistered.append(list(handle_list))


def _bare_agent(backends):
    """An nixl_agent without the C extension: only the Python bookkeeping."""
    agent = nixl_agent.__new__(nixl_agent)
    agent.backends = backends
    agent.plugin_list = list(backends)
    return agent


def test_backends_lists_register_each_engine_once(monkeypatch):
    agent = _bare_agent({"UCX": 11, "POSIX": 22})
    binding = _RecordingBinding()
    agent.agent = binding
    monkeypatch.setattr(agent, "get_reg_descs", lambda reg_list, mem_type: reg_list)

    # Repeated spellings of one backend (mixed case, padding, duplicates) must
    # reach the engine exactly once, in first-appearance order.
    agent.register_memory("descs", backends=["UCX", " ucx ", "POSIX", "UCX"])
    agent.deregister_memory("descs", backends=["posix ", "UCX", "ucx"])

    assert binding.registered == [[11, 22]]
    assert binding.deregistered == [[22, 11]]


def test_backends_lists_keep_raising_on_unknown_spelling(monkeypatch):
    agent = _bare_agent({"UCX": 11})
    agent.agent = _RecordingBinding()
    monkeypatch.setattr(agent, "get_reg_descs", lambda reg_list, mem_type: reg_list)

    with pytest.raises(KeyError):
        agent.register_memory("descs", backends=["UCX", "no_such_backend"])
