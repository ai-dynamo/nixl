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

import uuid

import pytest

import nixl._bindings as bindings
import nixl._utils as utils
from nixl._api import nixl_agent

# NIXL pytest fixtures


@pytest.fixture
def one_ucx_agent():
    return nixl_agent(str(uuid.uuid4()))


@pytest.fixture
def two_ucx_agents():
    return (nixl_agent(str(uuid.uuid4())), nixl_agent(str(uuid.uuid4())))


def test_invalid_backend_name(one_ucx_agent):
    # "UVX" is a typo for "UCX"
    with pytest.raises(bindings.nixlNotFoundError):
        one_ucx_agent.create_backend("UVX")


def test_metadata_pass(two_ucx_agents):
    agent1, agent2 = two_ucx_agents

    addr = utils.malloc_passthru(1024)

    agent1_reg_descs = agent1.get_reg_descs([(addr, 1024, 0, "test")], "DRAM")

    assert agent1.register_memory(agent1_reg_descs) is not None

    passed_name = agent2.add_remote_agent(agent1.get_agent_metadata())
    assert passed_name == agent1.name.encode()


# monkeypatch limits scope of env change to this test
# skipping because plugin manager is only created one time statically
# (changing env here does nothing)
@pytest.mark.skip
def test_incorrect_plugin_env(monkeypatch):
    monkeypatch.setenv("NIXL_PLUGIN_DIR", "some/incorrect/path")

    with pytest.raises(RuntimeError):
        nixl_agent("bad env agent")
