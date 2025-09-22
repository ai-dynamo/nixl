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
import uuid

import pytest
import torch

import nixl._bindings as bindings
import nixl._utils as utils
from nixl._api import nixl_agent, nixl_agent_config

# NIXL pytest fixtures


@pytest.fixture()
def one_empty_agent():
    config = nixl_agent_config(backends=[])
    return nixl_agent(str(uuid.uuid4()), config)


@pytest.fixture
def one_ucx_agent():
    return nixl_agent(str(uuid.uuid4()))


@pytest.fixture
def two_ucx_agents():
    return (nixl_agent(str(uuid.uuid4())), nixl_agent(str(uuid.uuid4())))


@pytest.fixture
def two_connected_ucx_agents():
    agent1, agent2 = (nixl_agent(str(uuid.uuid4())), nixl_agent(str(uuid.uuid4())))
    agent1.add_remote_agent(agent2.get_agent_metadata())
    agent2.add_remote_agent(agent1.get_agent_metadata())
    yield (agent1, agent2)
    agent1.remove_remote_agent(agent2.name)
    agent2.remove_remote_agent(agent1.name)


@pytest.fixture
def one_reg_list():
    return bindings.nixlRegDList(bindings.DRAM_SEG)


@pytest.fixture
def one_xfer_list():
    return bindings.nixlXferDList(bindings.DRAM_SEG)


@pytest.fixture
def two_xfer_lists():
    return (
        bindings.nixlXferDList(bindings.DRAM_SEG),
        bindings.nixlXferDList(bindings.DRAM_SEG),
    )


def test_empty_agent_name():
    # pybind11 std::invalid_argument translates to python ValueError
    with pytest.raises(ValueError):
        nixl_agent("")


# This test passses locally, but fails in CI because GDS is confused about CUDA installation.
# Skipping until we have a CI that is compatible with GDS.
@pytest.mark.skip
def test_instantiate_all():
    agent1 = nixl_agent("test", nixl_conf=None, instantiate_all=True)

    assert len(agent1.plugin_list) == len(agent1.backends)


def test_make_invalid_op(one_empty_agent, two_xfer_lists):
    # Only READ/WRITE are supported
    with pytest.raises(KeyError):
        one_empty_agent.make_prepped_xfer("RD", 0, [], 0, [])

    list1, list2 = two_xfer_lists
    with pytest.raises(KeyError):
        one_empty_agent.initialize_xfer("WR", list1, list2, "nobody")


def test_invalid_plugin_name(one_ucx_agent):
    # "UVX" is a typo for "UCX"
    plugin_mems = one_ucx_agent.get_plugin_mem_types("UVX")
    plugin_params = one_ucx_agent.get_plugin_params("UVX")

    backend_mems = one_ucx_agent.get_backend_mem_types("UVX")
    backend_params = one_ucx_agent.get_backend_mem_types("UVX")

    assert len(plugin_mems) == 0 and len(plugin_params) == 0
    assert len(backend_mems) == 0 and len(backend_params) == 0


def test_invalid_backend_name_creation(one_ucx_agent):
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
    utils.free_passthru(addr)


@pytest.mark.timeout(5)
def test_empty_notif_tag(two_connected_ucx_agents):
    agent1, agent2 = two_connected_ucx_agents

    agent1.send_notif(agent2.name, b"whatever")

    found = False
    while not found:
        # empty bytes will consume any message
        found = agent2.check_remote_xfer_done(agent1.name, b"")


def test_improper_get_xfer_descs(one_empty_agent, one_reg_list):
    # xfer list should be 3-tuple, not 4-tuple
    bad_list = [(1, 2, 3, 4)]
    ok_list = [(1, 2, 3)]

    ret = one_empty_agent.get_xfer_descs(bad_list)
    assert ret is None

    # With 3-tuple list, mem_type must be specified
    ret = one_empty_agent.get_xfer_descs(ok_list, mem_type=None)
    assert ret is None

    # Invalid memory types will give a key error
    with pytest.raises(KeyError):
        ret = one_empty_agent.get_xfer_descs(ok_list, mem_type="V-RAM")

    # Passing reg list will not work
    ret = one_empty_agent.get_xfer_descs(one_reg_list)
    assert ret is None


def test_improper_get_reg_descs(one_empty_agent, one_xfer_list):
    # reg list should be 4-tuple, not 3-tuple
    bad_list = [(1, 2, 3)]
    ok_list = [(1, 2, 3, 4)]

    ret = one_empty_agent.get_reg_descs(bad_list)
    assert ret is None

    # With 4-tuple list, mem_type must be specified
    ret = one_empty_agent.get_reg_descs(ok_list, mem_type=None)
    assert ret is None

    # Invalid memory types will give a key error
    with pytest.raises(KeyError):
        ret = one_empty_agent.get_reg_descs(ok_list, mem_type="V-RAM")

    # Passing reg list will not work
    ret = one_empty_agent.get_reg_descs(one_xfer_list)
    assert ret is None


def test_noncontiguous_tensor(one_empty_agent):
    cont_tensor = torch.arange(8).reshape(2, 4)
    non_cont_tensor = torch.transpose(cont_tensor, 0, 1)
    assert non_cont_tensor.is_contiguous() is False

    reg_descs = one_empty_agent.get_reg_descs(non_cont_tensor)
    assert reg_descs is None

    xfer_descs = one_empty_agent.get_xfer_descs(non_cont_tensor)
    assert xfer_descs is None


# monkeypatch limits scope of env change to this test
# skipping because plugin manager is only created one time statically
# (changing env here does nothing)
@pytest.mark.skip
def test_incorrect_plugin_env(monkeypatch):
    monkeypatch.setenv("NIXL_PLUGIN_DIR", "some/incorrect/path")

    with pytest.raises(RuntimeError):
        nixl_agent("bad env agent")


def test_get_xfer_telemetry():
    os.environ["NIXL_TELEMETRY_ENABLE"] = "y"

    agent1 = nixl_agent(str(uuid.uuid4()))
    agent2 = nixl_agent(str(uuid.uuid4()))

    mem_size = 128
    addr1 = utils.malloc_passthru(mem_size)
    addr2 = utils.malloc_passthru(mem_size)

    try:
        reg1 = agent1.get_reg_descs([(addr1, mem_size, 0, "")], mem_type="DRAM")
        reg2 = agent2.get_reg_descs([(addr2, mem_size, 0, "")], mem_type="DRAM")
        agent1.register_memory(reg1)
        agent2.register_memory(reg2)

        agent1.add_remote_agent(agent2.get_agent_metadata())
        src = agent1.get_xfer_descs(
            [(addr1, mem_size // 2, 0), (addr1 + mem_size // 2, mem_size // 2, 0)],
            mem_type="DRAM",
        )
        dst = agent1.get_xfer_descs(
            [(addr2, mem_size // 2, 0), (addr2 + mem_size // 2, mem_size // 2, 0)],
            mem_type="DRAM",
        )

        handle = agent1.initialize_xfer("WRITE", src, dst, agent2.name)
        st = agent1.transfer(handle)
        assert st in ("DONE", "PROC")

        while True:
            st = agent1.check_xfer_state(handle)
            assert st in ("DONE", "PROC")
            if st == "DONE":
                break

        telem = agent1.get_xfer_telemetry(handle)
        assert telem.descCount == 2
        assert telem.totalBytes == mem_size
        assert telem.startTime > 0
        assert telem.postDuration > 0
        assert telem.xferDuration > 0
        assert telem.xferDuration >= telem.postDuration

        agent1.release_xfer_handle(handle)
    finally:
        utils.free_passthru(addr1)
        utils.free_passthru(addr2)
        os.environ.pop("NIXL_TELEMETRY_ENABLE")


def test_get_xfer_telemetry_cfg():
    os.environ["NIXL_TELEMETRY_ENABLE"] = "m"  # invalid value not to enable
    os.environ["NIXL_TELEMETRY_DIR"] = "/tmp/dummy"  # to be ignored

    agent1 = nixl_agent(
        str(uuid.uuid4()), nixl_conf=nixl_agent_config(capture_telemetry=True)
    )
    agent2 = nixl_agent(str(uuid.uuid4()))

    mem_size = 128
    addr1 = utils.malloc_passthru(mem_size)
    addr2 = utils.malloc_passthru(mem_size)

    try:
        reg1 = agent1.get_reg_descs([(addr1, mem_size, 0, "")], mem_type="DRAM")
        reg2 = agent2.get_reg_descs([(addr2, mem_size, 0, "")], mem_type="DRAM")
        agent1.register_memory(reg1)
        agent2.register_memory(reg2)

        agent1.add_remote_agent(agent2.get_agent_metadata())
        src = agent1.get_xfer_descs(
            [(addr1, mem_size // 2, 0), (addr1 + mem_size // 2, mem_size // 2, 0)],
            mem_type="DRAM",
        )
        dst = agent1.get_xfer_descs(
            [(addr2, mem_size // 2, 0), (addr2 + mem_size // 2, mem_size // 2, 0)],
            mem_type="DRAM",
        )

        handle = agent1.initialize_xfer("WRITE", src, dst, agent2.name)
        st = agent1.transfer(handle)
        assert st in ("DONE", "PROC")

        while True:
            st = agent1.check_xfer_state(handle)
            assert st in ("DONE", "PROC")
            if st == "DONE":
                break

        telem = agent1.get_xfer_telemetry(handle)
        assert telem.descCount == 2
        assert telem.totalBytes == mem_size
        assert telem.startTime > 0
        assert telem.postDuration > 0
        assert telem.xferDuration > 0
        assert telem.xferDuration >= telem.postDuration

        agent1.release_xfer_handle(handle)
    finally:
        utils.free_passthru(addr1)
        utils.free_passthru(addr2)
        os.environ.pop("NIXL_TELEMETRY_ENABLE")
        os.environ.pop("NIXL_TELEMETRY_DIR")
