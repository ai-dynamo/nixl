# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tempfile
import uuid

import pytest
import torch

import nixl._bindings as bindings
import nixl._utils as utils
from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

# NIXL pytest fixtures


@pytest.fixture()
def one_empty_agent():
    config = nixl_agent_config(backends=[])
    return nixl_agent(str(uuid.uuid4()), config)


@pytest.fixture
def one_agent(backend_name):
    return nixl_agent(
        str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
    )


@pytest.fixture
def two_agents(backend_name):
    return (
        nixl_agent(
            str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
        ),
        nixl_agent(
            str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
        ),
    )


@pytest.fixture
def two_connected_agents(backend_name):
    agent1, agent2 = (
        nixl_agent(
            str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
        ),
        nixl_agent(
            str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
        ),
    )
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


@pytest.mark.parametrize("sync_mode", [None] + list(nixl_thread_sync_t))
@pytest.mark.parametrize("enable_listen", [True, False])
def test_sync_mode_agent(monkeypatch, sync_mode, enable_listen):
    captured = {}
    real_ctor = bindings.nixlAgent

    def spy_ctor(agent_name, agent_config):
        captured["syncMode"] = agent_config.syncMode
        return real_ctor(agent_name, agent_config)

    monkeypatch.setattr(bindings, "nixlAgent", spy_ctor)
    # listen_port=0 lets the OS pick an ephemeral port, so concurrent agents (parametrized
    # cases here, or parallel CI jobs sharing a node) don't collide on the fixed default port.
    config = nixl_agent_config(
        sync_mode=sync_mode, enable_listen_thread=enable_listen, listen_port=0
    )
    nixl_agent(str(uuid.uuid4()), nixl_conf=config)
    if sync_mode is not None:
        assert captured["syncMode"] == sync_mode.value
    elif enable_listen:
        assert captured["syncMode"] == nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT.value
    else:
        assert captured["syncMode"] == nixl_thread_sync_t.NIXL_THREAD_SYNC_NONE.value


def test_nixl_conf_bad_sync_mode():
    with pytest.raises(TypeError, match="sync_mode must be a nixl_thread_sync_t"):
        nixl_agent_config(sync_mode=1)


def test_make_invalid_op(one_empty_agent, two_xfer_lists):
    # Only READ/WRITE are supported
    with pytest.raises(KeyError):
        one_empty_agent.make_prepped_xfer("RD", 0, [], 0, [])

    list1, list2 = two_xfer_lists
    with pytest.raises(KeyError):
        one_empty_agent.initialize_xfer("WR", list1, list2, "nobody")


def test_invalid_plugin_name(one_agent):
    # "UVX" is a typo for "UCX"
    plugin_mems = one_agent.get_plugin_mem_types("UVX")
    plugin_params = one_agent.get_plugin_params("UVX")

    backend_mems = one_agent.get_backend_mem_types("UVX")
    backend_params = one_agent.get_backend_mem_types("UVX")

    assert len(plugin_mems) == 0 and len(plugin_params) == 0
    assert len(backend_mems) == 0 and len(backend_params) == 0


def test_invalid_backend_name_creation(one_agent):
    # "UVX" is a typo for "UCX"
    with pytest.raises(bindings.nixlNotFoundError):
        one_agent.create_backend("UVX")


def test_metadata_pass(two_agents):
    agent1, agent2 = two_agents

    addr = utils.malloc_passthru(1024)

    agent1_reg_descs = agent1.get_reg_descs([(addr, 1024, 0, "test")], "DRAM")

    assert agent1.register_memory(agent1_reg_descs) is not None

    passed_name = agent2.add_remote_agent(agent1.get_agent_metadata())
    assert passed_name == agent1.name.encode()
    utils.free_passthru(addr)


@pytest.mark.timeout(5, func_only=True)
def test_empty_notif_tag(two_connected_agents):
    agent1, agent2 = two_connected_agents

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


def _prep_mem_view_worker(rank, exch_dir, backend_name):
    # One process per GPU. The remote overload of prep_mem_view needs a real
    # device-capable lane (cuda_ipc) to the peer, which is an inter-process /
    # inter-GPU mechanism -- a same-process, same-GPU loopback has no such lane
    # (UCX v1.21.x tolerated it, newer UCX rejects it with "lane not found for
    # element 0"). So each rank runs in its own process, on its own GPU, and the
    # two connect for real -- mirroring examples/device/ep.
    import json
    import sys
    import time

    peer = 1 - rank
    torch.cuda.set_device(rank)
    # Allocate the VRAM buffer (and its CUDA context) BEFORE creating the agent,
    # so the UCX worker comes up GPU-capable.
    buf = torch.zeros(1024, dtype=torch.uint8, device=f"cuda:{rank}")
    torch.cuda.synchronize()
    addr, size, dev = buf.data_ptr(), buf.numel(), rank
    name = f"pmv_agent_{rank}"

    agent = nixl_agent(name, nixl_agent_config(backends=[backend_name]))
    # Register BEFORE publishing metadata: get_agent_metadata() snapshots the
    # local section at call time, so the buffer must already be registered for
    # the peer to find a matching backend (else NIXL_ERR_NOT_FOUND).
    agent.register_memory(agent.get_reg_descs([(addr, size, dev, "")], mem_type="VRAM"))

    def _publish(tag, data=None):
        path = os.path.join(exch_dir, f"{tag}_{rank}")
        mode, payload = ("wb", data) if isinstance(data, bytes) else ("w", data or "")
        with open(path, mode) as f:
            f.write(payload)

    def _await(tag):
        path = os.path.join(exch_dir, f"{tag}_{peer}")
        while not os.path.exists(path):
            time.sleep(0.05)
        return path

    # File-based rendezvous: exchange agent metadata + the peer's buffer coords.
    _publish("md", agent.get_agent_metadata())
    _publish("info", json.dumps({"name": name, "addr": addr, "size": size, "dev": dev}))
    _publish("published")
    _await("published")
    with open(os.path.join(exch_dir, f"md_{peer}"), "rb") as f:
        peer_md = f.read()
    with open(os.path.join(exch_dir, f"info_{peer}")) as f:
        peer_info = json.load(f)
    agent.add_remote_agent(peer_md)

    # Establish the endpoint BEFORE prep_mem_view: the remote device mem-list needs
    # a live cuda_ipc lane, which only exists once the peers are connected. A notif
    # round-trip forces the wire-up to complete.
    agent.make_connection(peer_info["name"])
    agent.send_notif(peer_info["name"], b"connect")
    deadline = time.time() + 20
    while peer_info["name"] not in agent.get_new_notifs():
        assert time.time() < deadline, f"rank {rank}: no notif from peer"
        time.sleep(0.05)
    _publish("connected")
    _await("connected")

    # Local overload: a nixlXferDList describing this rank's own buffer.
    local_mvh = agent.prep_mem_view(
        agent.get_xfer_descs([(addr, size, dev)], mem_type="VRAM")
    )
    assert isinstance(local_mvh, int)
    assert local_mvh != 0

    # Remote overload: a nixlRemoteDList describing the peer's buffer on the peer's GPU.
    remote_descs = agent.get_remote_descs(
        [(peer_info["addr"], peer_info["size"], peer_info["dev"], peer_info["name"])],
        mem_type="VRAM",
    )
    remote_mvh = agent.prep_mem_view(remote_descs)
    assert isinstance(remote_mvh, int)
    assert remote_mvh != 0

    agent.release_mem_view(local_mvh)
    agent.release_mem_view(remote_mvh)

    # Final barrier: the remote overload reaches into the PEER's endpoint, so
    # neither rank may tear down until BOTH have finished every NIXL op --
    # otherwise one rank's exit kills the endpoint the other still needs.
    _publish("done")
    _await("done")

    # Both ranks are done. Exit hard, skipping Python/C++/UCX/CUDA destructors:
    # tearing those down in a spawned worker can segfault on exit (fragile
    # ordering / cross-process UCX disconnect), which would flake the test even
    # though the checks above passed. A raised assertion above still propagates
    # normally (mp.spawn writes an error file before this point).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    not bindings.HAVE_UCX_GPU_DEVICE_API,
    reason="prep_mem_view requires NIXL built against a UCX with the GPU device API",
)
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="prep_mem_view's remote overload needs a real cuda_ipc device peer, "
    "i.e. two GPUs driven by two processes",
)
def test_prep_mem_view(backend_name, capfd):
    # Two processes, one GPU each: this is the setup the UCX GPU device API is
    # built for (real cuda_ipc peers). Each worker asserts internally; mp.spawn
    # re-raises any worker failure here, so the test fails if either rank fails.
    #
    # capfd.disabled(): pytest captures stdout/stderr at the file-descriptor
    # level, and the spawned children write there from C (UCX/CUDA) -- leaving
    # that capture in place segfaults the children on teardown. Disabling it
    # around the spawn restores the real fds (equivalent to running with -s).
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory(prefix="pmv_exch_") as exch_dir:
        with capfd.disabled():
            mp.spawn(
                _prep_mem_view_worker,
                args=(exch_dir, backend_name),
                nprocs=2,
                join=True,
            )


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


def _run_xfer_telemetry_check(agent1, agent2, expect_telemetry: bool = True) -> None:
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

        handle = agent1.initialize_xfer("WRITE", src, dst, agent2.name, b"telem_msg")
        st = agent1.transfer(handle)
        assert st in ("DONE", "PROC")

        while True:
            st = agent1.check_xfer_state(handle)
            assert st in ("DONE", "PROC")
            if st == "DONE":
                break

        while not agent2.check_remote_xfer_done(agent1.name, b"telem_msg"):
            pass

        if not expect_telemetry:
            with pytest.raises(bindings.nixlNoTelemetryError):
                agent1.get_xfer_telemetry(handle)
            agent1.release_xfer_handle(handle)
            return

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


def test_get_xfer_telemetry_without_sink(backend_name):
    # Telemetry enabled with no sink still collects in-process via the NOP
    # fallback, so get_xfer_telemetry() works. Clear any inherited sink vars so
    # the sinkless path is exercised.
    prev_enable = os.environ.get("NIXL_TELEMETRY_ENABLE")
    os.environ["NIXL_TELEMETRY_ENABLE"] = "y"
    prev_dir = os.environ.pop("NIXL_TELEMETRY_DIR", None)
    prev_exporter = os.environ.pop("NIXL_TELEMETRY_EXPORTER", None)
    try:
        agent1 = nixl_agent(
            str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
        )
        agent2 = nixl_agent(
            str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
        )
        _run_xfer_telemetry_check(agent1, agent2, expect_telemetry=True)
    finally:
        os.environ.pop("NIXL_TELEMETRY_ENABLE", None)
        if prev_enable is not None:
            os.environ["NIXL_TELEMETRY_ENABLE"] = prev_enable
        if prev_dir is not None:
            os.environ["NIXL_TELEMETRY_DIR"] = prev_dir
        if prev_exporter is not None:
            os.environ["NIXL_TELEMETRY_EXPORTER"] = prev_exporter


def test_get_xfer_telemetry_with_buffer(backend_name):
    os.environ["NIXL_TELEMETRY_ENABLE"] = "y"
    with tempfile.TemporaryDirectory() as telemetry_dir:
        os.environ["NIXL_TELEMETRY_DIR"] = telemetry_dir
        try:
            agent1 = nixl_agent(
                str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
            )
            agent2 = nixl_agent(
                str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
            )
            _run_xfer_telemetry_check(agent1, agent2)
        finally:
            os.environ.pop("NIXL_TELEMETRY_ENABLE")
            os.environ.pop("NIXL_TELEMETRY_DIR")


def test_get_xfer_telemetry_cfg(backend_name):
    os.environ["NIXL_TELEMETRY_ENABLE"] = "no"
    os.environ["NIXL_TELEMETRY_DIR"] = "/tmp/dummy"  # to be ignored
    try:
        agent1 = nixl_agent(
            str(uuid.uuid4()),
            nixl_conf=nixl_agent_config(
                capture_telemetry=True, backends=[backend_name]
            ),
        )
        agent2 = nixl_agent(
            str(uuid.uuid4()), nixl_conf=nixl_agent_config(backends=[backend_name])
        )
        _run_xfer_telemetry_check(agent1, agent2, expect_telemetry=False)
    finally:
        os.environ.pop("NIXL_TELEMETRY_ENABLE")
        os.environ.pop("NIXL_TELEMETRY_DIR")
