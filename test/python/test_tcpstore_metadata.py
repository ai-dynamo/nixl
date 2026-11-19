# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from datetime import timedelta

import pytest
import torch.distributed as dist

import nixl._utils as utils
from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t


@pytest.mark.timeout(60)
def test_tcpstore_metadata_exchange(monkeypatch):
    """Publish and fetch agent metadata through a real PyTorch TCPStore."""
    # Let the kernel pick a free port: a port chosen ahead of time can be taken
    # by the time the store binds it.
    tcp_store = dist.TCPStore(
        host_name="127.0.0.1",
        port=0,
        world_size=None,
        is_master=True,
        timeout=timedelta(seconds=30),
        wait_for_workers=False,
    )

    # CI normally exports NIXL_ETCD_ENDPOINTS for the rest of the Python suite.
    # The metadata manager permits exactly one name-addressed backend per agent.
    monkeypatch.delenv("NIXL_ETCD_ENDPOINTS", raising=False)
    monkeypatch.setenv("NIXL_TCPSTORE_ENDPOINT", f"127.0.0.1:{tcp_store.port}")

    config = nixl_agent_config(
        enable_prog_thread=False,
        enable_listen_thread=False,
        backends=["UCX"],
        sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
    )
    suffix = uuid.uuid4().hex
    source = nixl_agent(f"tcpstore_source_{suffix}", config)
    target = nixl_agent(f"tcpstore_target_{suffix}", config)

    source_addr = utils.malloc_passthru(1024)
    target_addr = utils.malloc_passthru(1024)
    source_reg = source.get_reg_descs(
        [(source_addr, 1024, 0, "tcpstore-source")], mem_type="DRAM"
    )
    target_reg = target.get_reg_descs(
        [(target_addr, 1024, 0, "tcpstore-target")], mem_type="DRAM"
    )

    try:
        assert source.register_memory(source_reg) is not None
        assert target.register_memory(target_reg) is not None

        source.send_local_metadata()
        target.fetch_remote_metadata(source.name)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if target.check_remote_metadata(source.name):
                break
            time.sleep(0.01)
        else:
            pytest.fail("TCPStore metadata was not fetched within 10 seconds")

        src_meta = source.get_agent_metadata()
        store_meta = tcp_store.get(f"nixl/agents/metadata/{source.name}/NULL_AGENT")
        if not src_meta == store_meta:
            pytest.fail("TCPStore metadata does not match source metadata")

        # TCPStore does not notify peers when a key is deleted, so clean up both
        # the published key and the target's cached metadata explicitly.
        source.invalidate_local_metadata()
        target.remove_remote_agent(source.name)

        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if not tcp_store.check([f"nixl/agents/metadata/{source.name}/NULL_AGENT"]):
                break
            time.sleep(0.01)
        else:
            pytest.fail("TCPStore metadata was not removed within 1 second")
    finally:
        source.deregister_memory(source_reg)
        target.deregister_memory(target_reg)
        # Stop the agents' TCPStore workers while the in-process store is alive.
        del target
        del source
        utils.free_passthru(source_addr)
        utils.free_passthru(target_addr)

def test_tcpstore_metadata_reconnection(monkeypatch, capfd):
    """Test TCPStore metadata reconnection."""
    # Let the kernel pick a free port: a port chosen ahead of time can be taken
    # by the time the store binds it.
    tcp_store = dist.TCPStore(
        host_name="127.0.0.1",
        port=0,
        world_size=None,
        is_master=True,
        timeout=timedelta(seconds=30),
        wait_for_workers=False,
    )

    # CI normally exports NIXL_ETCD_ENDPOINTS for the rest of the Python suite.
    # The metadata manager permits exactly one name-addressed backend per agent.
    tcp_store_port = tcp_store.port
    monkeypatch.delenv("NIXL_ETCD_ENDPOINTS", raising=False)
    monkeypatch.setenv("NIXL_TCPSTORE_ENDPOINT", f"127.0.0.1:{tcp_store_port}")
    monkeypatch.setenv("NIXL_TCPSTORE_TIMEOUT_MS", "15000")
    monkeypatch.setenv("NIXL_TCPSTORE_OP_TIMEOUT_MS", "1000")

    config = nixl_agent_config(
        enable_prog_thread=False,
        enable_listen_thread=False,
        backends=["UCX"],
        sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
    )
    suffix = uuid.uuid4().hex
    source = nixl_agent(f"tcpstore_source_{suffix}", config)
    target = nixl_agent(f"tcpstore_target_{suffix}", config)

    source_addr = utils.malloc_passthru(1024)
    target_addr = utils.malloc_passthru(1024)
    source_reg = source.get_reg_descs(
        [(source_addr, 1024, 0, "tcpstore-source")], mem_type="DRAM"
    )
    target_reg = target.get_reg_descs(
        [(target_addr, 1024, 0, "tcpstore-target")], mem_type="DRAM"
    )

    try:
        assert source.register_memory(source_reg) is not None
        assert target.register_memory(target_reg) is not None

        source.send_local_metadata()
        target.fetch_remote_metadata(source.name)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if target.check_remote_metadata(source.name):
                break
            time.sleep(0.1)
        else:
            pytest.fail("TCPStore metadata was not fetched within 10 seconds")

        # Shutdown the TCPStore to simulate a network problem.
        del tcp_store

        # This should trigger I/O error and connection retry.
        source.invalidate_local_metadata()
        target.remove_remote_agent(source.name)
        target.fetch_remote_metadata(source.name)
        # After 10 seconds, connection retry should stop. Then the
        # fetch retry would kick in and trigger another connection retry.
        # After another 10 seconds, connection retry should stop again,
        # and the fetch retry should give up.
        time.sleep(25)

        captured = capfd.readouterr()
        assert captured.err.count("connect failed: Connection refused.") == 30
        assert captured.err.count("Max connection retry count reached.") == 3
        assert captured.err.count("gave up waiting for key") == 1

        # Restart the TCPStore.
        tcp_store = dist.TCPStore(
            host_name="127.0.0.1",
            port=tcp_store_port,
            world_size=None,
            is_master=True,
        )

        # This should trigger a new connection.
        source.send_local_metadata()
        target.fetch_remote_metadata(source.name)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if target.check_remote_metadata(source.name):
                break
            time.sleep(0.1)
        else:
            pytest.fail("TCPStore metadata was not fetched within 10 seconds")
    finally:
        source.deregister_memory(source_reg)
        target.deregister_memory(target_reg)
        # Stop the agents' TCPStore workers while the in-process store is alive.
        del target
        del source
        utils.free_passthru(source_addr)
        utils.free_passthru(target_addr)
