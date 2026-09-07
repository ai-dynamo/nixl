# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import uuid
from datetime import timedelta

import pytest
import torch.distributed as dist

import nixl._utils as utils
from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t


@pytest.mark.timeout(20)
def test_tcpstore_metadata_exchange(monkeypatch):
    """Publish and fetch agent metadata through a real PyTorch TCPStore."""
    # CI assigns the port from the range this executor owns, see
    # .gitlab/test_python.sh; a local run lets the kernel pick one.
    tcp_store = dist.TCPStore(
        host_name="127.0.0.1",
        port=int(os.environ.get("NIXL_TCPSTORE_PORT", "0")),
        world_size=None,
        is_master=True,
        timeout=timedelta(seconds=5),
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

        # TCPStore does not notify peers when a key is deleted, so clean up both
        # the published key and the target's cached metadata explicitly.
        source.invalidate_local_metadata()
        target.remove_remote_agent(source.name)
    finally:
        source.deregister_memory(source_reg)
        target.deregister_memory(target_reg)
        # Stop the agents' TCPStore workers while the in-process store is alive.
        del target
        del source
        utils.free_passthru(source_addr)
        utils.free_passthru(target_addr)
