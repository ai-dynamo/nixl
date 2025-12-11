#!/usr/bin/env python3

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
import time
from multiprocessing import Process

import tcp_server
from nixl_metadata_utils import (
    publish_agent_metadata,
    publish_descriptors,
    retrieve_agent_metadata,
    retrieve_descriptors,
)

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

# Configure logging
logger = get_logger(__name__)


def target_process():
    """Target process - receives data"""
    buf_size = 256
    logger.info("[target] Starting target process")

    # Create agent
    agent_config = nixl_agent_config(backends=["UCX"])
    target_agent = nixl_agent("target", agent_config)

    # Allocate and register memory
    target_addr = nixl_utils.malloc_passthru(buf_size * 2)
    target_addr2 = target_addr + buf_size

    target_addrs = [(target_addr, buf_size, 0), (target_addr2, buf_size, 0)]
    target_strings = [(target_addr, buf_size, 0, "a"), (target_addr2, buf_size, 0, "b")]

    target_reg_descs = target_agent.get_reg_descs(target_strings, "DRAM")
    target_xfer_descs = target_agent.get_xfer_descs(target_addrs, "DRAM")

    assert target_agent.register_memory(target_reg_descs) is not None
    logger.info("[target] Memory registered")

    # Publish metadata and descriptors
    publish_agent_metadata(target_agent, "target_meta")
    publish_descriptors(target_agent, target_xfer_descs, "target_descs")
    logger.info("[target] Published metadata and xfer descriptors to TCP server")

    # Wait for initiator to complete transfers
    logger.info("[target] Waiting for transfers...")

    # Check for transfer 1 completion
    while not target_agent.check_remote_xfer_done("initiator", b"UUID1"):
        time.sleep(0.001)
    logger.info("[target] Transfer 1 done")

    # Check for transfer 2 completion
    while not target_agent.check_remote_xfer_done("initiator", b"UUID2"):
        time.sleep(0.001)
    logger.info("[target] Transfer 2 done")

    # Cleanup
    target_agent.deregister_memory(target_reg_descs)
    nixl_utils.free_passthru(target_addr)
    logger.info("[target] Target process complete")


def initiator_process():
    """Initiator process - sends data"""
    buf_size = 256
    logger.info("[initiator] Starting initiator process")

    # Create agent
    initiator_agent = nixl_agent("initiator", None)
    initiator_addr = nixl_utils.malloc_passthru(buf_size * 2)
    initiator_addr2 = initiator_addr + buf_size

    initiator_addrs = [(initiator_addr, buf_size, 0), (initiator_addr2, buf_size, 0)]
    initiator_strings = [(initiator_addr, buf_size, 0, "a"), (initiator_addr2, buf_size, 0, "b")]

    initiator_reg_descs = initiator_agent.get_reg_descs(initiator_strings, "DRAM")
    initiator_xfer_descs = initiator_agent.get_xfer_descs(initiator_addrs, "DRAM")

    initiator_descs = initiator_agent.register_memory(initiator_reg_descs)
    assert initiator_descs is not None
    logger.info("[initiator] Memory registered")

    # Retrieve target's metadata and descriptors
    remote_name = retrieve_agent_metadata(initiator_agent, "target_meta", role_name="initiator")
    if not remote_name:
        return

    target_xfer_descs = retrieve_descriptors(initiator_agent, "target_descs")
    logger.info("[initiator] Successfully retrieved target descriptors")

    # Transfer 1: initialize transfer mode
    logger.info("[initiator] Starting transfer 1 (READ)...")
    xfer_handle_1 = initiator_agent.initialize_xfer(
        "READ", initiator_xfer_descs, target_xfer_descs, remote_name, b"UUID1"
    )
    if not xfer_handle_1:
        logger.error("[initiator] Creating transfer failed")
        return

    state = initiator_agent.transfer(xfer_handle_1)
    logger.info("[initiator] Initial transfer state: %s", state)
    if state == "ERR":
        logger.error("[initiator] Transfer failed immediately")
        return

    # Wait for transfer 1 to complete
    init_done = False
    while not init_done:
        state = initiator_agent.check_xfer_state(xfer_handle_1)
        if state == "ERR":
            logger.error("[initiator] Transfer got to Error state")
            return
        elif state == "DONE":
            init_done = True
            logger.info("[initiator] Transfer 1 done")
        time.sleep(0.001)

    # Transfer 2: prep transfer mode
    logger.info("[initiator] Starting transfer 2 (WRITE)...")
    local_prep_handle = initiator_agent.prep_xfer_dlist(
        "NIXL_INIT_AGENT", [(initiator_addr, buf_size, 0), (initiator_addr2, buf_size, 0)], "DRAM"
    )
    remote_prep_handle = initiator_agent.prep_xfer_dlist(
        remote_name, target_xfer_descs, "DRAM"
    )

    assert local_prep_handle != 0
    assert remote_prep_handle != 0

    xfer_handle_2 = initiator_agent.make_prepped_xfer(
        "WRITE", local_prep_handle, [0, 1], remote_prep_handle, [1, 0], b"UUID2"
    )
    if not xfer_handle_2:
        logger.error("[initiator] Make prepped transfer failed")
        return

    state = initiator_agent.transfer(xfer_handle_2)
    if state == "ERR":
        logger.error("[initiator] Transfer 2 failed immediately")
        return

    # Wait for transfer 2 to complete
    init_done = False
    while not init_done:
        state = initiator_agent.check_xfer_state(xfer_handle_2)
        if state == "ERR":
            logger.error("[initiator] Transfer 2 got to Error state")
            return
        elif state == "DONE":
            init_done = True
            logger.info("[initiator] Transfer 2 done")
        time.sleep(0.001)

    # Cleanup
    initiator_agent.release_xfer_handle(xfer_handle_1)
    initiator_agent.release_xfer_handle(xfer_handle_2)
    initiator_agent.release_dlist_handle(local_prep_handle)
    initiator_agent.release_dlist_handle(remote_prep_handle)
    initiator_agent.remove_remote_agent("target")
    initiator_agent.deregister_memory(initiator_reg_descs)
    nixl_utils.free_passthru(initiator_addr)

    logger.info("[initiator] Initiator process complete")


if __name__ == "__main__":
    logger.info("Using NIXL Plugins from:\n%s", os.environ["NIXL_PLUGIN_DIR"])

    # Start TCP metadata server
    logger.info("[main] Starting TCP metadata server...")
    try:
        tcp_server.start_server(9998)
        time.sleep(0.2)
    except OSError:
        pass  # Server may already be running
    tcp_server.clear_metadata("127.0.0.1", 9998)

    logger.info("[main] Starting target and initiator processes...")

    # Start both processes
    target_proc = Process(target=target_process)
    initiator_proc = Process(target=initiator_process)

    target_proc.start()
    initiator_proc.start()

    # Wait for both to complete
    target_proc.join()
    initiator_proc.join()

    if target_proc.exitcode == 0 and initiator_proc.exitcode == 0:
        logger.info("[main] ✓ Test Complete - Both processes succeeded!")
    else:
        logger.error(f"[main] ✗ Process error - Target: {target_proc.exitcode}, Initiator: {initiator_proc.exitcode}")
