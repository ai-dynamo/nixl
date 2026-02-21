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

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)

# Configuration - Change these values to match your etcd setup
ETCD_ENDPOINT = "http://127.0.0.1:2379"
AGENT1_NAME = "EtcdAgent1"
AGENT2_NAME = "EtcdAgent2"


def register_memory(agent, backend_name, pattern):
    buffer_size = 1024
    addr = nixl_utils.malloc_passthru(buffer_size)

    # Fill buffer with pattern
    data = pattern * buffer_size
    reg_descs = agent.get_reg_descs([(addr, buffer_size, 0, data)], "DRAM")

    # Register memory
    agent.register_memory(reg_descs, backends=[backend_name])

    logger.info(f"Registered memory {hex(addr)} with agent {agent.name}")
    return addr, reg_descs


def main():
    # Set etcd endpoint if not already set
    if os.getenv("NIXL_ETCD_ENDPOINTS"):
        logger.info("NIXL_ETCD_ENDPOINTS is set")
    else:
        logger.info(f"NIXL_ETCD_ENDPOINTS is not set, setting to {ETCD_ENDPOINT}")
        os.environ["NIXL_ETCD_ENDPOINTS"] = ETCD_ENDPOINT

    logger.info("NIXL Etcd Metadata Example")

    # ===== 1. Create two agents (normally these would be in separate processes or machines) =====
    agent1_config = nixl_agent_config(backends=["UCX"])
    agent1 = nixl_agent(AGENT1_NAME, agent1_config)

    agent2_config = nixl_agent_config(backends=["UCX"])
    agent2 = nixl_agent(AGENT2_NAME, agent2_config)

    logger.info(f"Available plugins: {agent1.plugin_list}")

    # Get plugin parameters
    logger.info(
        "Plugin parameters:\n%s\n%s",
        agent1.get_plugin_mem_types("UCX"),
        agent1.get_plugin_params("UCX"),
    )

    logger.info(
        "Backend parameters:\n%s\n%s",
        agent1.get_backend_mem_types("UCX"),
        agent1.get_backend_params("UCX"),
    )

    # ===== 2. Register memory with both agents =====
    addr1, reg_descs1 = register_memory(agent1, "UCX", "a")
    addr2, reg_descs2 = register_memory(agent2, "UCX", "b")

    # ===== 3. Send Local Metadata to etcd =====
    logger.info("Sending local metadata to etcd...")

    # Both agents send their metadata to etcd
    agent1.send_local_metadata()
    agent2.send_local_metadata()

    # Give etcd time to process
    time.sleep(1)

    # ===== 4. Fetch Remote Metadata from etcd =====
    logger.info("Fetching remote metadata from etcd...")

    # Agent1 fetches metadata for Agent2
    agent1.fetch_remote_metadata(AGENT2_NAME)

    # Agent2 fetches metadata for Agent1
    agent2.fetch_remote_metadata(AGENT1_NAME)

    # Wait for metadata to be available (fetch_remote_metadata is asynchronous)
    while not (
        agent1.check_remote_metadata(AGENT2_NAME)
        and agent2.check_remote_metadata(AGENT1_NAME)
    ):
        time.sleep(0.5)

    logger.info("Metadata exchange successful!")

    # ===== 5. Do transfer from Agent 1 to Agent 2 =====
    req_size = 8
    dst_offset = 8

    logger.info(f"Agent1's address: {hex(addr1)}")
    logger.info(f"Agent2's address: {hex(addr2)}")

    # Create transfer descriptors
    req_src_descs = agent1.get_xfer_descs(
        [(addr1 + 16, req_size, 0)], "DRAM"
    )  # random offset
    req_dst_descs = agent2.get_xfer_descs(
        [(addr2 + dst_offset, req_size, 0)], "DRAM"
    )  # random offset

    logger.info(f"Transfer request from {hex(addr1 + 16)} to {hex(addr2 + dst_offset)}")

    # Create and post transfer request with notification
    xfer_handle = agent1.initialize_xfer(
        "WRITE", req_src_descs, req_dst_descs, AGENT2_NAME, b"notification"
    )
    logger.info("Transfer request created")
    state = agent1.transfer(xfer_handle)
    logger.info(f"Transfer was posted, initial state: {state}")

    # Wait for transfer completion and notification
    notifs = {}
    while state != "DONE" or len(notifs) == 0:
        if state != "DONE":
            state = agent1.check_xfer_state(xfer_handle)
        if len(notifs) == 0:
            notifs = agent2.get_new_notifs()
        time.sleep(0.5)

    logger.info(f"Received notifications: {notifs}")
    logger.info("Transfer verified")

    # Release transfer handle
    agent1.release_xfer_handle(xfer_handle)

    # Deregister memory
    agent1.deregister_memory(reg_descs1, backends=["UCX"])
    agent2.deregister_memory(reg_descs2, backends=["UCX"])


if __name__ == "__main__":
    main()
