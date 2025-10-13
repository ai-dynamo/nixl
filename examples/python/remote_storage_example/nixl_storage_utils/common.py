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

"""
Common utilities for NIXL storage operations.
Provides core functionality for memory management and storage operations.
"""

import argparse
import os

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
from nixl._bindings import DRAM_SEG
from nixl.logging import get_logger

logger = get_logger(__name__)


def create_agent_with_plugins(agent_name, port):
    """Create a NIXL agent with required plugins."""
    agent_config = nixl_agent_config(True, True, port, backends=[])
    new_nixl_agent = nixl_agent(agent_name, agent_config)

    plugin_list = new_nixl_agent.get_plugin_list()

    if "GDS" in plugin_list:
        new_nixl_agent.create_backend("GDS")
        logger.info("Using GDS storage backend")
    if "POSIX" in plugin_list:
        new_nixl_agent.create_backend("POSIX")
        logger.info("Using POSIX storage backend")

    if "GDS" not in plugin_list and "POSIX" not in plugin_list:
        logger.error("No storage backends available, exiting")
        exit(-1)

    if "UCX" not in plugin_list:
        logger.error("UCX not available for transfer, exiting")
        exit(-1)
    else:
        new_nixl_agent.create_backend("UCX")

    logger.info("Initialized backends")
    return new_nixl_agent


def setup_memory_and_files(agent, batch_size, buf_size, fileprefix):
    """Setup memory and file resources."""
    my_mem_list = []
    my_file_list = []
    nixl_mem_reg_list = []
    nixl_file_reg_list = []

    for i in range(batch_size):
        my_mem_list.append(nixl_utils.malloc_passthru(buf_size))
        my_file_list.append(os.open(f"{fileprefix}_{i}", os.O_RDWR | os.O_CREAT))
        nixl_mem_reg_list.append((my_mem_list[-1], buf_size, 0, str(i)))
        nixl_file_reg_list.append((0, buf_size, my_file_list[-1], str(i)))

    nixl_mem_reg_descs = agent.register_memory(nixl_mem_reg_list, "DRAM")
    nixl_file_reg_descs = agent.register_memory(nixl_file_reg_list, "FILE")

    assert nixl_mem_reg_descs is not None
    assert nixl_file_reg_descs is not None

    return my_mem_list, my_file_list, nixl_mem_reg_descs, nixl_file_reg_descs


def cleanup_resources(agent, mem_reg_descs, file_reg_descs, mem_list, file_list):
    """Cleanup memory and file resources."""
    agent.deregister_memory(mem_reg_descs)

    if mem_reg_descs.getType() == DRAM_SEG:
        agent.deregister_memory(file_reg_descs, backends=["POSIX"])

        for mem in mem_list:
            nixl_utils.free_passthru(mem)
    else:
        agent.deregister_memory(file_reg_descs, backends=["GDS"])
        # TODO: cudaFree

    for file in file_list:
        os.close(file)


def get_base_parser():
    """Get base argument parser with common arguments."""
    parser = argparse.ArgumentParser(description="NIXL Storage Sample")
    parser.add_argument("--fileprefix", type=str, help="Path to the files for testing")
    parser.add_argument(
        "--buf_size",
        type=int,
        default=4096,
        help="Buffer size in bytes (default: 4096)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (default: 1)"
    )
    return parser


def wait_for_transfer(agent, handle):
    """Wait for transfer to complete."""
    status = agent.check_xfer_state(handle)
    while status != "DONE":
        if status == "ERR":
            logger.error("Transfer got to Error state.")
            exit()
        status = agent.check_xfer_state(handle)
