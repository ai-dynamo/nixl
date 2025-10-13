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
# See the for the specific language governing permissions and
# limitations under the License.

"""
NIXL Peer-to-Peer Storage Example
Demonstrates peer-to-peer storage transfers using NIXL with initiator and target modes.
"""

import time

import nixl_storage_utils as nsu

from nixl.logging import get_logger

logger = get_logger(__name__)


def execute_transfer(my_agent, local_descs, remote_descs, remote_name, operation):
    handle = my_agent.initialize_xfer(operation, local_descs, remote_descs, remote_name)
    my_agent.transfer(handle)
    nsu.wait_for_transfer(my_agent, handle)
    my_agent.release_xfer_handle(handle)


def remote_storage_transfer(my_agent, my_mem_descs, operation, remote_agent_name):
    """Initiate remote memory transfer."""
    if operation != "READ" and operation != "WRITE":
        logger.error("Invalid operation, exiting")
        exit(-1)

    if operation == "WRITE":
        operation = b"WRTE"
    else:
        operation = b"READ"

    # Send the descriptors that you want to read into or write from
    logger.info(f"Sending {operation} request to {remote_agent_name}")
    test_descs_str = my_agent.get_serialized_descs(my_mem_descs)
    my_agent.send_notif(remote_agent_name, operation + test_descs_str)

    while not my_agent.check_remote_xfer_done(remote_agent_name, b"COMPLETE"):
        continue


def connect_to_agents(my_agent, agents_file):
    target_agents = []
    with open(agents_file, "r") as f:
        for line in f:
            # Each line in file should be: "<agent_name> <ip> <port>"
            parts = line.strip().split()
            if len(parts) == 3:
                target_agents.append(parts[0])
                my_agent.send_local_metadata(parts[1], int(parts[2]))
                my_agent.fetch_remote_metadata(parts[0], parts[1], int(parts[2]))

                while my_agent.check_remote_metadata(parts[0]) is False:
                    logger.info(f"Waiting for remote metadata for {parts[0]}...")
                    time.sleep(0.2)

                logger.info(f"Remote metadata for {parts[0]} fetched")
            else:
                logger.error(f"Invalid line in {agents_file}: {line}")
                exit(-1)

    logger.info("All remote metadata fetched")

    return target_agents


def handle_remote_transfer_request(my_agent, my_mem_descs, my_file_descs):
    """Handle remote memory and storage transfers as target."""
    # Wait for initiator to send list of memory descriptors
    notifs = my_agent.get_new_notifs()

    logger.info("Waiting for a remote transfer request...")

    while len(notifs) == 0:
        notifs = my_agent.get_new_notifs()

    for req_agent in notifs:
        recv_msg = notifs[req_agent][0]

        operation = None
        if recv_msg[:4] == b"READ":
            operation = "READ"
        elif recv_msg[:4] == b"WRTE":
            operation = "WRITE"
        else:
            logger.error("Invalid operation, exiting")
            exit(-1)

        sent_descs = my_agent.deserialize_descs(recv_msg[4:])

        logger.info("Checking to ensure metadata is loaded...")
        while my_agent.check_remote_metadata(req_agent, sent_descs) is False:
            continue

        if operation == "READ":
            logger.info("Starting READ operation")

            # Read from file first
            execute_transfer(
                my_agent, my_mem_descs, my_file_descs, my_agent.name, "READ"
            )
            # Send to client
            execute_transfer(my_agent, my_mem_descs, sent_descs, req_agent, "WRITE")

        elif operation == "WRITE":
            logger.info("Starting WRITE operation")

            # Read from client first
            execute_transfer(my_agent, my_mem_descs, sent_descs, req_agent, "READ")
            # Write to storage
            execute_transfer(
                my_agent, my_mem_descs, my_file_descs, my_agent.name, "WRITE"
            )

        # Send completion notification to initiator
        my_agent.send_notif(req_agent, b"COMPLETE")

    logger.info("One transfer test complete.")


def run_client(my_agent, nixl_mem_reg_descs, nixl_file_reg_descs, agents_file):
    logger.info("Client initialized, ready for local transfer test...")

    # For sample purposes, write to and then read from local storage
    logger.info("Starting local transfer test...")
    execute_transfer(
        my_agent,
        nixl_mem_reg_descs.trim(),
        nixl_file_reg_descs.trim(),
        my_agent.name,
        "WRITE",
    )
    execute_transfer(
        my_agent,
        nixl_mem_reg_descs.trim(),
        nixl_file_reg_descs.trim(),
        my_agent.name,
        "READ",
    )
    logger.info("Local transfer test complete")

    logger.info("Starting remote transfer test...")

    target_agents = connect_to_agents(my_agent, agents_file)

    # For sample purposes, write to and then read from each target agent
    for target_agent in target_agents:
        remote_storage_transfer(
            my_agent, nixl_mem_reg_descs.trim(), "WRITE", target_agent
        )
        remote_storage_transfer(
            my_agent, nixl_mem_reg_descs.trim(), "READ", target_agent
        )

    logger.info("Remote transfer test complete")


def run_storage_server(my_agent, nixl_mem_reg_descs, nixl_file_reg_descs):
    logger.info("Server initialized, ready for remote transfer test...")
    while True:
        handle_remote_transfer_request(
            my_agent, nixl_mem_reg_descs.trim(), nixl_file_reg_descs.trim()
        )


if __name__ == "__main__":
    parser = nsu.get_base_parser()
    parser.add_argument(
        "--role",
        type=str,
        choices=["server", "client"],
        required=True,
        help="Role of this node (server or client)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port to listen on for remote transfers (only needed for server)",
    )
    parser.add_argument("--name", type=str, help="NIXL agent name")
    parser.add_argument(
        "--agents_file",
        type=str,
        help="File containing list of target agents (only needed for client)",
    )
    args = parser.parse_args()

    my_agent = nsu.create_agent_with_plugins(args.name, args.port)

    (
        my_mem_list,
        my_file_list,
        nixl_mem_reg_descs,
        nixl_file_reg_descs,
    ) = nsu.setup_memory_and_files(
        my_agent, args.batch_size, args.buf_size, args.fileprefix
    )

    if args.role == "client":
        if not args.agents_file:
            parser.error("--agents_file is required when role is client")
        try:
            run_client(
                my_agent, nixl_mem_reg_descs, nixl_file_reg_descs, args.agents_file
            )
        finally:
            nsu.cleanup_resources(
                my_agent,
                nixl_mem_reg_descs,
                nixl_file_reg_descs,
                my_mem_list,
                my_file_list,
            )
    else:
        if args.agents_file:
            logger.warning("Warning: --agents_file is ignored when role is server")
        try:
            run_storage_server(my_agent, nixl_mem_reg_descs, nixl_file_reg_descs)
        finally:
            nsu.cleanup_resources(
                my_agent,
                nixl_mem_reg_descs,
                nixl_file_reg_descs,
                my_mem_list,
                my_file_list,
            )

    logger.info("Test Complete.")
