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
# See the for the specific language governing permissions and
# limitations under the License.

"""
NIXL Peer-to-Peer Storage Example
Demonstrates peer-to-peer storage transfers using NIXL with initiator and target modes.
"""

import concurrent.futures
import time

import nixl_storage_utils as nsu

from nixl.logging import get_logger

logger = get_logger(__name__)


def execute_transfer(
    my_agent, local_descs, remote_descs, remote_name, operation, use_backends=[]
):
    handle = my_agent.initialize_xfer(
        operation, local_descs, remote_descs, remote_name, backends=use_backends
    )
    my_agent.transfer(handle)
    nsu.wait_for_transfer(my_agent, handle)
    my_agent.release_xfer_handle(handle)


def remote_storage_transfer(
    my_agent, my_mem_descs, operation, remote_agent_name, iterations
):
    """Initiate remote memory transfer."""
    if operation != "READ" and operation != "WRITE":
        logger.error("Invalid operation, exiting")
        exit(-1)

    if operation == "WRITE":
        operation = b"WRTE"
    else:
        operation = b"READ"

    iterations_str = bytes(f"{iterations:04d}", "utf-8")
    # Send the descriptors that you want to read into or write from
    logger.info(
        "Sending %s request to %s", operation.decode("utf-8"), remote_agent_name
    )
    test_descs_str = my_agent.get_serialized_descs(my_mem_descs)

    start_time = time.time()

    my_agent.send_notif(remote_agent_name, operation + iterations_str + test_descs_str)

    while not my_agent.check_remote_xfer_done(remote_agent_name, b"COMPLETE"):
        continue

    elapsed = time.time() - start_time

    logger.info("Time for %d iterations: %f seconds", iterations, elapsed)


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
                    logger.info("Waiting for remote metadata for %s...", parts[0])
                    time.sleep(1.0)

                logger.info("Remote metadata for %s fetched", parts[0])
            else:
                logger.error("Invalid line in %s: %s", agents_file, line)
                exit(-1)

    logger.info("All remote metadata fetched")

    return target_agents


def pipeline_reads(
    my_agent, req_agent, my_mem_descs, my_file_descs, sent_descs, iterations
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        n = 0
        s = 0
        futures = []

        while n < iterations or s < iterations:
            if s == 0:
                futures.append(
                    executor.submit(
                        execute_transfer,
                        my_agent,
                        my_mem_descs,
                        my_file_descs,
                        my_agent.name,
                        "READ",
                    )
                )
                s += 1
                continue

            if s == iterations:
                futures.append(
                    executor.submit(
                        execute_transfer,
                        my_agent,
                        my_mem_descs,
                        sent_descs,
                        req_agent,
                        "WRITE",
                    )
                )
                n += 1
                continue

            # Do two storage and network in parallel
            futures.append(
                executor.submit(
                    execute_transfer,
                    my_agent,
                    my_mem_descs,
                    my_file_descs,
                    my_agent.name,
                    "READ",
                )
            )
            futures.append(
                executor.submit(
                    execute_transfer,
                    my_agent,
                    my_mem_descs,
                    sent_descs,
                    req_agent,
                    "WRITE",
                )
            )
            s += 1
            n += 1

        _, not_done = concurrent.futures.wait(
            futures, return_when=concurrent.futures.ALL_COMPLETED
        )
        assert not not_done


def pipeline_writes(
    my_agent, req_agent, my_mem_descs, my_file_descs, sent_descs, iterations
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        n = 0
        s = 1
        futures = []

        futures.append(
            executor.submit(
                execute_transfer,
                my_agent,
                my_mem_descs,
                sent_descs,
                req_agent,
                "READ",
            )
        )
        while n < iterations or s < iterations:
            if s == iterations:
                futures.append(
                    executor.submit(
                        execute_transfer,
                        my_agent,
                        my_mem_descs,
                        my_file_descs,
                        my_agent.name,
                        "WRITE",
                    )
                )
                n += 1
                continue

            # Do two storage and network in parallel
            futures.append(
                executor.submit(
                    execute_transfer,
                    my_agent,
                    my_mem_descs,
                    sent_descs,
                    req_agent,
                    "READ",
                )
            )
            futures.append(
                executor.submit(
                    execute_transfer,
                    my_agent,
                    my_mem_descs,
                    my_file_descs,
                    my_agent.name,
                    "WRITE",
                )
            )
            s += 1
            n += 1

        _, not_done = concurrent.futures.wait(
            futures, return_when=concurrent.futures.ALL_COMPLETED
        )
        assert not not_done


def handle_remote_transfer_request(my_agent, my_mem_descs, my_file_descs):
    """Handle remote memory and storage transfers as target."""
    # Wait for initiator to send list of memory descriptors
    notifs = my_agent.get_new_notifs()

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

        iterations = int(recv_msg[4:8])

        logger.info("Performing %s with %d iterations", operation, iterations)

        sent_descs = my_agent.deserialize_descs(recv_msg[8:])

        if operation == "READ":
            pipeline_reads(
                my_agent, req_agent, my_mem_descs, my_file_descs, sent_descs, iterations
            )
        elif operation == "WRITE":
            pipeline_writes(
                my_agent, req_agent, my_mem_descs, my_file_descs, sent_descs, iterations
            )

        # Send completion notification to initiator
        my_agent.send_notif(req_agent, b"COMPLETE")


def run_client(
    my_agent, nixl_mem_reg_descs, nixl_file_reg_descs, agents_file, iterations
):
    logger.info("Client initialized, ready for local transfer test...")

    # For sample purposes, write to and then read from local storage
    logger.info("Starting local transfer test...")

    start_time = time.time()

    for i in range(1, iterations):
        execute_transfer(
            my_agent,
            nixl_mem_reg_descs.trim(),
            nixl_file_reg_descs.trim(),
            my_agent.name,
            "WRITE",
            ["GDS_MT"],
        )

    elapsed = time.time() - start_time

    logger.info("Time for %d WRITE iterations: %f seconds", iterations, elapsed)

    start_time = time.time()

    for i in range(1, iterations):
        execute_transfer(
            my_agent,
            nixl_mem_reg_descs.trim(),
            nixl_file_reg_descs.trim(),
            my_agent.name,
            "READ",
            ["GDS_MT"],
        )

    elapsed = time.time() - start_time

    logger.info("Time for %d READ iterations: %f seconds", iterations, elapsed)

    logger.info("Local transfer test complete")

    logger.info("Starting remote transfer test...")

    target_agents = connect_to_agents(my_agent, agents_file)

    # For sample purposes, write to and then read from each target agent
    for target_agent in target_agents:
        remote_storage_transfer(
            my_agent, nixl_mem_reg_descs.trim(), "WRITE", target_agent, iterations
        )
        remote_storage_transfer(
            my_agent, nixl_mem_reg_descs.trim(), "READ", target_agent, iterations
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for each transfer",
    )
    args = parser.parse_args()

    mem = "DRAM"

    if args.role == "client":
        mem = "VRAM"

    my_agent = nsu.create_agent_with_plugins(args.name, args.port)

    (
        my_mem_list,
        my_file_list,
        nixl_mem_reg_descs,
        nixl_file_reg_descs,
    ) = nsu.setup_memory_and_files(
        my_agent, args.batch_size, args.buf_size, args.fileprefix, mem
    )

    if args.role == "client":
        if not args.agents_file:
            parser.error("--agents_file is required when role is client")
        try:
            run_client(
                my_agent,
                nixl_mem_reg_descs,
                nixl_file_reg_descs,
                args.agents_file,
                args.iterations,
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
