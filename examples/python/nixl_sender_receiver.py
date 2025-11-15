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

"""
NIXL Sender-Receiver Example: Queue-Based Flow Control

Demonstrates a producer-consumer pattern using head/tail pointers with RDMA WRITE operations.
The sender fills a circular queue of buffers, and the receiver consumes them, with bandwidth reporting.
"""

import os
import sys
import time
from multiprocessing import Process

import numpy as np
import tcp_server
from nixl_memory_utils import read_data, read_uint64, write_data, write_uint64
from nixl_metadata_utils import (
    publish_agent_metadata,
    publish_descriptors,
    retrieve_agent_metadata,
    retrieve_descriptors,
)

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)

NUM_BUFFERS = 2  # Queue size (optimal)
BUFFER_SIZE = 16 * 1024 * 1024  # 16MB (optimal)
NUM_TRANSFERS = 100


def receiver_process():
    """Receiver with queue-based flow control"""
    logger.info("[receiver] Starting")

    # Create NIXL agent (single worker is optimal for point-to-point)
    config = nixl_agent_config(backends=["UCX"])
    agent = nixl_agent("receiver", config)

    # Allocate Buffers
    # Tail + data buffers (remote writes here)
    tail_and_buffers_size = 8 + (NUM_BUFFERS * BUFFER_SIZE)
    tail_and_buffers_addr = nixl_utils.malloc_passthru(tail_and_buffers_size)

    # Head (local, will write to remote)
    head_addr = nixl_utils.malloc_passthru(8)

    # Register and create descriptors
    tail_reg_desc = [(tail_and_buffers_addr, tail_and_buffers_size, 0, "tail_buffers")]
    head_reg_desc = [(head_addr, 8, 0, "head")]

    tail_reg_descs = agent.get_reg_descs(tail_reg_desc, "DRAM")
    head_reg_descs = agent.get_reg_descs(head_reg_desc, "DRAM")

    agent.register_memory(tail_reg_descs)
    agent.register_memory(head_reg_descs)

    # Create xfer_descs for transfers - separate tail pointer from data buffers
    tail_pointer_xfer_desc = [(tail_and_buffers_addr, 8, 0)]  # Just the tail pointer
    head_xfer_desc = [(head_addr, 8, 0)]

    # Create individual buffer descriptors (one for each buffer slot)
    data_start_addr = tail_and_buffers_addr + 8
    buffers_xfer_desc = [(data_start_addr + i * BUFFER_SIZE, BUFFER_SIZE, 0) for i in range(NUM_BUFFERS)]

    tail_pointer_xfer_descs = agent.get_xfer_descs(tail_pointer_xfer_desc, "DRAM")
    buffers_xfer_descs = agent.get_xfer_descs(buffers_xfer_desc, "DRAM")
    head_xfer_descs = agent.get_xfer_descs(head_xfer_desc, "DRAM")

    logger.info(f"[receiver] Allocated: tail_buffers at 0x{tail_and_buffers_addr:x}, head at 0x{head_addr:x}")

    # Exchange metadata and descriptors
    publish_agent_metadata(agent, "receiver_meta")
    publish_descriptors(agent, tail_pointer_xfer_descs, "receiver_tail_desc")
    publish_descriptors(agent, buffers_xfer_descs, "receiver_buffers_desc")
    publish_descriptors(agent, head_xfer_descs, "receiver_head_desc")

    # Retrieve sender's metadata and descriptors
    remote_name = retrieve_agent_metadata(agent, "sender_meta", role_name="receiver")
    if not remote_name:
        return

    # Note: sender_tail_desc not needed by receiver
    sender_head_descs = retrieve_descriptors(agent, "sender_head_desc")

    logger.info(f"[receiver] Connected to {remote_name}")

    # Create xfer_handler for writing Head to sender
    # Use deserialized reg_descs directly as the example does
    local_prep = agent.prep_xfer_dlist("NIXL_INIT_AGENT", [(head_addr, 8, 0)], "DRAM")
    remote_prep = agent.prep_xfer_dlist(remote_name, sender_head_descs, "DRAM")
    head_xfer_handle = agent.make_prepped_xfer("WRITE", local_prep, [0], remote_prep, [0], b"HEAD_UPDATE")

    if not head_xfer_handle:
        logger.error("[receiver] Failed to create head xfer handle")
        return

    # Init local Head and Tail to 0 (empty)
    local_head = 0
    local_tail_addr = tail_and_buffers_addr
    write_uint64(head_addr, 0)
    write_uint64(local_tail_addr, 0)

    # Transfer initial Head value
    state = agent.transfer(head_xfer_handle)
    if state == "ERR":
        logger.error("[receiver] Failed to transfer initial head")
        return

    # Wait for transfer to complete
    while agent.check_xfer_state(head_xfer_handle) == "PENDING":
        time.sleep(0.001)

    logger.info("[receiver] Initialized, starting main loop")

    # Main loop
    transfers_received = 0
    data_start_addr = tail_and_buffers_addr + 8

    # Performance tracking
    start_time = time.time()
    first_transfer_time = None

    while transfers_received < NUM_TRANSFERS:
        # Read local tail (sender writes here)
        remote_tail = read_uint64(local_tail_addr)

        # Check if not empty: tail != head
        if remote_tail != local_head:
            # Process buffer using NumPy
            buffer_idx = local_head % NUM_BUFFERS
            buffer_offset = data_start_addr + (buffer_idx * BUFFER_SIZE)

            # Read header (8 bytes) as NumPy array
            header_data = read_data(buffer_offset, 8)
            received_id = int(header_data.view(np.uint64)[0])

            if received_id != transfers_received:
                logger.error(f"[receiver] Mismatch! Expected {transfers_received}, got {received_id}")

            # Track first transfer time
            if first_transfer_time is None:
                first_transfer_time = time.time()

            # Update head
            local_head = (local_head + 1) % NUM_BUFFERS
            write_uint64(head_addr, local_head)

            # Transfer head
            state = agent.transfer(head_xfer_handle)
            if state == "ERR":
                logger.error("[receiver] Transfer head failed")
                break

            transfers_received += 1

            if transfers_received % 10 == 0:
                logger.info(f"[receiver] Processed {transfers_received}/{NUM_TRANSFERS}")
        else:
            # Queue is empty, wait
            time.sleep(0.0001)

    end_time = time.time()

    # Calculate performance metrics
    total_time = end_time - start_time
    if first_transfer_time:
        actual_transfer_time = end_time - first_transfer_time
    else:
        actual_transfer_time = total_time

    total_bytes = transfers_received * BUFFER_SIZE
    bandwidth_mbps = (total_bytes / actual_transfer_time) / (1024 * 1024) if actual_transfer_time > 0 else 0

    logger.info(f"[receiver] Completed {transfers_received} transfers in {actual_transfer_time:.3f}s")
    logger.info(f"[receiver] Bandwidth: {bandwidth_mbps:.2f} MB/s")

    # Cleanup
    agent.release_xfer_handle(head_xfer_handle)
    agent.release_dlist_handle(local_prep)
    agent.release_dlist_handle(remote_prep)
    agent.deregister_memory(tail_reg_descs)
    agent.deregister_memory(head_reg_descs)
    nixl_utils.free_passthru(tail_and_buffers_addr)
    nixl_utils.free_passthru(head_addr)


def sender_process():
    """Sender with queue-based flow control"""
    logger.info("[sender] Starting")

    # Create NIXL agent (single worker is optimal for point-to-point)
    config = nixl_agent_config(backends=["UCX"])
    agent = nixl_agent("sender", config)

    # Allocate Buffers
    tail_addr = nixl_utils.malloc_passthru(8)
    buffers_size = NUM_BUFFERS * BUFFER_SIZE
    buffers_addr = nixl_utils.malloc_passthru(buffers_size)

    head_addr = nixl_utils.malloc_passthru(8)

    # Register and create descriptors
    tail_reg_desc = [(tail_addr, 8, 0, "tail")]
    buffers_reg_desc = [(buffers_addr, buffers_size, 0, "buffers")]
    head_reg_desc = [(head_addr, 8, 0, "head")]

    tail_reg_descs = agent.get_reg_descs(tail_reg_desc, "DRAM")
    buffers_reg_descs = agent.get_reg_descs(buffers_reg_desc, "DRAM")
    head_reg_descs = agent.get_reg_descs(head_reg_desc, "DRAM")

    agent.register_memory(tail_reg_descs)
    agent.register_memory(buffers_reg_descs)
    agent.register_memory(head_reg_descs)

    # Create xfer_descs for transfers
    tail_xfer_desc = [(tail_addr, 8, 0)]
    head_xfer_desc = [(head_addr, 8, 0)]

    tail_xfer_descs = agent.get_xfer_descs(tail_xfer_desc, "DRAM")
    head_xfer_descs = agent.get_xfer_descs(head_xfer_desc, "DRAM")

    logger.info(f"[sender] Allocated: tail at 0x{tail_addr:x}, buffers at 0x{buffers_addr:x}, head at 0x{head_addr:x}")

    # Exchange metadata and descriptors
    publish_agent_metadata(agent, "sender_meta")
    publish_descriptors(agent, tail_xfer_descs, "sender_tail_desc")
    publish_descriptors(agent, head_xfer_descs, "sender_head_desc")

    # Retrieve receiver's metadata and descriptors
    remote_name = retrieve_agent_metadata(agent, "receiver_meta", role_name="sender")
    if not remote_name:
        return

    receiver_tail_descs = retrieve_descriptors(agent, "receiver_tail_desc")
    receiver_buffers_descs = retrieve_descriptors(agent, "receiver_buffers_desc")
    # Note: receiver_head_desc not needed by sender

    logger.info(f"[sender] Connected to {remote_name}")

    # Create xfer_handlers using prep_xfer_dlist
    local_buffer_list = [(buffers_addr + i * BUFFER_SIZE, BUFFER_SIZE, 0) for i in range(NUM_BUFFERS)]
    local_buffers_prep = agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_buffer_list, "DRAM")

    remote_buffers_prep = agent.prep_xfer_dlist(remote_name, receiver_buffers_descs, "DRAM")

    tail_local_prep = agent.prep_xfer_dlist("NIXL_INIT_AGENT", [(tail_addr, 8, 0)], "DRAM")
    tail_remote_prep = agent.prep_xfer_dlist(remote_name, receiver_tail_descs, "DRAM")
    tail_xfer_handle = agent.make_prepped_xfer("WRITE", tail_local_prep, [0], tail_remote_prep, [0], b"TAIL_UPDATE")

    if not tail_xfer_handle or not local_buffers_prep or not remote_buffers_prep:
        logger.error("[sender] Failed to create transfer handles")
        return

    # Pre-create all buffer transfer handles (reuse them!)
    buffer_xfer_handles = []
    for i in range(NUM_BUFFERS):
        handle = agent.make_prepped_xfer(
            "WRITE",
            local_buffers_prep, [i],
            remote_buffers_prep, [i],
            f"BUFFER_{i}".encode('utf-8')
        )
        if not handle:
            logger.error(f"[sender] Failed to create buffer handle {i}")
            return
        buffer_xfer_handles.append(handle)

    logger.info(f"[sender] Prepared {NUM_BUFFERS} buffer slots for transfers")

    np_buffer = np.zeros(BUFFER_SIZE, dtype=np.uint8)
    np_header = np_buffer[:8].view(np.uint64)
    np_payload = np_buffer[8:]
    logger.info(f"[sender] Pre-allocated NumPy buffer ({BUFFER_SIZE / (1024 * 1024):.1f} MB)")

    # Init local Head and Tail to 0 (empty)
    local_tail = 0
    write_uint64(tail_addr, 0)
    write_uint64(head_addr, 0)

    # Transfer initial Tail value
    state = agent.transfer(tail_xfer_handle)
    if state == "ERR":
        logger.error("[sender] Failed to transfer initial tail")
        return

    # Wait for transfer to complete
    while agent.check_xfer_state(tail_xfer_handle) != "DONE":
        time.sleep(0.001)

    logger.info("[sender] Initialized, starting main loop")

    # Main loop
    transfers_sent = 0

    # Performance tracking
    start_time = time.time()
    first_transfer_time = None

    while transfers_sent < NUM_TRANSFERS:
        # Read local head (receiver writes here)
        remote_head = read_uint64(head_addr)

        # Check if not full: (tail + 1) % NUM_BUFFERS != head
        next_tail = (local_tail + 1) % NUM_BUFFERS
        if next_tail != remote_head:
            # Prepare data in local buffer
            buffer_idx = local_tail % NUM_BUFFERS
            buffer_offset = buffers_addr + (buffer_idx * BUFFER_SIZE)

            # Prepare data using NumPy arrays
            np_header[0] = transfers_sent  # Write transfer ID directly
            np_payload.fill(transfers_sent % 256)  # Fill payload
            write_data(buffer_offset, np_buffer)  # Write entire NumPy buffer

            # Track first transfer time
            if first_transfer_time is None:
                first_transfer_time = time.time()

            # Transfer buffer using prepped transfer
            buffer_xfer_handle = buffer_xfer_handles[buffer_idx]
            state = agent.transfer(buffer_xfer_handle)

            if state == "ERR":
                logger.error("[sender] Transfer buffer failed")
                break

            # Wait for buffer transfer to complete
            while agent.check_xfer_state(buffer_xfer_handle) != "DONE":
                time.sleep(0.0001)

            # Update tail
            local_tail = (local_tail + 1) % NUM_BUFFERS
            write_uint64(tail_addr, local_tail)

            # Transfer tail
            state = agent.transfer(tail_xfer_handle)

            if state == "ERR":
                logger.error("[sender] Transfer tail failed")
                break

            # Wait for tail transfer to complete before next iteration
            while agent.check_xfer_state(tail_xfer_handle) != "DONE":
                time.sleep(0.0001)

            transfers_sent += 1

            if transfers_sent % 10 == 0:
                logger.info(f"[sender] Sent {transfers_sent}/{NUM_TRANSFERS}")
        else:
            # Queue is full, wait
            time.sleep(0.0001)

    end_time = time.time()

    # Calculate performance metrics
    total_time = end_time - start_time
    if first_transfer_time:
        actual_transfer_time = end_time - first_transfer_time
    else:
        actual_transfer_time = total_time

    total_bytes = transfers_sent * BUFFER_SIZE
    bandwidth_mbps = (total_bytes / actual_transfer_time) / (1024 * 1024) if actual_transfer_time > 0 else 0

    logger.info(f"[sender] Completed {transfers_sent} transfers in {actual_transfer_time:.3f}s")
    logger.info(f"[sender] Bandwidth: {bandwidth_mbps:.2f} MB/s")

    # Cleanup
    agent.release_xfer_handle(tail_xfer_handle)
    # Release pre-created buffer handles
    for handle in buffer_xfer_handles:
        agent.release_xfer_handle(handle)
    agent.release_dlist_handle(local_buffers_prep)
    agent.release_dlist_handle(remote_buffers_prep)
    agent.release_dlist_handle(tail_local_prep)
    agent.release_dlist_handle(tail_remote_prep)
    agent.deregister_memory(tail_reg_descs)
    agent.deregister_memory(buffers_reg_descs)
    agent.deregister_memory(head_reg_descs)
    nixl_utils.free_passthru(tail_addr)
    nixl_utils.free_passthru(buffers_addr)
    nixl_utils.free_passthru(head_addr)


def run_test(num_buffers, buffer_size, num_transfers):
    """Run a single test with given parameters"""
    global NUM_BUFFERS, BUFFER_SIZE, NUM_TRANSFERS
    NUM_BUFFERS = num_buffers
    BUFFER_SIZE = buffer_size
    NUM_TRANSFERS = num_transfers

    tcp_server.clear_metadata("127.0.0.1", 9998)

    receiver_proc = Process(target=receiver_process)
    sender_proc = Process(target=sender_process)

    receiver_proc.start()
    sender_proc.start()

    receiver_proc.join(timeout=15)
    sender_proc.join(timeout=15)

    success = receiver_proc.exitcode == 0 and sender_proc.exitcode == 0

    # Terminate if hanging
    if receiver_proc.is_alive():
        receiver_proc.terminate()
    if sender_proc.is_alive():
        sender_proc.terminate()

    return success


def main():
    if "NIXL_PLUGIN_DIR" not in os.environ:
        logger.error("[main] NIXL_PLUGIN_DIR not set")
        sys.exit(1)

    # Start TCP server
    try:
        tcp_server.start_server(9998)
        time.sleep(0.2)
    except OSError:
        pass  # Server may already be running

    logger.info(f"[main] Starting sender-receiver: queue_size={NUM_BUFFERS}, num_transfers={NUM_TRANSFERS}, buffer_size={BUFFER_SIZE}")
    success = run_test(NUM_BUFFERS, BUFFER_SIZE, NUM_TRANSFERS)
    if success:
        logger.info("[main] ✓ Success!")
    else:
        logger.error("[main] ✗ Error")


if __name__ == "__main__":
    main()
