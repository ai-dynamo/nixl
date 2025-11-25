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

import ctypes
import os
import sys
import time
from multiprocessing import Process

import numpy as np
import tcp_server
from nixl_memory_utils import read_data, read_uint8, read_uint64, write_data, write_uint8, write_uint64
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

NUM_BUFFERS = 64  # Queue size
BUFFER_SIZE = 16 * 1024 * 1024  # 16MB - larger = more efficient RDMA
NUM_TRANSFERS = 1000  # Many transfers to amortize overhead
HEAD_UPDATE_INTERVAL = max(1, NUM_BUFFERS // 2)  # Not used in streaming mode


# Define offsets within the memory allocation (using uint8 for head/tail to save space)
HEAD_OFFSET = 0              # Head pointer at offset 0 (1 byte for uint8, values 0-NUM_BUFFERS)
TAIL_OFFSET = 1              # Tail pointer at offset 1 (1 byte for uint8, values 0-NUM_BUFFERS)
BUFFER_BASE_OFFSET = 2       # Buffers start at offset 2 
BUFFER_ENTRY_SIZE = BUFFER_SIZE  # Each buffer's size (in bytes)
TOTAL_MEMORY_SIZE = BUFFER_BASE_OFFSET + NUM_BUFFERS * BUFFER_ENTRY_SIZE

def get_buffer_offset(i):
    """Return the offset for buffer i within the memory allocation"""
    return BUFFER_BASE_OFFSET + i * BUFFER_ENTRY_SIZE


def receiver_process():
    """Receiver - streaming mode with sequence number polling (no head/tail RDMA)"""
    logger.info("[receiver] Starting")

    # Create NIXL agent
    config = nixl_agent_config(backends=["UCX"])
    agent = nixl_agent("receiver", config)

    # Allocate and register shared memory for buffers only
    memory_addr = nixl_utils.malloc_passthru(TOTAL_MEMORY_SIZE)
    memory_reg_desc = [(memory_addr, TOTAL_MEMORY_SIZE, 0, "shared_memory")]
    memory_reg_descs = agent.get_reg_descs(memory_reg_desc, "DRAM")
    agent.register_memory(memory_reg_descs)

    # Create buffer descriptors (sender will RDMA write to these)
    buffers_xfer_desc = [(memory_addr + BUFFER_BASE_OFFSET + i * BUFFER_ENTRY_SIZE, BUFFER_SIZE, 0) for i in range(NUM_BUFFERS)]
    buffers_xfer_descs = agent.get_xfer_descs(buffers_xfer_desc, "DRAM")

    logger.info(f"[receiver] Allocated shared memory at 0x{memory_addr:x}, size {TOTAL_MEMORY_SIZE} bytes")

    # Initialize all buffer headers to -1 (invalid sequence number)
    buffer_base_addr = memory_addr + BUFFER_BASE_OFFSET
    for i in range(NUM_BUFFERS):
        write_uint64(buffer_base_addr + i * BUFFER_ENTRY_SIZE, 0xFFFFFFFFFFFFFFFF)

    # Exchange metadata and descriptors
    publish_agent_metadata(agent, "receiver_metadata")
    publish_descriptors(agent, buffers_xfer_descs, "receiver_buffers_desc")
    
    # Retrieve sender's metadata
    sender_name = retrieve_agent_metadata(agent, "sender_metadata", role_name="receiver")
    if not sender_name:
        return

    logger.info(f"[receiver] Connected to {sender_name}")
    logger.info("[receiver] Initialized, starting main loop")

    # Main loop - poll buffer headers for sequence numbers (no head/tail RDMA!)
    transfers_received = 0

    # Performance tracking
    start_time = time.time()
    first_transfer_time = None
    time_poll = 0
    time_verify = 0

    while transfers_received < NUM_TRANSFERS:
        buffer_idx = transfers_received % NUM_BUFFERS
        buffer_offset = buffer_base_addr + (buffer_idx * BUFFER_ENTRY_SIZE)
        
        # Poll until we see the expected sequence number in the buffer header
        t0 = time.perf_counter()
        while True:
            seq = read_uint64(buffer_offset)
            if seq == transfers_received:
                break
            # Still waiting for data
        time_poll += time.perf_counter() - t0

        # Track first transfer time
        if first_transfer_time is None:
            first_transfer_time = time.time()

        # Verify sequence (already checked in poll, but for safety)
        t0 = time.perf_counter()
        if seq != transfers_received:
            logger.error(f"[receiver] Mismatch! Expected {transfers_received}, got {seq}")
        time_verify += time.perf_counter() - t0

        # Reset buffer header for next round (if wrapping)
        write_uint64(buffer_offset, 0xFFFFFFFFFFFFFFFF)

        transfers_received += 1

        if transfers_received % 100 == 0:
            logger.info(f"[receiver] Processed {transfers_received}/{NUM_TRANSFERS}")

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
    
    # Timing breakdown
    logger.info(f"[receiver] Timing breakdown:")
    logger.info(f"  Poll for data:  {time_poll*1000:.2f} ms ({time_poll/actual_transfer_time*100:.1f}%)")
    logger.info(f"  Verify:         {time_verify*1000:.2f} ms ({time_verify/actual_transfer_time*100:.1f}%)")
    total_measured = time_poll + time_verify
    logger.info(f"  Other/overhead: {(actual_transfer_time-total_measured)*1000:.2f} ms ({(actual_transfer_time-total_measured)/actual_transfer_time*100:.1f}%)")

    # Wait a bit for sender to finish its final checks before cleanup
    time.sleep(0.5)

    # Cleanup
    agent.deregister_memory(memory_reg_descs)
    nixl_utils.free_passthru(memory_addr)


def sender_process():
    """Sender - streaming mode with sequence numbers (no head/tail RDMA)"""
    logger.info("[sender] Starting")

    # Create NIXL agent
    config = nixl_agent_config(backends=["UCX"])
    agent = nixl_agent("sender", config)

    # Allocate buffers only (no head/tail pointers needed)
    buffers_size = NUM_BUFFERS * BUFFER_SIZE
    buffers_addr = nixl_utils.malloc_passthru(buffers_size)

    # Register buffers
    buffers_reg_desc = [(buffers_addr, buffers_size, 0, "buffers")]
    buffers_reg_descs = agent.get_reg_descs(buffers_reg_desc, "DRAM")
    agent.register_memory(buffers_reg_descs)

    logger.info(f"[sender] Allocated buffers at 0x{buffers_addr:x}, size {buffers_size} bytes")

    # Exchange metadata
    publish_agent_metadata(agent, "sender_metadata")

    # Retrieve receiver's buffer descriptors
    remote_name = retrieve_agent_metadata(agent, "receiver_metadata", role_name="sender")
    if not remote_name:
        return

    receiver_buffers_descs = retrieve_descriptors(agent, "receiver_buffers_desc")
    logger.info(f"[sender] Connected to {remote_name}")

    # Create transfer handles for each buffer slot
    local_buffer_list = [(buffers_addr + i * BUFFER_SIZE, BUFFER_SIZE, 0) for i in range(NUM_BUFFERS)]
    local_buffers_prep = agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_buffer_list, "DRAM")
    remote_buffers_prep = agent.prep_xfer_dlist(remote_name, receiver_buffers_descs, "DRAM")

    if not local_buffers_prep or not remote_buffers_prep:
        logger.error("[sender] Failed to create prep lists")
        return

    # Pre-create transfer handles for each buffer slot
    buffer_xfer_handles = []
    for i in range(NUM_BUFFERS):
        handle = agent.make_prepped_xfer("WRITE", local_buffers_prep, [i], remote_buffers_prep, [i], f"BUF_{i}".encode())
        buffer_xfer_handles.append(handle)
    
    logger.info(f"[sender] Ready to transfer {NUM_BUFFERS} buffer slots ({BUFFER_SIZE / (1024 * 1024):.1f} MB each)")
    logger.info("[sender] Initialized, starting main loop")

    # Main loop - just send with sequence numbers, no flow control
    transfers_sent = 0

    # Performance tracking
    start_time = time.time()
    first_transfer_time = None
    
    # Timing breakdown
    time_write_header = 0
    time_transfer_buffer = 0
    time_wait_buffer = 0

    while transfers_sent < NUM_TRANSFERS:
        buffer_idx = transfers_sent % NUM_BUFFERS
        buffer_xfer_handle = buffer_xfer_handles[buffer_idx]
        
        # Wait if this buffer's previous transfer is still in progress
        t0 = time.perf_counter()
        try:
            while agent.check_xfer_state(buffer_xfer_handle) == "PROC":
                pass  # Spin wait for completion
        except Exception:
            pass  # Handle never used yet - ready to transfer
        time_wait_buffer += time.perf_counter() - t0
        
        # Write sequence number to buffer header
        buffer_offset = buffers_addr + (buffer_idx * BUFFER_SIZE)
        t0 = time.perf_counter()
        write_uint64(buffer_offset, transfers_sent)
        time_write_header += time.perf_counter() - t0

        # Track first transfer time
        if first_transfer_time is None:
            first_transfer_time = time.time()

        # Transfer buffer (fire-and-forget)
        t0 = time.perf_counter()
        state = agent.transfer(buffer_xfer_handle)
        time_transfer_buffer += time.perf_counter() - t0

        if state == "ERR":
            logger.error("[sender] Transfer buffer failed")
            break

        transfers_sent += 1

        if transfers_sent % 100 == 0:
            logger.info(f"[sender] Sent {transfers_sent}/{NUM_TRANSFERS}")

    # Record send completion time (before waiting for in-flight)
    send_end_time = time.time()

    # Wait for all in-flight transfers to complete (for clean shutdown)
    for i in range(NUM_BUFFERS):
        try:
            while agent.check_xfer_state(buffer_xfer_handles[i]) == "PROC":
                pass
        except Exception:
            pass

    end_time = time.time()

    # Calculate performance metrics
    total_time = end_time - start_time
    if first_transfer_time:
        actual_transfer_time = end_time - first_transfer_time
    else:
        actual_transfer_time = total_time

    total_bytes = transfers_sent * BUFFER_SIZE
    bandwidth_mbps = (total_bytes / actual_transfer_time) / (1024 * 1024) if actual_transfer_time > 0 else 0

    # Calculate send-only time (before waiting for completion)
    send_time = send_end_time - first_transfer_time if first_transfer_time else total_time
    send_bandwidth = (total_bytes / send_time) / (1024 * 1024) if send_time > 0 else 0

    logger.info(f"[sender] Completed {transfers_sent} transfers in {actual_transfer_time:.3f}s")
    logger.info(f"[sender] Bandwidth: {bandwidth_mbps:.2f} MB/s")
    logger.info(f"[sender] Send-only time: {send_time:.3f}s ({send_bandwidth:.2f} MB/s)")
    
    # Timing breakdown
    logger.info(f"[sender] Timing breakdown:")
    logger.info(f"  Write header:     {time_write_header*1000:.2f} ms ({time_write_header/actual_transfer_time*100:.1f}%)")
    logger.info(f"  Transfer buffer:  {time_transfer_buffer*1000:.2f} ms ({time_transfer_buffer/actual_transfer_time*100:.1f}%)")
    logger.info(f"  Wait for buffer:  {time_wait_buffer*1000:.2f} ms ({time_wait_buffer/actual_transfer_time*100:.1f}%)")
    total_measured = time_write_header + time_transfer_buffer + time_wait_buffer
    logger.info(f"  Other/overhead:   {(actual_transfer_time-total_measured)*1000:.2f} ms ({(actual_transfer_time-total_measured)/actual_transfer_time*100:.1f}%)")

    # Cleanup
    for handle in buffer_xfer_handles:
        agent.release_xfer_handle(handle)
    agent.release_dlist_handle(local_buffers_prep)
    agent.release_dlist_handle(remote_buffers_prep)
    agent.deregister_memory(buffers_reg_descs)
    nixl_utils.free_passthru(buffers_addr)


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
