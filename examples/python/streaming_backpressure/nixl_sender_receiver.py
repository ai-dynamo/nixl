#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
NIXL Sender-Receiver Example: Streaming Mode with Backpressure

Demonstrates a high-throughput producer-consumer pattern using:
- Sequence numbers in buffer headers for data arrival detection
- Notification-based backpressure to prevent buffer overruns
- Pre-created transfer handles for minimal per-transfer overhead

Usage:
    # Run both sender and receiver in one process (local testing)
    python3 nixl_sender_receiver.py

    # Run sender and receiver separately (distributed testing)
    python3 nixl_sender_receiver.py --mode receiver --tcp-host 10.0.0.1
    python3 nixl_sender_receiver.py --mode sender --tcp-host 10.0.0.1

    # Use etcd for metadata exchange
    python3 nixl_sender_receiver.py --use-etcd

    # Custom buffer configuration
    python3 nixl_sender_receiver.py --num-buffers 128 --buffer-size 33554432 --num-transfers 2000
"""

import argparse
import os
import sys
import time
from multiprocessing import Process

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import (  # noqa: E402
    clear_metadata,
    publish_agent_metadata,
    publish_descriptors,
    read_uint64,
    retrieve_agent_metadata,
    retrieve_descriptors,
    start_server,
    write_uint64,
)

import nixl._utils as nixl_utils  # noqa: E402
from nixl._api import nixl_agent, nixl_agent_config  # noqa: E402
from nixl.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def use_etcd():
    """Check if etcd mode is enabled (via environment variable for multiprocessing)"""
    return os.environ.get("NIXL_USE_ETCD", "0") == "1"


def receiver_process(num_buffers, buffer_size, num_transfers, tcp_host, tcp_port):
    """Receiver - streaming mode with sequence number polling (no head/tail RDMA)"""
    logger.info("[receiver] Starting")

    # Calculate constants from parameters
    progress_update_interval = max(1, num_buffers // 4)
    buffer_base_offset = 2  # Buffers start at offset 2
    buffer_entry_size = buffer_size
    total_memory_size = buffer_base_offset + num_buffers * buffer_entry_size

    # Variables for cleanup
    receiver_agent = None
    memory_addr = None
    memory_reg_descs = None

    try:
        # Create NIXL agent
        config = nixl_agent_config(backends=["UCX"])
        receiver_agent = nixl_agent("receiver", config)

        # Allocate and register shared memory for buffers only
        memory_addr = nixl_utils.malloc_passthru(total_memory_size)
        memory_reg_desc = [(memory_addr, total_memory_size, 0, "shared_memory")]
        memory_reg_descs = receiver_agent.get_reg_descs(memory_reg_desc, "DRAM")
        receiver_agent.register_memory(memory_reg_descs)

        # Create buffer descriptors (sender will RDMA write to these)
        buffers_xfer_desc = [
            (memory_addr + buffer_base_offset + i * buffer_entry_size, buffer_size, 0)
            for i in range(num_buffers)
        ]
        buffers_xfer_descs = receiver_agent.get_xfer_descs(buffers_xfer_desc, "DRAM")

        logger.info(
            f"[receiver] Allocated shared memory at 0x{memory_addr:x}, size {total_memory_size} bytes"
        )

        # Initialize all buffer headers to -1 (invalid sequence number)
        buffer_base_addr = memory_addr + buffer_base_offset
        for i in range(num_buffers):
            write_uint64(buffer_base_addr + i * buffer_entry_size, 0xFFFFFFFFFFFFFFFF)

        # Exchange metadata and descriptors
        if use_etcd():
            # Publish receiver metadata
            publish_agent_metadata(
                receiver_agent,
                "receiver_metadata",
                use_nixl_builtin=True,
                reg_descs=memory_reg_descs,
            )
            # Fetch sender metadata (for bidirectional notifications)
            sender_name = retrieve_agent_metadata(
                receiver_agent,
                "sender_metadata",
                role_name="receiver",
                use_nixl_builtin=True,
                remote_agent_name="sender",
            )
            if not sender_name:
                logger.error("[receiver] Failed to retrieve sender metadata")
                return

            # Exchange: receiver keeps sending READY until sender's READY arrives,
            # then sends DESCS. Sender keeps sending READY until DESCS arrives.
            logger.info("[receiver] Waiting for sender READY (retrying)...")
            got_ready = False
            while not got_ready:
                # Keep sending our READY so sender knows we're alive
                receiver_agent.send_notif("sender", b"READY")
                # Check for sender's READY
                notifs = receiver_agent.get_new_notifs()
                if "sender" in notifs and b"READY" in notifs["sender"]:
                    got_ready = True
                else:
                    time.sleep(0.01)
            logger.info("[receiver] Got sender READY, sending descriptors")

            # Now send DESCS - sender is waiting for this
            serialized_descs = receiver_agent.get_serialized_descs(buffers_xfer_descs)
            receiver_agent.send_notif("sender", b"DESCS:" + serialized_descs)
            logger.info("[receiver] Sent buffer descriptors to sender")
        else:
            publish_agent_metadata(receiver_agent, "receiver_metadata")
            publish_descriptors(
                receiver_agent, buffers_xfer_descs, "receiver_buffers_desc"
            )
            sender_name = retrieve_agent_metadata(
                receiver_agent, "sender_metadata", role_name="receiver"
            )

        if not sender_name:
            logger.error("[receiver] Failed to connect to sender")
            return

        logger.info(f"[receiver] Connected to {sender_name}")
        logger.info("[receiver] Initialized, starting main loop")

        # Main loop - poll buffer headers for sequence numbers with backpressure notifications
        transfers_received = 0
        progress_updates_sent = 0
        sequence_mismatches = 0

        # Performance tracking
        start_time = time.time()
        first_transfer_time = None
        time_poll = 0
        time_verify = 0
        time_notify = 0

        while transfers_received < num_transfers:
            buffer_idx = transfers_received % num_buffers
            buffer_offset = buffer_base_addr + (buffer_idx * buffer_entry_size)

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

            # Verify sequence - mismatch means buffer overrun!
            t0 = time.perf_counter()
            if seq != transfers_received:
                logger.error(
                    f"[receiver] Mismatch! Expected {transfers_received}, got {seq}"
                )
                sequence_mismatches += 1
            time_verify += time.perf_counter() - t0

            # Reset buffer header for next round (if wrapping)
            write_uint64(buffer_offset, 0xFFFFFFFFFFFFFFFF)

            transfers_received += 1

            # Send progress notification to sender (batched for efficiency)
            if transfers_received % progress_update_interval == 0:
                t0 = time.perf_counter()
                receiver_agent.send_notif(
                    sender_name, f"P:{transfers_received}".encode()
                )
                progress_updates_sent += 1
                time_notify += time.perf_counter() - t0

            if transfers_received % 100 == 0:
                logger.info(
                    f"[receiver] Processed {transfers_received}/{num_transfers}"
                )

        end_time = time.time()

        # Calculate performance metrics
        total_time = end_time - start_time
        if first_transfer_time:
            actual_transfer_time = end_time - first_transfer_time
        else:
            actual_transfer_time = total_time

        total_bytes = transfers_received * buffer_size
        bandwidth_mbps = (
            (total_bytes / actual_transfer_time) / (1024 * 1024)
            if actual_transfer_time > 0
            else 0
        )

        logger.info(
            f"[receiver] Completed {transfers_received} transfers in {actual_transfer_time:.3f}s"
        )
        logger.info(f"[receiver] Bandwidth: {bandwidth_mbps:.2f} MB/s")
        logger.info(f"[receiver] Progress updates sent: {progress_updates_sent}")
        if sequence_mismatches == 0:
            logger.info("[receiver] ✓ No buffer overrun (0 mismatches)")
        else:
            logger.error(
                f"[receiver] ⚠️  BUFFER OVERRUN: {sequence_mismatches} mismatches!"
            )

        # Timing breakdown
        logger.info("[receiver] Timing breakdown:")
        logger.info(
            f"  Poll for data:  {time_poll * 1000:.2f} ms ({time_poll / actual_transfer_time * 100:.1f}%)"
        )
        logger.info(
            f"  Verify:         {time_verify * 1000:.2f} ms ({time_verify / actual_transfer_time * 100:.1f}%)"
        )
        logger.info(
            f"  Send notifs:    {time_notify * 1000:.2f} ms ({time_notify / actual_transfer_time * 100:.1f}%)"
        )
        total_measured = time_poll + time_verify + time_notify
        logger.info(
            f"  Other/overhead: {(actual_transfer_time - total_measured) * 1000:.2f} ms ({(actual_transfer_time - total_measured) / actual_transfer_time * 100:.1f}%)"
        )

        # Wait a bit for sender to finish its final checks before cleanup
        time.sleep(0.5)

    except Exception:
        logger.exception("[receiver] Fatal error during transfer")
        raise
    finally:
        # ========================================
        # Teardown
        # ========================================
        logger.info("[receiver] Tearing down...")
        try:
            if receiver_agent and memory_reg_descs:
                receiver_agent.deregister_memory(memory_reg_descs)
        except Exception:
            logger.exception("[receiver] Error deregistering memory")

        try:
            if memory_addr:
                nixl_utils.free_passthru(memory_addr)
        except Exception:
            logger.exception("[receiver] Error freeing memory")

        logger.info("[receiver] Teardown complete")


def sender_process(num_buffers, buffer_size, num_transfers, tcp_host, tcp_port):
    """Sender - streaming mode with sequence numbers (no head/tail RDMA)"""
    logger.info("[sender] Starting")

    # Calculate constants from parameters
    progress_update_interval = max(1, num_buffers // 4)
    backpressure_threshold = num_buffers - 4

    # Variables for cleanup
    sender_agent = None
    buffers_addr = None
    buffers_reg_descs = None
    buffer_xfer_handles = []
    local_buffers_prep = None
    remote_buffers_prep = None

    try:
        # Create NIXL agent
        config = nixl_agent_config(backends=["UCX"])
        sender_agent = nixl_agent("sender", config)

        # Allocate buffers only (no head/tail pointers needed)
        buffers_size = num_buffers * buffer_size
        buffers_addr = nixl_utils.malloc_passthru(buffers_size)

        # Register buffers
        buffers_reg_desc = [(buffers_addr, buffers_size, 0, "buffers")]
        buffers_reg_descs = sender_agent.get_reg_descs(buffers_reg_desc, "DRAM")
        sender_agent.register_memory(buffers_reg_descs)

        logger.info(
            f"[sender] Allocated buffers at 0x{buffers_addr:x}, size {buffers_size} bytes"
        )

        # Exchange metadata
        if use_etcd():
            # Publish sender metadata first
            publish_agent_metadata(
                sender_agent,
                "sender_metadata",
                use_nixl_builtin=True,
                reg_descs=buffers_reg_descs,
            )
            # Fetch receiver metadata (for bidirectional notifications)
            receiver_name = retrieve_agent_metadata(
                sender_agent,
                "receiver_metadata",
                role_name="sender",
                use_nixl_builtin=True,
                remote_agent_name="receiver",
            )
            if not receiver_name:
                logger.error("[sender] Failed to retrieve receiver metadata")
                return

            # Exchange: sender keeps sending READY until DESCS arrives
            # (DESCS is proof that receiver got our READY)
            logger.info("[sender] Waiting for receiver descriptors (sending READY)...")
            receiver_buffers_descs = None
            ready_count = 0
            while receiver_buffers_descs is None:
                # Send READY periodically (not every iteration to avoid flooding)
                if ready_count % 5 == 0:
                    sender_agent.send_notif("receiver", b"READY")
                ready_count += 1
                # Check for receiver's DESCS (check a few times before sleeping)
                for _ in range(3):
                    notifs = sender_agent.get_new_notifs()
                    if "receiver" in notifs:
                        for msg in notifs["receiver"]:
                            if msg.startswith(b"DESCS:"):
                                serialized_descs = msg[6:]  # Remove "DESCS:" prefix
                                receiver_buffers_descs = sender_agent.deserialize_descs(
                                    serialized_descs
                                )
                                logger.info(
                                    "[sender] Received buffer descriptors from receiver"
                                )
                                break
                    if receiver_buffers_descs is not None:
                        break
                if receiver_buffers_descs is None:
                    time.sleep(0.01)
        else:
            publish_agent_metadata(sender_agent, "sender_metadata")
            receiver_name = retrieve_agent_metadata(
                sender_agent, "receiver_metadata", role_name="sender"
            )
            receiver_buffers_descs = retrieve_descriptors(
                sender_agent, "receiver_buffers_desc"
            )

        if not receiver_name:
            logger.error("[sender] Failed to connect to receiver")
            return

        logger.info(f"[sender] Connected to {receiver_name}")

        # Create transfer handles for each buffer slot
        local_buffer_list = [
            (buffers_addr + i * buffer_size, buffer_size, 0) for i in range(num_buffers)
        ]
        local_buffers_prep = sender_agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_buffer_list, "DRAM"
        )
        remote_buffers_prep = sender_agent.prep_xfer_dlist(
            receiver_name, receiver_buffers_descs, "DRAM"
        )

        if not local_buffers_prep or not remote_buffers_prep:
            logger.error("[sender] Failed to create prep lists")
            return

        # Pre-create transfer handles for each buffer slot
        for i in range(num_buffers):
            handle = sender_agent.make_prepped_xfer(
                "WRITE",
                local_buffers_prep,
                [i],
                remote_buffers_prep,
                [i],
                f"BUF_{i}".encode(),
            )
            buffer_xfer_handles.append(handle)

        logger.info(
            f"[sender] Ready to transfer {num_buffers} buffer slots ({buffer_size / (1024 * 1024):.1f} MB each)"
        )
        logger.info("[sender] Initialized, starting main loop")

        # Main loop - send with sequence numbers and backpressure support
        transfers_sent = 0
        receiver_progress = 0  # Last known receiver progress
        backpressure_checks = 0
        backpressure_waits = 0
        max_ahead = 0  # Track maximum distance sender got ahead of receiver

        # Performance tracking
        start_time = time.time()
        first_transfer_time = None

        # Timing breakdown
        time_write_header = 0
        time_transfer_buffer = 0
        time_wait_buffer = 0
        time_backpressure = 0

        while transfers_sent < num_transfers:
            buffer_idx = transfers_sent % num_buffers
            buffer_xfer_handle = buffer_xfer_handles[buffer_idx]

            # Backpressure check: if we're too far ahead of receiver, wait for progress
            t0 = time.perf_counter()
            ahead_count = transfers_sent - receiver_progress
            if ahead_count > max_ahead:
                max_ahead = ahead_count
            if ahead_count >= backpressure_threshold:
                backpressure_checks += 1
                # Poll for progress notifications from receiver
                while ahead_count >= backpressure_threshold:
                    notifs = sender_agent.get_new_notifs()
                    if receiver_name in notifs:
                        for msg in notifs[receiver_name]:
                            if msg.startswith(b"P:"):
                                progress = int(msg[2:])
                                if progress > receiver_progress:
                                    receiver_progress = progress
                    ahead_count = transfers_sent - receiver_progress
                    if ahead_count >= backpressure_threshold:
                        backpressure_waits += 1
                        time.sleep(0.0001)  # 100us sleep to avoid busy spinning
            time_backpressure += time.perf_counter() - t0

            # Wait if this buffer's previous transfer is still in progress
            t0 = time.perf_counter()
            try:
                while sender_agent.check_xfer_state(buffer_xfer_handle) == "PROC":
                    pass  # Spin wait for completion
            except Exception:
                pass  # Handle never used yet - ready to transfer
            time_wait_buffer += time.perf_counter() - t0

            # Write sequence number to buffer header
            buffer_offset = buffers_addr + (buffer_idx * buffer_size)
            t0 = time.perf_counter()
            write_uint64(buffer_offset, transfers_sent)
            time_write_header += time.perf_counter() - t0

            # Track first transfer time
            if first_transfer_time is None:
                first_transfer_time = time.time()

            # Transfer buffer (fire-and-forget)
            t0 = time.perf_counter()
            state = sender_agent.transfer(buffer_xfer_handle)
            time_transfer_buffer += time.perf_counter() - t0

            if state == "ERR":
                logger.error("[sender] Transfer buffer failed")
                break

            transfers_sent += 1

            # Opportunistically check for progress updates (non-blocking)
            if transfers_sent % progress_update_interval == 0:
                notifs = sender_agent.get_new_notifs()
                if receiver_name in notifs:
                    for msg in notifs[receiver_name]:
                        if msg.startswith(b"P:"):
                            progress = int(msg[2:])
                            if progress > receiver_progress:
                                receiver_progress = progress

            if transfers_sent % 100 == 0:
                logger.info(
                    f"[sender] Sent {transfers_sent}/{num_transfers} (receiver at {receiver_progress})"
                )

        # Record send completion time (before waiting for in-flight)
        send_end_time = time.time()

        # Wait for all in-flight transfers to complete (for clean shutdown)
        for i in range(num_buffers):
            try:
                while sender_agent.check_xfer_state(buffer_xfer_handles[i]) == "PROC":
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

        total_bytes = transfers_sent * buffer_size
        bandwidth_mbps = (
            (total_bytes / actual_transfer_time) / (1024 * 1024)
            if actual_transfer_time > 0
            else 0
        )

        # Calculate send-only time (before waiting for completion)
        send_time = (
            send_end_time - first_transfer_time if first_transfer_time else total_time
        )
        send_bandwidth = (
            (total_bytes / send_time) / (1024 * 1024) if send_time > 0 else 0
        )

        logger.info(
            f"[sender] Completed {transfers_sent} transfers in {actual_transfer_time:.3f}s"
        )
        logger.info(f"[sender] Bandwidth: {bandwidth_mbps:.2f} MB/s")
        logger.info(
            f"[sender] Send-only time: {send_time:.3f}s ({send_bandwidth:.2f} MB/s)"
        )
        logger.info(
            f"[sender] Backpressure: {backpressure_checks} checks, {backpressure_waits * 0.1:.1f}ms wait, max ahead: {max_ahead}/{num_buffers}"
        )

        # Timing breakdown
        logger.info("[sender] Timing breakdown:")
        logger.info(
            f"  Write header:     {time_write_header * 1000:.2f} ms ({time_write_header / actual_transfer_time * 100:.1f}%)"
        )
        logger.info(
            f"  Transfer buffer:  {time_transfer_buffer * 1000:.2f} ms ({time_transfer_buffer / actual_transfer_time * 100:.1f}%)"
        )
        logger.info(
            f"  Wait for buffer:  {time_wait_buffer * 1000:.2f} ms ({time_wait_buffer / actual_transfer_time * 100:.1f}%)"
        )
        logger.info(
            f"  Backpressure:     {time_backpressure * 1000:.2f} ms ({time_backpressure / actual_transfer_time * 100:.1f}%)"
        )
        total_measured = (
            time_write_header
            + time_transfer_buffer
            + time_wait_buffer
            + time_backpressure
        )
        logger.info(
            f"  Other/overhead:   {(actual_transfer_time - total_measured) * 1000:.2f} ms ({(actual_transfer_time - total_measured) / actual_transfer_time * 100:.1f}%)"
        )

    except Exception:
        logger.exception("[sender] Fatal error during transfer")
        raise
    finally:
        # ========================================
        # Teardown
        # ========================================
        logger.info("[sender] Tearing down...")
        try:
            for handle in buffer_xfer_handles:
                sender_agent.release_xfer_handle(handle)
        except Exception:
            logger.exception("[sender] Error releasing transfer handles")

        try:
            if sender_agent and local_buffers_prep:
                sender_agent.release_dlist_handle(local_buffers_prep)
        except Exception:
            logger.exception("[sender] Error releasing local prep handle")

        try:
            if sender_agent and remote_buffers_prep:
                sender_agent.release_dlist_handle(remote_buffers_prep)
        except Exception:
            logger.exception("[sender] Error releasing remote prep handle")

        try:
            if sender_agent and buffers_reg_descs:
                sender_agent.deregister_memory(buffers_reg_descs)
        except Exception:
            logger.exception("[sender] Error deregistering memory")

        try:
            if buffers_addr:
                nixl_utils.free_passthru(buffers_addr)
        except Exception:
            logger.exception("[sender] Error freeing memory")

        logger.info("[sender] Teardown complete")


def run_both(num_buffers, buffer_size, num_transfers, tcp_host, tcp_port):
    """Run both sender and receiver processes for local testing"""
    if not use_etcd():
        clear_metadata(tcp_host, tcp_port)

    receiver_proc = Process(
        target=receiver_process,
        args=(num_buffers, buffer_size, num_transfers, tcp_host, tcp_port),
    )
    sender_proc = Process(
        target=sender_process,
        args=(num_buffers, buffer_size, num_transfers, tcp_host, tcp_port),
    )

    receiver_proc.start()
    sender_proc.start()

    receiver_proc.join(timeout=30)
    sender_proc.join(timeout=30)

    success = receiver_proc.exitcode == 0 and sender_proc.exitcode == 0

    # Terminate if hanging
    if receiver_proc.is_alive():
        logger.warning("[main] Receiver process hanging, terminating...")
        receiver_proc.terminate()
    if sender_proc.is_alive():
        logger.warning("[main] Sender process hanging, terminating...")
        sender_proc.terminate()

    return success


def parse_args():
    parser = argparse.ArgumentParser(
        description="NIXL Sender-Receiver Example: Streaming with Backpressure"
    )
    parser.add_argument(
        "--mode",
        choices=["both", "sender", "receiver"],
        default="both",
        help="Run mode: 'both' spawns both processes (local testing), 'sender'/'receiver' runs single role (distributed testing)",
    )
    parser.add_argument(
        "--num-buffers",
        type=int,
        default=64,
        help="Number of circular buffer slots (default: 64)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=16 * 1024 * 1024,
        help="Size of each buffer in bytes (default: 16MB)",
    )
    parser.add_argument(
        "--num-transfers",
        type=int,
        default=1000,
        help="Number of transfers to perform (default: 1000)",
    )
    parser.add_argument(
        "--tcp-host",
        default="127.0.0.1",
        help="TCP metadata server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=9998,
        help="TCP metadata server port (default: 9998)",
    )
    parser.add_argument(
        "--use-etcd",
        action="store_true",
        help="Use NIXL's built-in etcd for metadata exchange (requires NIXL built with etcd support and NIXL_ETCD_ENDPOINTS env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set environment variable so child processes can see it
    if args.use_etcd:
        os.environ["NIXL_USE_ETCD"] = "1"

    if "NIXL_PLUGIN_DIR" not in os.environ:
        logger.error("[main] NIXL_PLUGIN_DIR not set")
        exit(1)

    # Setup metadata exchange
    if use_etcd():
        # Set default etcd endpoint if not defined
        if not os.environ.get("NIXL_ETCD_ENDPOINTS"):
            os.environ["NIXL_ETCD_ENDPOINTS"] = "http://127.0.0.1:2379"
            logger.info("[main] Using default etcd endpoint: http://127.0.0.1:2379")

        # Verify etcd is running
        import urllib.request

        try:
            with urllib.request.urlopen(
                os.environ["NIXL_ETCD_ENDPOINTS"] + "/version", timeout=2
            ) as resp:
                if resp.status == 200:
                    logger.info(
                        "[main] etcd is running, using NIXL built-in etcd for metadata exchange"
                    )
        except Exception as e:
            logger.error(
                f"[main] etcd not available at {os.environ['NIXL_ETCD_ENDPOINTS']}: {e}"
            )
            exit(1)

        # Clear stale metadata from previous runs
        import subprocess

        result = subprocess.run(
            ["etcdctl", "del", "--prefix", "/nixl/"],
            env={**os.environ, "ETCDCTL_API": "3"},
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"[main] Cleared {result.stdout.strip()} stale etcd keys")
    else:
        # Start TCP server if needed
        if args.mode in ["both", "receiver"]:
            try:
                start_server(args.tcp_port)
                time.sleep(0.2)
                logger.info(
                    f"[main] Started TCP metadata server on {args.tcp_host}:{args.tcp_port}"
                )
            except OSError:
                logger.info(
                    f"[main] TCP metadata server already running on {args.tcp_host}:{args.tcp_port}"
                )
        logger.info("[main] Using TCP server for metadata exchange")

    logger.info(
        f"[main] Configuration: mode={args.mode}, num_buffers={args.num_buffers}, "
        f"buffer_size={args.buffer_size / (1024 * 1024):.1f}MB, num_transfers={args.num_transfers}"
    )

    # Run based on mode
    try:
        if args.mode == "both":
            logger.info("[main] Starting both sender and receiver processes...")
            success = run_both(
                args.num_buffers,
                args.buffer_size,
                args.num_transfers,
                args.tcp_host,
                args.tcp_port,
            )
            if success:
                logger.info("[main] ✓ Success!")
            else:
                logger.error("[main] ✗ Error")
                exit(1)
        elif args.mode == "receiver":
            logger.info("[main] Starting receiver...")
            receiver_process(
                args.num_buffers,
                args.buffer_size,
                args.num_transfers,
                args.tcp_host,
                args.tcp_port,
            )
            logger.info("[main] Receiver complete")
        elif args.mode == "sender":
            logger.info("[main] Starting sender...")
            sender_process(
                args.num_buffers,
                args.buffer_size,
                args.num_transfers,
                args.tcp_host,
                args.tcp_port,
            )
            logger.info("[main] Sender complete")
    except KeyboardInterrupt:
        logger.info("[main] Interrupted by user")
        exit(130)
    except Exception:
        logger.exception("[main] Fatal error")
        exit(1)


if __name__ == "__main__":
    main()
