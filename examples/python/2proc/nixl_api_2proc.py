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
NIXL Two-Process Example

Usage:
    python3 nixl_api_2proc.py              # Use TCP server for metadata
    python3 nixl_api_2proc.py --use-etcd   # Use NIXL's built-in etcd
"""

import argparse
import os
import sys
import time
from multiprocessing import Process

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nixl_utils import (  # noqa: E402
    clear_metadata,
    publish_agent_metadata,
    publish_descriptors,
    retrieve_agent_metadata,
    retrieve_descriptors,
    start_server,
)

import nixl._utils as nixl_utils  # noqa: E402
from nixl._api import nixl_agent, nixl_agent_config  # noqa: E402
from nixl.logging import get_logger  # noqa: E402

# Configure logging
logger = get_logger(__name__)


def use_etcd():
    """Check if etcd mode is enabled (via environment variable for multiprocessing)"""
    return os.environ.get("NIXL_USE_ETCD", "0") == "1"


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
    if use_etcd():
        # etcd mode: publish agent metadata, then exchange descriptors via notifications
        publish_agent_metadata(
            target_agent,
            "target_meta",
            use_nixl_builtin=True,
            reg_descs=target_reg_descs,
        )
        logger.info("[target] Published metadata via NIXL etcd")

        # Fetch initiator's metadata so we can send notifications to them
        logger.info("[target] Waiting for initiator metadata...")
        initiator_name = retrieve_agent_metadata(
            target_agent,
            "initiator_meta",
            role_name="target",
            use_nixl_builtin=True,
            remote_agent_name="initiator",
        )
        if not initiator_name:
            logger.error("[target] Failed to fetch initiator metadata")
            return
        logger.info("[target] Loaded initiator metadata")

        # Exchange: target keeps sending READY until initiator's READY arrives,
        # then sends DESCS. Initiator keeps sending READY until DESCS arrives.
        logger.info("[target] Waiting for initiator READY (retrying)...")
        got_ready = False
        while not got_ready:
            # Keep sending our READY so initiator knows we're alive
            target_agent.send_notif("initiator", b"READY")
            # Check for initiator's READY
            notifs = target_agent.get_new_notifs()
            if "initiator" in notifs and b"READY" in notifs["initiator"]:
                got_ready = True
            else:
                time.sleep(0.01)
        logger.info("[target] Got initiator READY, sending descriptors")

        # Now send DESCS - initiator is waiting for this
        serialized_descs = target_agent.get_serialized_descs(target_xfer_descs)
        target_agent.send_notif("initiator", b"DESCS:" + serialized_descs)
        logger.info("[target] Sent descriptors to initiator")
    else:
        # TCP mode: publish both metadata and descriptors to server
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
    initiator_strings = [
        (initiator_addr, buf_size, 0, "a"),
        (initiator_addr2, buf_size, 0, "b"),
    ]

    initiator_reg_descs = initiator_agent.get_reg_descs(initiator_strings, "DRAM")
    initiator_xfer_descs = initiator_agent.get_xfer_descs(initiator_addrs, "DRAM")

    initiator_descs = initiator_agent.register_memory(initiator_reg_descs)
    assert initiator_descs is not None
    logger.info("[initiator] Memory registered")

    # Retrieve target's metadata and descriptors
    if use_etcd():
        # etcd mode: publish our metadata first, then fetch target's metadata
        # Both sides need each other's metadata for bidirectional notifications
        publish_agent_metadata(
            initiator_agent,
            "initiator_meta",
            use_nixl_builtin=True,
            reg_descs=initiator_reg_descs,
        )
        logger.info("[initiator] Published metadata via NIXL etcd")

        # Fetch target's metadata
        remote_name = retrieve_agent_metadata(
            initiator_agent,
            "target_meta",
            role_name="initiator",
            use_nixl_builtin=True,
            remote_agent_name="target",
        )
        if not remote_name:
            logger.error("[initiator] Failed to retrieve target metadata from etcd")
            return
        logger.info("[initiator] Loaded target metadata")

        # Exchange: initiator keeps sending READY until DESCS arrives
        # (DESCS is proof that target got our READY)
        logger.info("[initiator] Waiting for target descriptors (sending READY)...")
        target_xfer_descs = None
        ready_count = 0
        while target_xfer_descs is None:
            # Send READY periodically (not every iteration to avoid flooding)
            if ready_count % 5 == 0:
                initiator_agent.send_notif("target", b"READY")
            ready_count += 1
            # Check for target's DESCS (check a few times before sleeping)
            for _ in range(3):
                notifs = initiator_agent.get_new_notifs()
                if "target" in notifs:
                    for msg in notifs["target"]:
                        if msg.startswith(b"DESCS:"):
                            serialized_descs = msg[6:]  # Remove "DESCS:" prefix
                            target_xfer_descs = initiator_agent.deserialize_descs(
                                serialized_descs
                            )
                            logger.info("[initiator] Received descriptors from target")
                            break
                if target_xfer_descs is not None:
                    break
            if target_xfer_descs is None:
                time.sleep(0.01)
    else:
        # TCP mode: retrieve metadata and descriptors from server
        remote_name = retrieve_agent_metadata(
            initiator_agent, "target_meta", role_name="initiator"
        )
        target_xfer_descs = retrieve_descriptors(initiator_agent, "target_descs")

    if not remote_name:
        return

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
        "NIXL_INIT_AGENT",
        [(initiator_addr, buf_size, 0), (initiator_addr2, buf_size, 0)],
        "DRAM",
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
    parser = argparse.ArgumentParser(description="NIXL Two-Process Example")
    parser.add_argument(
        "--use-etcd",
        action="store_true",
        help="Use NIXL's built-in etcd for metadata exchange (requires NIXL built with etcd support and NIXL_ETCD_ENDPOINTS env var)",
    )
    args = parser.parse_args()

    # Set environment variable so child processes can see it
    if args.use_etcd:
        os.environ["NIXL_USE_ETCD"] = "1"

    logger.info("Using NIXL Plugins from:\n%s", os.environ["NIXL_PLUGIN_DIR"])

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
            sys.exit(1)

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
        # Start TCP metadata server
        logger.info("[main] Starting TCP metadata server...")
        try:
            start_server(9998)
            time.sleep(0.2)
        except OSError:
            pass  # Server may already be running
        clear_metadata("127.0.0.1", 9998)

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
        logger.error(
            f"[main] ✗ Process error - Target: {target_proc.exitcode}, Initiator: {initiator_proc.exitcode}"
        )
