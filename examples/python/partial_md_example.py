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

import argparse
import os

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
from nixl._bindings import nixlNotFoundError


def exchange_target_metadata(
    target_agent,
    init_agent,
    ip_addr,
    init_port,
    etcd_endpoints,
    label,
    target_reg_descs,
):
    """
    Exchange metadata from target to initiator
    """
    if etcd_endpoints:
        # With ETCD, target sends MD to server and initiator fetches it
        target_agent.send_partial_agent_metadata(
            target_reg_descs, True, ["UCX"], label=label
        )
        init_agent.fetch_remote_metadata(target_agent.name, label=label)
    else:
        # With socket exchange, target sends MD directly to initiator
        target_agent.send_partial_agent_metadata(
            target_reg_descs, True, ["UCX"], ip_addr, init_port
        )


def invalidate_target_metadata(
    target_agent,
    ip_addr,
    init_port,
    etcd_endpoints,
):
    """
    Invalidate metadata from target to initiator
    """
    if etcd_endpoints:
        target_agent.invalidate_local_metadata()
    else:
        target_agent.invalidate_local_metadata(ip_addr, init_port)


if __name__ == "__main__":
    buf_size = 256
    # Allocate memory and register with NIXL

    parser = argparse.ArgumentParser(description="NIXL Partial Metadata Example")
    parser.add_argument(
        "--etcd",
        action="store_true",
        help="Use ETCD for metadata exchange. Must set NIXL_ETCD_ENDPOINTS environment variable.",
    )
    args = parser.parse_args()

    print("Using NIXL Plugins from:")
    print(os.environ["NIXL_PLUGIN_DIR"])

    if args.etcd:
        etcd_endpoints = os.getenv("NIXL_ETCD_ENDPOINTS", "")
        if etcd_endpoints:
            print("NIXL_ETCD_ENDPOINTS is set, using endpoints: ", etcd_endpoints)
        else:
            raise ValueError(
                "NIXL_ETCD_ENDPOINTS is not set, but --etcd flag is provided"
            )
    else:
        etcd_endpoints = ""
        del os.environ["NIXL_ETCD_ENDPOINTS"]
        print("NIXL_ETCD_ENDPOINTS is not set, using socket exchange")

    # Needed for socket exchange
    ip_addr = "127.0.0.1"
    target_port = 5555
    init_port = 7777
    # Example using nixl_agent_config
    agent_config1 = nixl_agent_config(True, True, target_port)

    target_agent = nixl_agent("target", agent_config1)

    malloc_addrs = []

    target_strs1 = []
    for _ in range(10):
        addr1 = nixl_utils.malloc_passthru(buf_size)
        target_strs1.append((addr1, buf_size, 0, "test"))
        malloc_addrs.append(addr1)

    target_strs2 = []
    for _ in range(10):
        addr1 = nixl_utils.malloc_passthru(buf_size)
        target_strs2.append((addr1, buf_size, 0, "test"))
        malloc_addrs.append(addr1)

    target_reg_descs1 = target_agent.get_reg_descs(
        target_strs1, "DRAM", is_sorted=False
    )
    target_reg_descs2 = target_agent.get_reg_descs(
        target_strs2, "DRAM", is_sorted=False
    )
    target_xfer_descs1 = target_reg_descs1.trim()
    target_xfer_descs2 = target_reg_descs2.trim()

    assert target_agent.register_memory(target_reg_descs1) is not None
    assert target_agent.register_memory(target_reg_descs2) is not None

    # Default port for initiator
    agent_config2 = nixl_agent_config(True, True, init_port)
    init_agent = nixl_agent("initiator", agent_config2)

    init_strs = []
    for _ in range(10):
        addr1 = nixl_utils.malloc_passthru(buf_size)
        init_strs.append((addr1, buf_size, 0, "test"))
        malloc_addrs.append(addr1)

    init_reg_descs = init_agent.get_reg_descs(init_strs, "DRAM", is_sorted=False)
    init_xfer_descs = init_reg_descs.trim()

    assert init_agent.register_memory(init_reg_descs) is not None

    exchange_target_metadata(
        target_agent,
        init_agent,
        ip_addr,
        init_port,
        etcd_endpoints,
        "label_1",
        target_reg_descs1,
    )

    # Wait for metadata to be loaded
    ready = False

    while not ready:
        ready = init_agent.check_remote_metadata("target", target_xfer_descs1)

    xfer_handle_1 = init_agent.initialize_xfer(
        "READ", init_xfer_descs, target_xfer_descs1, "target", b"UUID1"
    )

    state = init_agent.transfer(xfer_handle_1)
    assert state != "ERR"

    target_done = False
    init_done = False

    while (not init_done) or (not target_done):
        if not init_done:
            state = init_agent.check_xfer_state(xfer_handle_1)
            if state == "ERR":
                print("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                init_done = True
                print("Initiator done")

        if not target_done:
            if target_agent.check_remote_xfer_done("initiator", b"UUID1"):
                target_done = True
                print("Target done")

    # Second set of descs was not sent, should fail
    try:
        xfer_handle_2 = init_agent.initialize_xfer(
            "READ", init_xfer_descs, target_xfer_descs2, "target", b"UUID1"
        )
    except nixlNotFoundError:
        print("Correct exception")
    else:
        print("Incorrect success")
        os.abort()

    # Now send rest of descs
    exchange_target_metadata(
        target_agent,
        init_agent,
        ip_addr,
        init_port,
        etcd_endpoints,
        "label_2",
        target_reg_descs2,
    )

    # Wait for metadata to be loaded
    ready = False
    xfer_handle_2 = 0
    while not ready:
        try:
            # initialize transfer mode
            xfer_handle_2 = init_agent.initialize_xfer(
                "READ", init_xfer_descs, target_xfer_descs2, "target", b"UUID1"
            )
        except nixlNotFoundError:
            ready = False
        else:
            ready = True

    state = init_agent.transfer(xfer_handle_2)
    assert state != "ERR"

    target_done = False
    init_done = False

    while (not init_done) or (not target_done):
        if not init_done:
            state = init_agent.check_xfer_state(xfer_handle_2)
            if state == "ERR":
                print("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                init_done = True
                print("Initiator done")

        if not target_done:
            if target_agent.check_remote_xfer_done("initiator", b"UUID1"):
                target_done = True
                print("Target done")

    init_agent.release_xfer_handle(xfer_handle_1)
    init_agent.release_xfer_handle(xfer_handle_2)
    init_agent.remove_remote_agent("target")

    target_agent.deregister_memory(target_reg_descs1)
    target_agent.deregister_memory(target_reg_descs2)
    init_agent.deregister_memory(init_reg_descs)

    for addr in malloc_addrs:
        nixl_utils.free_passthru(addr)

    del init_agent
    del target_agent

    print("Test Complete.")
