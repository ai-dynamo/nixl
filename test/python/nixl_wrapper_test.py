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

import torch

import nixl._utils as nixl_utils
from nixl._api import nixl_agent

if __name__ == "__main__":
    buf_size = 256
    # Allocate memory and register with NIXL
    nixl_agent1 = nixl_agent("target", None)

    plugin_list = nixl_agent1.get_plugin_list("UCX")
    assert "UCX" in plugin_list
    print(plugin_list)

    print(agent1.get_backend_mem_types("UCX"))
    print(agent1.get_backend_params("UCX"))

    nixl_agent1.create_backend("UCX")

    print(nixl_agent1.get_backend_params("UCX"))

    addr1 = nixl_utils.malloc_passthru(buf_size * 2)
    addr2 = addr1 + buf_size

    agent1_addrs = [(addr1, buf_size, 0), (addr2, buf_size, 0)]
    agent1_strings = [(addr1, buf_size, 0, "a"), (addr2, buf_size, 0, "b")]

    agent1_reg_descs = nixl_agent1.get_reg_descs(agent1_strings, "DRAM", True)
    agent1_xfer_descs = nixl_agent1.get_xfer_descs(agent1_addrs, "DRAM", True)

    # Just for tensor test
    tensors = [torch.zeros(10, dtype=torch.float32) for _ in range(2)]
    agent1_tensor_reg_descs = nixl_agent1.get_reg_descs(tensors)
    agent1_tensor_xfer_descs = nixl_agent1.get_xfer_descs(tensors)

    assert nixl_agent1.register_memory(agent1_reg_descs) is not None

    nixl_agent2 = nixl_agent("initiator", None)
    addr3 = nixl_utils.malloc_passthru(buf_size * 2)
    addr4 = addr3 + buf_size

    agent2_addrs = [(addr3, buf_size, 0), (addr4, buf_size, 0)]
    agent2_strings = [(addr3, buf_size, 0, "a"), (addr4, buf_size, 0, "b")]

    agent2_reg_descs = nixl_agent1.get_reg_descs(agent2_strings, "DRAM", True)
    agent2_xfer_descs = nixl_agent1.get_xfer_descs(agent2_addrs, "DRAM", True)

    agent2_descs = nixl_agent2.register_memory(agent2_reg_descs, True)
    assert agent2_descs is not None

    # Exchange metadata
    meta = nixl_agent1.get_agent_metadata()
    remote_name = nixl_agent2.add_remote_agent(meta)
    print("Loaded name from metadata:", remote_name)

    serdes = nixl_agent1.get_serialized_descs(agent1_reg_descs)
    src_descs_recvd = nixl_agent2.deserialize_descs(serdes)
    assert src_descs_recvd == agent1_reg_descs

    # initialize transfer mode
    xfer_handle_1 = nixl_agent2.initialize_xfer(
        agent2_xfer_descs, agent1_xfer_descs, remote_name, "UUID1", "READ"
    )
    if not xfer_handle_1:
        print("Creating transfer failed.")
        exit()

    state = nixl_agent2.transfer(xfer_handle_1)
    assert state != "ERR"

    target_done = False
    init_done = False

    while (not init_done) or (not target_done):
        if not init_done:
            state = nixl_agent2.check_xfer_state(xfer_handle_1)
            if state == "ERR":
                print("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                init_done = True
                print("Initiator done")

        if not target_done:
            if nixl_agent1.check_remote_xfer_done("initiator", "UUID1"):
                target_done = True
                print("Target done")

    # prep transfer mode
    local_prep_handle = nixl_agent2.prep_xfer_side(
        "", [(addr3, buf_size, 0), (addr4, buf_size, 0)], "DRAM", True
    )
    remote_prep_handle = nixl_agent2.prep_xfer_side(
        remote_name, agent1_xfer_descs, "DRAM"
    )

    assert local_prep_handle != 0
    assert remote_prep_handle != 0

    xfer_handle_2 = nixl_agent2.make_prepped_xfer(
        local_prep_handle, [0, 1], remote_prep_handle, [1, 0], "UUID2", "WRITE"
    )
    if not local_prep_handle or not remote_prep_handle:
        print("Preparing transfer side handles failed.")
        exit()

    if not xfer_handle_2:
        print("Make prepped transfer failed.")
        exit()

    state = nixl_agent2.transfer(xfer_handle_2)
    assert state != "ERR"

    target_done = False
    init_done = False

    while (not init_done) or (not target_done):
        if not init_done:
            state = nixl_agent2.check_xfer_state(xfer_handle_2)
            if state == "ERR":
                print("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                init_done = True
                print("Initiator done")

        if not target_done:
            if nixl_agent1.check_remote_xfer_done("initiator", "UUID2"):
                target_done = True
                print("Target done")

    nixl_agent2.abort_xfer(xfer_handle_1)
    nixl_agent2.abort_xfer(xfer_handle_2)
    nixl_agent2.delete_xfer_side(local_prep_handle)
    nixl_agent2.delete_xfer_side(remote_prep_handle)
    nixl_agent2.remove_remote_agent("target")
    nixl_agent1.deregister_memory(agent1_reg_descs)
    nixl_agent2.deregister_memory(agent2_reg_descs)

    nixl_utils.free_passthru(addr1)
    nixl_utils.free_passthru(addr3)

    print("Test Complete.")
