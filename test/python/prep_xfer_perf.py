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

import time

import numpy as np

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config


def init_agent():
    agent_config = nixl_agent_config(backends=["UCX"])
    agent = nixl_agent("agent", agent_config)
    return agent


def prep_handles(agent: nixl_agent, xfer_dlist, reg_dlist, indices):
    start = time.perf_counter()
    xfer_dlist_trim = reg_dlist.trim()
    elapsed = time.perf_counter() - start
    assert xfer_dlist_trim.descCount() == xfer_dlist.descCount()
    print(f"Trim nixlRegDList:\t{elapsed:.4f} sec")

    start = time.perf_counter()
    assert agent.register_memory(reg_dlist) is not None
    elapsed = time.perf_counter() - start
    print(f"register_memory:\t{elapsed:.4f} sec")

    start = time.perf_counter()
    local_prep_handle = agent.prep_xfer_dlist(
        "NIXL_INIT_AGENT", xfer_dlist, "DRAM", False
    )
    elapsed = time.perf_counter() - start
    assert local_prep_handle
    print(f"prep_xfer_dlist INIT:\t{elapsed:.4f} sec")

    start = time.perf_counter()
    remote_prep_handle = agent.prep_xfer_dlist("agent", xfer_dlist, "DRAM", False)
    elapsed = time.perf_counter() - start
    assert remote_prep_handle
    print(f"prep_xfer_dlist SELF:\t{elapsed:.4f} sec")

    start = time.perf_counter()
    xfer_handle = agent.make_prepped_xfer(
        "WRITE", local_prep_handle, indices, remote_prep_handle, indices, b"UUID2"
    )
    elapsed = time.perf_counter() - start
    assert xfer_handle
    print(f"make_prepped_xfer:\t{elapsed:.4f} sec")

    return local_prep_handle, remote_prep_handle, xfer_handle


def perf_test_list(num_descs: int, addr_base: int, length: int):
    print("-" * 40)
    print("Starting list test...")
    print("-" * 40)
    agent = init_agent()
    descs_list = [(addr_base + i * length, length, 0) for i in range(num_descs)]
    indices = list(range(num_descs))

    start = time.perf_counter()
    xfer_dlist = agent.get_xfer_descs(descs_list, "DRAM", False)
    elapsed = time.perf_counter() - start
    assert xfer_dlist.descCount() == num_descs
    print(f"get_xfer_descs:\t\t{elapsed:.4f} sec")

    blob_descs_list = [
        (addr_base + i * length, length, 0, b"") for i in range(num_descs)
    ]
    start = time.perf_counter()
    reg_dlist = agent.get_reg_descs(blob_descs_list, "DRAM", False)
    elapsed = time.perf_counter() - start
    assert reg_dlist.descCount() == num_descs
    print(f"get_reg_descs:\t\t{elapsed:.4f} sec")

    local_prep_handle, remote_prep_handle, xfer_handle = prep_handles(
        agent, xfer_dlist, reg_dlist, indices
    )

    del agent


def perf_test_array(num_descs: int, addr_base: int, length: int):
    print("-" * 40)
    print("Starting array test...")
    print("-" * 40)
    agent = init_agent()
    descs_np = np.zeros((num_descs, 3), dtype=np.uint64)
    indices = np.arange(num_descs)
    descs_np[:, 0] = addr_base + indices * length
    descs_np[:, 1] = length
    descs_np[:, 2] = 0

    start = time.perf_counter()
    xfer_dlist = agent.get_xfer_descs(descs_np, "DRAM", False)
    elapsed = time.perf_counter() - start
    assert xfer_dlist.descCount() == num_descs
    print(f"get_xfer_descs:\t\t{elapsed:.4f} sec")

    start = time.perf_counter()
    reg_dlist = agent.get_reg_descs(descs_np, "DRAM", False)
    elapsed = time.perf_counter() - start
    assert reg_dlist.descCount() == num_descs
    print(f"get_reg_descs:\t\t{elapsed:.4f} sec")

    local_prep_handle, remote_prep_handle, xfer_handle = prep_handles(
        agent, xfer_dlist, reg_dlist, indices
    )

    del agent


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["list", "array"],
        help="Test mode: list or array based descriptors",
    )
    args = parser.parse_args()

    print("Using NIXL Plugins from:")
    print(os.environ["NIXL_PLUGIN_DIR"])

    # Example using nixl_agent_config
    agent = init_agent()

    num_descs = 2**20
    length = 64
    addr_base = nixl_utils.malloc_passthru(num_descs * length)
    print(f"Performance test: Creating nixlXferDList with {num_descs} descriptors")

    if args.mode == "list":
        perf_test_list(num_descs, addr_base, length)
    elif args.mode == "array":
        perf_test_array(num_descs, addr_base, length)
