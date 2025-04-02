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

import pickle

import nixl._bindings as nixl
import nixl._utils as nixl_utils

# These should automatically be run by pytest because of function names


def test_list():
    descs = [(1000, 105, 0), (2000, 30, 0), (1010, 20, 0)]
    test_list = nixl.nixlXferDList(nixl.DRAM_SEG, descs, False)

    assert test_list.descCount() == 3

    test_list.print()

    pickled_list = pickle.dumps(test_list)

    print(pickled_list)

    unpickled_list = pickle.loads(pickled_list)

    assert unpickled_list == test_list

    assert test_list.getType() == nixl.DRAM_SEG

    print(test_list.descCount())
    assert test_list.descCount() == 3

    test_list.remDesc(1)
    assert test_list.descCount() == 2

    assert test_list[0] == descs[0]

    test_list.clear()

    assert test_list.isEmpty()

    test_list.addDesc((2000, 100, 0))


def test_agent():
    name1 = "Agent1"
    name2 = "Agent2"

    devices = nixl.nixlAgentConfig(False)

    agent1 = nixl.nixlAgent(name1, devices)
    agent2 = nixl.nixlAgent(name2, devices)

    ucx1 = agent1.createBackend("UCX", {})
    ucx2 = agent2.createBackend("UCX", {})

    size = 256
    addr1 = nixl_utils.malloc_passthru(size)
    addr2 = nixl_utils.malloc_passthru(size)

    nixl_utils.ba_buf(addr1, size)

    reg_list1 = nixl.nixlRegDList(nixl.DRAM_SEG, False)
    reg_list1.addDesc((addr1, size, 0, "dead"))

    reg_list2 = nixl.nixlRegDList(nixl.DRAM_SEG, False)
    reg_list2.addDesc((addr2, size, 0, "dead"))

    ret = agent1.registerMem(reg_list1, [ucx1])
    assert ret == nixl.NIXL_SUCCESS

    ret = agent2.registerMem(reg_list2, [ucx2])
    assert ret == nixl.NIXL_SUCCESS

    meta1 = agent1.getLocalMD()
    meta2 = agent2.getLocalMD()

    print("Agent1 MD: ")
    print(meta1)
    print("Agent2 MD: ")
    print(meta2)

    ret_name = agent1.loadRemoteMD(meta2)
    assert ret_name.decode(encoding="UTF-8") == name2
    ret_name = agent2.loadRemoteMD(meta1)
    assert ret_name.decode(encoding="UTF-8") == name1

    offset = 8
    req_size = 8

    src_list = nixl.nixlXferDList(nixl.DRAM_SEG, False)
    src_list.addDesc((addr1 + offset, req_size, 0))

    dst_list = nixl.nixlXferDList(nixl.DRAM_SEG, False)
    dst_list.addDesc((addr2 + offset, req_size, 0))

    print("Transfer from " + str(addr1 + offset) + " to " + str(addr2 + offset))

    noti_str = "n\0tification"
    print(noti_str)

    print(src_list)
    print(dst_list)

    handle = agent1.createXferReq(nixl.NIXL_WRITE, src_list, dst_list, name2, noti_str)
    assert handle != 0

    print(handle)

    status = agent1.postXferReq(handle)
    assert status == nixl.NIXL_SUCCESS or status == nixl.NIXL_IN_PROG

    print("Transfer posted")

    notifMap = {}

    while status != nixl.NIXL_SUCCESS or len(notifMap) == 0:
        if status != nixl.NIXL_SUCCESS:
            status = agent1.getXferStatus(handle)

        if len(notifMap) == 0:
            notifMap = agent2.getNotifs(notifMap)

        assert status == nixl.NIXL_SUCCESS or status == nixl.NIXL_IN_PROG

    nixl_utils.verify_transfer(addr1 + offset, addr2 + offset, req_size)
    assert len(notifMap[name1]) == 1
    print(notifMap[name1][0])
    assert notifMap[name1][0] == noti_str.encode()

    print("Transfer verified")

    agent1.releaseXferReq(handle)

    ret = agent1.deregisterMem(reg_list1, [ucx1])
    assert ret == nixl.NIXL_SUCCESS

    ret = agent2.deregisterMem(reg_list2, [ucx2])
    assert ret == nixl.NIXL_SUCCESS

    # Only initiator should call invalidate
    agent1.invalidateRemoteMD(name2)
    # agent2.invalidateRemoteMD(name1)

    nixl_utils.free_passthru(addr1)
    nixl_utils.free_passthru(addr2)
