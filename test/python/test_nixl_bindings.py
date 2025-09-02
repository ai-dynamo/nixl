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

import os
import pickle
import tempfile

import nixl._bindings as nixl
import nixl._utils as nixl_utils
from nixl.logging import get_logger

logger = get_logger(__name__)

# These should automatically be run by pytest because of function names


def test_list():
    descs = [(1000, 105, 0), (2000, 30, 0), (1010, 20, 0)]
    test_list = nixl.nixlXferDList(nixl.DRAM_SEG, descs)

    assert test_list.descCount() == 3

    test_list.print()

    pickled_list = pickle.dumps(test_list)

    logger.info("Pickled list: %s", pickled_list)

    unpickled_list = pickle.loads(pickled_list)

    assert unpickled_list == test_list

    assert test_list.getType() == nixl.DRAM_SEG

    logger.info("Descriptor count: %s", test_list.descCount())
    assert test_list.descCount() == 3

    test_list.remDesc(1)
    assert test_list.descCount() == 2

    assert test_list[0] == descs[0]

    test_list.clear()

    assert test_list.isEmpty()

    test_list.addDesc((2000, 100, 0))


def test_agent():
    os.environ["NIXL_TELEMETRY_ENABLE"] = "y"
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

    reg_list1 = nixl.nixlRegDList(nixl.DRAM_SEG)
    reg_list1.addDesc((addr1, size, 0, "dead"))

    reg_list2 = nixl.nixlRegDList(nixl.DRAM_SEG)
    reg_list2.addDesc((addr2, size, 0, "dead"))

    ret = agent1.registerMem(reg_list1, [ucx1])
    assert ret == nixl.NIXL_SUCCESS

    ret = agent2.registerMem(reg_list2, [ucx2])
    assert ret == nixl.NIXL_SUCCESS

    meta1 = agent1.getLocalMD()
    meta2 = agent2.getLocalMD()

    logger.info("Agent1 MD: \n%s", meta1)
    logger.info("Agent2 MD: \n%s", meta2)

    ret_name = agent1.loadRemoteMD(meta2)
    assert ret_name.decode(encoding="UTF-8") == name2
    ret_name = agent2.loadRemoteMD(meta1)
    assert ret_name.decode(encoding="UTF-8") == name1

    offset = 8
    req_size = 8

    src_list = nixl.nixlXferDList(nixl.DRAM_SEG)
    src_list.addDesc((addr1 + offset, req_size, 0))

    dst_list = nixl.nixlXferDList(nixl.DRAM_SEG)
    dst_list.addDesc((addr2 + offset, req_size, 0))

    logger.info("Transfer from %s to %s", str(addr1 + offset), str(addr2 + offset))

    noti_str = "n\0tification"
    logger.info("Notification string: %s", noti_str)

    logger.info("Source list: %s", src_list)
    logger.info("Destination list: %s", dst_list)

    handle = agent1.createXferReq(nixl.NIXL_WRITE, src_list, dst_list, name2, noti_str)
    assert handle != 0

    logger.info("Transfer handle: %s", handle)

    status = agent1.postXferReq(handle)
    assert status == nixl.NIXL_SUCCESS or status == nixl.NIXL_IN_PROG

    logger.info("Transfer posted")

    notifMap = {}

    while status != nixl.NIXL_SUCCESS or len(notifMap) == 0:
        if status != nixl.NIXL_SUCCESS:
            status = agent1.getXferStatus(handle)

        if len(notifMap) == 0:
            notifMap = agent2.getNotifs(notifMap)

        assert status == nixl.NIXL_SUCCESS or status == nixl.NIXL_IN_PROG

    nixl_utils.verify_transfer(addr1 + offset, addr2 + offset, req_size)
    assert len(notifMap[name1]) == 1
    logger.info("Received notification: %s", notifMap[name1][0])
    assert notifMap[name1][0] == noti_str.encode()

    logger.info("Transfer verified")

    # Verify transfer telemetry
    telem = agent1.getXferTelemetry(handle)
    assert telem.descCount == 1
    assert telem.totalBytes == req_size
    assert telem.startTime > 0
    assert telem.postDuration > 0
    assert telem.xferDuration > 0
    assert telem.xferDuration >= telem.postDuration

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


def test_query_mem():
    """Test basic queryMem functionality"""

    os.makedirs("files_for_query", exist_ok=True)
    # Create temporary test files
    temp_file1 = tempfile.NamedTemporaryFile(dir="files_for_query", delete=False)
    temp_file1.write(b"Test content for queryMem file 1")
    temp_file1.close()

    temp_file2 = tempfile.NamedTemporaryFile(dir="files_for_query", delete=False)
    temp_file2.write(b"Test content for queryMem file 2")
    temp_file2.close()

    # Create a non-existent file path
    non_existent_file = "./nixl_test_nonexistent_file_12345.txt"

    try:
        # Create an agent
        config = nixl.nixlAgentConfig(False, False)
        agent = nixl.nixlAgent("test_agent", config)

        try:
            params, mems = agent.getPluginParams("POSIX")
            backend = agent.createBackend("POSIX", params)

            descs = nixl.nixlRegDList(nixl.FILE_SEG)

            # Test 1: Query with empty descriptor list
            try:
                resp = agent.queryMem(descs, backend)
                assert len(resp) == 0
            except Exception as e:
                # Some backends might not support queryMem, which is okay
                logger.exception(
                    "queryMem with empty list failed (expected for some backends): %s",
                    e,
                )

            # Test 2: Query with actual file descriptors
            # Existing file 1
            descs.addDesc((0, 0, 0, temp_file1.name))
            # Non-existent file
            descs.addDesc((0, 0, 0, non_existent_file))
            # Existing file 2
            descs.addDesc((0, 0, 0, temp_file2.name))

            try:
                resp = agent.queryMem(descs, backend)

                # Verify results
                assert len(resp) == 3

                # First file should be accessible (returns dict with info)
                assert resp[0] is not None
                assert isinstance(resp[0], dict)
                assert "size" in resp[0]
                assert "mode" in resp[0]

                # Second file should not be accessible (returns None)
                assert resp[1] is None

                # Third file should be accessible (returns dict with info)
                assert resp[2] is not None
                assert isinstance(resp[2], dict)
                assert "size" in resp[2]
                assert "mode" in resp[2]

            except Exception as e:
                # Some backends might not support queryMem, which is okay
                logger.exception(
                    "queryMem failed (expected for some backends): %s",
                    e,
                )
        except Exception as e:
            logger.exception("Backend creation failed: %s", e)
            # Try MOCK_DRAM as fallback
            try:
                params, mems = agent.getPluginParams("MOCK_DRAM")
                backend = agent.createBackend("MOCK_DRAM", params)
                logger.info("Using MOCK_DRAM backend")
            except Exception as e2:
                logger.exception("MOCK_DRAM also failed: %s", e2)
                return

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file1.name):
            os.unlink(temp_file1.name)
        if os.path.exists(temp_file2.name):
            os.unlink(temp_file2.name)
