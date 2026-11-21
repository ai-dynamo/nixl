/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Scality AI Connector helpers that need no RDMA device: the DC RDMA descriptor
// wire format and the cufile.json readers.

#include <gtest/gtest.h>

#include "rest_accel/scality_ai_connector/cufile_nics.h"
#include "rest_accel/scality_ai_connector/dc_descriptor.h"

namespace gtest::obj {

// Golden DC RDMA descriptor: ADDR:SIZE:RKEY:LID:DCTN:1:GID.
TEST(scalityUtilsTest, DcDescriptorMatchesWireFormat) {
    const uint8_t gid[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0x0a, 0x0a, 0x30, 0xd0};
    const std::string d =
        formatDcDescriptor(0x7f1234567890ULL, 0x100000U, 0xaabbccddU, 0, 0x42U, gid);
    EXPECT_EQ(d,
              "00007f1234567890:00100000:aabbccdd:0000:000042:1:"
              "00000000000000000000ffff0a0a30d0");
}

TEST(scalityUtilsTest, DcDescriptorFieldsAreZeroPadded) {
    const uint8_t gid[16] = {};
    const std::string d = formatDcDescriptor(0, 0, 0, 0, 0, gid);
    EXPECT_EQ(d,
              "0000000000000000:00000000:00000000:0000:000000:1:"
              "00000000000000000000000000000000");
}

// rdma_dev_addr_list, tolerating // comments and ignoring commented-out copies of
// the key in other sections.
TEST(scalityUtilsTest, ParseRdmaDevAddrList) {
    const std::string json = R"({
        "properties": {
            // client-side rdma list
            "rdma_dev_addr_list": [ "mlx5_1", "mlx5_2", "mlx5_7", "mlx5_8" ],
            "rdma_load_balancing_policy": "RoundRobin"
        },
        "fs": {
            "lustre": {
                //"rdma_dev_addr_list" : ["10.0.0.1"]
            }
        }
    })";
    const std::vector<std::string> nics = parseRdmaDevAddrList(json);
    ASSERT_EQ(nics.size(), 4u);
    EXPECT_EQ(nics[0], "mlx5_1");
    EXPECT_EQ(nics[3], "mlx5_8");
}

TEST(scalityUtilsTest, ParseRdmaDevAddrListIpsAndMissing) {
    EXPECT_EQ(parseRdmaDevAddrList("{ \"foo\": 1 }").size(), 0u);
    const std::vector<std::string> nics =
        parseRdmaDevAddrList("\"rdma_dev_addr_list\": [\"10.10.40.208\", \"10.10.48.208\"]");
    ASSERT_EQ(nics.size(), 2u);
    EXPECT_EQ(nics[0], "10.10.40.208");
    EXPECT_EQ(nics[1], "10.10.48.208");
}

TEST(scalityUtilsTest, ParseRdmaDcKey) {
    EXPECT_EQ(parseRdmaDcKey("\"rdma_dc_key\": \"0x12345678\","), "0x12345678");
    // Commented out, as in the default cufile.json: treated as absent.
    EXPECT_EQ(parseRdmaDcKey("    //\"rdma_dc_key\": \"0xffeeddcc\""), "");
    EXPECT_EQ(parseRdmaDcKey("{ \"properties\": {} }"), "");
}

} // namespace gtest::obj
