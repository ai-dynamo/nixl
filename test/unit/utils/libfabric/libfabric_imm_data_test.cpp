/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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

/*
 * Unit test for the immediate data bit layout shared by every libfabric rail.
 *
 * The fields packed into the immediate data are how a receiver learns which peer sent an operation
 * and which transfer it belongs to. A field that silently truncates does not drop the operation, it
 * attributes it to the wrong peer or the wrong transfer, so the encode/decode pair is worth pinning
 * down independently of any rail or connection state.
 */

#include "libfabric/libfabric_common.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define TEST_ASSERT(cond, msg)                                                           \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "FAIL: " << (msg) << " [" << __FILE__ << ":" << __LINE__ << "]" \
                      << std::endl;                                                      \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

namespace {

// The fields must tile the immediate data exactly, with nothing hanging off the top. The header
// asserts this too; repeated here so a layout change that skips the header review still trips a
// test.
static int
testLayoutTilesTheField() {
    TEST_ASSERT(NIXL_MSG_TYPE_SHIFT == 0, "MSG_TYPE must start at bit 0");
    TEST_ASSERT(NIXL_AGENT_INDEX_SHIFT == NIXL_MSG_TYPE_BITS, "AGENT_INDEX must follow MSG_TYPE");
    TEST_ASSERT(NIXL_XFER_ID_SHIFT == NIXL_MSG_TYPE_BITS + NIXL_AGENT_INDEX_BITS,
                "XFER_ID must follow AGENT_INDEX");
    TEST_ASSERT(NIXL_XFER_ID_SHIFT + NIXL_XFER_ID_BITS == NIXL_IMM_DATA_BITS,
                "fields must reach exactly the top of the immediate data");

    TEST_ASSERT(NIXL_MSG_TYPE_MASK == (1U << NIXL_MSG_TYPE_BITS) - 1, "MSG_TYPE mask width");
    TEST_ASSERT(NIXL_AGENT_INDEX_MASK == (1U << NIXL_AGENT_INDEX_BITS) - 1,
                "AGENT_INDEX mask width");
    TEST_ASSERT(NIXL_XFER_ID_MASK == (1U << NIXL_XFER_ID_BITS) - 1, "XFER_ID mask width");

    TEST_ASSERT(NIXL_LIBFABRIC_MAX_AGENTS == (size_t{1} << NIXL_AGENT_INDEX_BITS),
                "the peer cap must be derived from the AGENT_INDEX width");

    std::cout << "PASS: layout tiles the immediate data (agent index = " << NIXL_AGENT_INDEX_BITS
              << " bits, max " << NIXL_LIBFABRIC_MAX_AGENTS << " peers)" << std::endl;
    return 0;
}

// Every message type actually sent on the wire has to survive the MSG_TYPE field, which is the
// narrowest of the three. A type added without checking would decode as some other type.
static int
testMessageTypesFit() {
    const std::vector<std::pair<const char *, uint64_t>> types = {
        {"NOTIFICATION", NIXL_LIBFABRIC_MSG_NOTIFICTION},
        {"TRANSFER", NIXL_LIBFABRIC_MSG_TRANSFER},
        {"HANDSHAKE", NIXL_LIBFABRIC_MSG_HANDSHAKE},
    };

    for (const auto &[name, value] : types) {
        TEST_ASSERT(value <= NIXL_MSG_TYPE_MASK,
                    std::string("message type does not fit MSG_TYPE: ") + name);
        const uint64_t imm = NIXL_MAKE_IMM_DATA(value, 0, 0);
        TEST_ASSERT(NIXL_GET_MSG_TYPE_FROM_IMM(imm) == value,
                    std::string("message type does not round trip: ") + name);
    }

    std::cout << "PASS: all " << types.size() << " message types fit and round trip" << std::endl;
    return 0;
}

// Each field must come back exactly as it went in, whatever the other two hold. Packing bugs
// usually show up as one field bleeding into its neighbour, so the interesting cases are the ones
// where a neighbour is saturated.
static int
testRoundTrip() {
    const uint64_t msg_types[] = {
        NIXL_LIBFABRIC_MSG_NOTIFICTION, NIXL_LIBFABRIC_MSG_TRANSFER, NIXL_LIBFABRIC_MSG_HANDSHAKE};
    // 255/256 straddle the old 8-bit agent index: the widening exists so 256 and up work.
    const uint64_t agent_indices[] = {0, 1, 255, 256, 700, NIXL_LIBFABRIC_MAX_AGENTS - 1};
    const uint64_t xfer_ids[] = {0, 1, 4095, 4096, 65534, NIXL_XFER_ID_MASK};

    size_t cases = 0;
    for (uint64_t msg_type : msg_types) {
        for (uint64_t agent_idx : agent_indices) {
            for (uint64_t xfer_id : xfer_ids) {
                const uint64_t imm = NIXL_MAKE_IMM_DATA(msg_type, agent_idx, xfer_id);

                TEST_ASSERT(imm <= 0xFFFFFFFFULL, "immediate data must stay within 32 bits");
                TEST_ASSERT(NIXL_GET_MSG_TYPE_FROM_IMM(imm) == msg_type,
                            "MSG_TYPE did not round trip");
                TEST_ASSERT(NIXL_GET_AGENT_INDEX_FROM_IMM(imm) == agent_idx,
                            "AGENT_INDEX did not round trip");
                TEST_ASSERT(NIXL_GET_XFER_ID_FROM_IMM(imm) == xfer_id,
                            "XFER_ID did not round trip");
                ++cases;
            }
        }
    }

    std::cout << "PASS: " << cases << " encode/decode round trips" << std::endl;
    return 0;
}

// A peer index at or past the cap cannot be represented. Callers are expected to reject it before
// encoding (createAgentConnection and the handshake both do); this pins down why, by showing that
// the encoder aliases it onto a valid-looking index rather than failing.
static int
testOutOfRangeAgentIndexAliases() {
    const uint64_t over_cap = NIXL_LIBFABRIC_MAX_AGENTS;
    const uint64_t imm = NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER, over_cap, 7);

    TEST_ASSERT(
        NIXL_GET_AGENT_INDEX_FROM_IMM(imm) == 0,
        "an index one past the cap should wrap to 0, which is why callers must range-check");
    TEST_ASSERT(NIXL_GET_XFER_ID_FROM_IMM(imm) == 7,
                "an over-range agent index must not corrupt XFER_ID");
    TEST_ASSERT(NIXL_GET_MSG_TYPE_FROM_IMM(imm) == NIXL_LIBFABRIC_MSG_TRANSFER,
                "an over-range agent index must not corrupt MSG_TYPE");

    std::cout << "PASS: an out-of-range agent index aliases silently (callers must range-check)"
              << std::endl;
    return 0;
}

} // namespace

int
main() {
    if (testLayoutTilesTheField() != 0) {
        return 1;
    }
    if (testMessageTypesFit() != 0) {
        return 1;
    }
    if (testRoundTrip() != 0) {
        return 1;
    }
    if (testOutOfRangeAgentIndexAliases() != 0) {
        return 1;
    }

    std::cout << "All immediate data layout tests passed" << std::endl;
    return 0;
}
