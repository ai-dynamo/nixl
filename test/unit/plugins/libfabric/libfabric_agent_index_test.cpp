/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Amazon.com, Inc. and affiliates.
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
 * Unit test for agent index allocation in nixlLibfabricEngine::createAgentConnection().
 *
 * The index this function hands a peer is that peer's identity on the wire: it travels in the
 * AGENT_INDEX field of the immediate data and is what the receiver uses to attribute an incoming
 * transfer or notification. Two peers sharing an index is silent misattribution, not a failure, so
 * the properties worth pinning down are that indices are unique, that they survive the immediate
 * data field, and that the engine refuses a peer it cannot name rather than aliasing it.
 *
 * The engine runs against the mocked fabric in libfabric_mock_stubs.h, so no hardware is needed.
 * Peer counts here deliberately exceed 512, which the previous 8-bit AGENT_INDEX field could not
 * represent.
 */

#include "libfabric_backend.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

// Two rails keep the mocked fabric cheap; nothing here depends on the rail count.
static const size_t NUM_FAKE_RAILS = 2;

// --- Mocked libfabric device discovery ---

extern "C" int
__wrap_numa_max_node() {
    return 1;
}

extern "C" int
__wrap_numa_num_configured_nodes() {
    return 2;
}

extern "C" int
__wrap_fi_getinfo(uint32_t /*version*/,
                  const char * /*node*/,
                  const char * /*service*/,
                  uint64_t /*flags*/,
                  const struct fi_info * /*hints*/,
                  struct fi_info **info) {
    fi_info *head = nullptr;
    fi_info *prev = nullptr;
    for (size_t i = 0; i < NUM_FAKE_RAILS; ++i) {
        fi_info *fi = malloc_zero<fi_info>();

        fi->domain_attr = malloc_zero<fi_domain_attr>();
        std::string name = "efa_" + std::to_string(i);
        fi->domain_attr->name = strdup(name.c_str());
        // Real EFA reports 4 bytes; rail init requires enough to hold the immediate data fields.
        fi->domain_attr->cq_data_size = NIXL_IMM_DATA_BITS / 8;

        fi->fabric_attr = malloc_zero<fi_fabric_attr>();
        fi->fabric_attr->prov_name = strdup("efa");
        fi->fabric_attr->name = strdup("efa");

        fi->ep_attr = malloc_zero<fi_ep_attr>();
        fi->ep_attr->type = FI_EP_RDM;

        fi->nic = malloc_zero<fid_nic>();
        fi->nic->bus_attr = malloc_zero<fi_bus_attr>();
        fi->nic->bus_attr->bus_type = FI_BUS_PCI;
        fi->nic->bus_attr->attr.pci.domain_id = 0;
        fi->nic->bus_attr->attr.pci.bus_id = static_cast<uint8_t>(i);
        fi->nic->bus_attr->attr.pci.device_id = 0;
        fi->nic->bus_attr->attr.pci.function_id = 0;

        fi->nic->link_attr = malloc_zero<fi_link_attr>();
        fi->nic->link_attr->speed = 100ull * NIXL_LIBFABRIC_GIGA;

        if (prev) {
            prev->next = fi;
        } else {
            head = fi;
        }
        prev = fi;
    }
    *info = head;
    return 0;
}

extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr * /*attr*/, struct fid_fabric **fabric, void * /*context*/) {
    *fabric = mock_fabric_create();
    return 0;
}

// --- Test helpers ---

#define TEST_ASSERT(cond, msg)                                                           \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "FAIL: " << (msg) << " [" << __FILE__ << ":" << __LINE__ << "]" \
                      << std::endl;                                                      \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

static const char *LOCAL_AGENT = "agent_local";

/**
 * @brief Drives nixlLibfabricEngine's private agent index allocation.
 *
 * Declared a friend of nixlLibfabricEngine so the tests can call createAgentConnection() and read
 * back the resulting table without going through loadRemoteConnInfo(), which would additionally
 * require a mocked handshake send path.
 */
class nixlLibfabricAgentIndexTest {
public:
    static int
    run();

private:
    // Larger than 512 so the assigned indices run past what the previous 8-bit AGENT_INDEX field
    // could hold.
    static constexpr size_t PEER_COUNT = 600;

    // The endpoint bytes are only handed to fi_av_insert(), which the mock ignores, so any
    // distinguishable content will do.
    static std::vector<std::array<char, LF_EP_NAME_MAX_LEN>>
    makeEndpoints(const std::string &peer) {
        std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> endpoints(NUM_FAKE_RAILS);
        for (size_t rail_id = 0; rail_id < endpoints.size(); ++rail_id) {
            endpoints[rail_id].fill(0);
            const std::string ep_name = peer + ":" + std::to_string(rail_id);
            std::memcpy(endpoints[rail_id].data(),
                        ep_name.data(),
                        std::min(ep_name.size(), endpoints[rail_id].size() - 1));
        }
        return endpoints;
    }

    static nixl_status_t
    addPeer(nixlLibfabricEngine &engine, const std::string &peer) {
        return engine.createAgentConnection(peer, makeEndpoints(peer));
    }

    static size_t
    tableSize(const nixlLibfabricEngine &engine) {
        return engine.agent_names_.size();
    }

    static const std::string &
    nameAt(const nixlLibfabricEngine &engine, size_t index) {
        return engine.agent_names_[index];
    }

    static bool
    isConnected(const nixlLibfabricEngine &engine, const std::string &peer) {
        return engine.connections_.find(peer) != engine.connections_.end();
    }

    static size_t
    agentIndexOf(const nixlLibfabricEngine &engine, const std::string &peer) {
        return engine.connections_.at(peer)->agent_index_;
    }

    static std::string
    peerName(size_t i) {
        return "peer_" + std::to_string(i);
    }

    static int
    testSelfTakesIndexZero(nixlLibfabricEngine &engine);
    static int
    testIndicesAreUniqueAndSurviveTheWire(nixlLibfabricEngine &engine);
    static int
    testKnownPeerReusesItsIndex(nixlLibfabricEngine &engine);
    static int
    testDisconnectRetiresTheSlotWithoutRenumbering(nixlLibfabricEngine &engine);
    static int
    testPeerCapIsRefusedNotAliased(nixlLibfabricEngine &engine);
};

// The engine's own self-connection is created in the constructor and must land at index 0, because
// every later index is allocated relative to it.
int
nixlLibfabricAgentIndexTest::testSelfTakesIndexZero(nixlLibfabricEngine &engine) {
    NIXL_INFO << "  testSelfTakesIndexZero";

    TEST_ASSERT(tableSize(engine) == 1, "only the self-connection exists after construction");
    TEST_ASSERT(nameAt(engine, 0) == LOCAL_AGENT, "the local agent occupies index 0");
    TEST_ASSERT(agentIndexOf(engine, LOCAL_AGENT) == 0, "the self-connection carries index 0");

    return 0;
}

// The property that matters: PEER_COUNT peers get PEER_COUNT distinct indices, and each one still
// decodes to itself after a trip through the immediate data. With the old 8-bit field, peers 256
// and up truncated onto indices already in use.
int
nixlLibfabricAgentIndexTest::testIndicesAreUniqueAndSurviveTheWire(nixlLibfabricEngine &engine) {
    NIXL_INFO << "  testIndicesAreUniqueAndSurviveTheWire (" << PEER_COUNT << " peers)";

    std::unordered_set<size_t> seen_indices;
    seen_indices.insert(0); // the self-connection

    for (size_t i = 0; i < PEER_COUNT; ++i) {
        const std::string peer = peerName(i);
        TEST_ASSERT(addPeer(engine, peer) == NIXL_SUCCESS,
                    std::string("createAgentConnection succeeded for ") + peer);

        const size_t index = agentIndexOf(engine, peer);
        TEST_ASSERT(seen_indices.insert(index).second,
                    std::string("index handed to ") + peer + " is not already in use");
        TEST_ASSERT(index < NIXL_LIBFABRIC_MAX_AGENTS,
                    std::string("index handed to ") + peer + " fits the AGENT_INDEX field");
        TEST_ASSERT(nameAt(engine, index) == peer,
                    std::string("the table maps the index back to ") + peer);

        const uint64_t imm = NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER, index, 0xBEEF);
        TEST_ASSERT(NIXL_GET_AGENT_INDEX_FROM_IMM(imm) == index,
                    std::string("index survives the immediate data for ") + peer);
    }

    TEST_ASSERT(tableSize(engine) == PEER_COUNT + 1, "table holds every peer plus the local agent");
    TEST_ASSERT(
        agentIndexOf(engine, peerName(PEER_COUNT - 1)) > 255U,
        "the last peer's index is past the old 8-bit field, which is the point of the test");

    return 0;
}

// A repeat connection request for a peer already in the table must not consume a second index; that
// would leak indices towards the cap and leave a stale name behind.
int
nixlLibfabricAgentIndexTest::testKnownPeerReusesItsIndex(nixlLibfabricEngine &engine) {
    NIXL_INFO << "  testKnownPeerReusesItsIndex";

    const std::string peer = peerName(7);
    const size_t index_before = agentIndexOf(engine, peer);
    const size_t size_before = tableSize(engine);

    TEST_ASSERT(addPeer(engine, peer) == NIXL_SUCCESS, "re-adding a known peer succeeds");
    TEST_ASSERT(agentIndexOf(engine, peer) == index_before, "a known peer keeps its index");
    TEST_ASSERT(tableSize(engine) == size_before, "re-adding a known peer does not grow the table");

    return 0;
}

// disconnect() clears the name but leaves the slot in place, and the next peer gets a fresh index
// rather than the retired one. Erasing the entry instead would renumber every later peer, since the
// table size is the index allocator; reusing the slot would point the new peer at address vector
// entries the retired peer never tore down.
int
nixlLibfabricAgentIndexTest::testDisconnectRetiresTheSlotWithoutRenumbering(
    nixlLibfabricEngine &engine) {
    NIXL_INFO << "  testDisconnectRetiresTheSlotWithoutRenumbering";

    const std::string retired = peerName(100);
    const std::string neighbour = peerName(101);
    const size_t retired_index = agentIndexOf(engine, retired);
    const size_t neighbour_index = agentIndexOf(engine, neighbour);
    const size_t size_before = tableSize(engine);

    TEST_ASSERT(engine.disconnect(retired) == NIXL_SUCCESS, "disconnect of a known peer succeeds");
    TEST_ASSERT(!isConnected(engine, retired), "the connection is gone");
    TEST_ASSERT(nameAt(engine, retired_index).empty(), "the retired slot no longer names a peer");
    TEST_ASSERT(tableSize(engine) == size_before, "retiring a peer does not shrink the table");
    TEST_ASSERT(agentIndexOf(engine, neighbour) == neighbour_index,
                "a later peer is not renumbered by the retirement");
    TEST_ASSERT(nameAt(engine, neighbour_index) == neighbour,
                "a later peer still resolves at its original index");

    const std::string fresh = "peer_after_retirement";
    TEST_ASSERT(addPeer(engine, fresh) == NIXL_SUCCESS, "a new peer can still connect");
    TEST_ASSERT(agentIndexOf(engine, fresh) == size_before,
                "the new peer gets a fresh index, not the retired one");
    TEST_ASSERT(nameAt(engine, retired_index).empty(), "the retired slot stays retired");

    return 0;
}

// The cap is what the widening buys and what still bounds it. Filling the table and asking for one
// more must fail loudly: an accepted index of NIXL_LIBFABRIC_MAX_AGENTS would encode as 0 and
// impersonate the local agent.
int
nixlLibfabricAgentIndexTest::testPeerCapIsRefusedNotAliased(nixlLibfabricEngine &engine) {
    NIXL_INFO << "  testPeerCapIsRefusedNotAliased (filling to " << NIXL_LIBFABRIC_MAX_AGENTS
              << " agents)";

    size_t filler = 0;
    while (tableSize(engine) < NIXL_LIBFABRIC_MAX_AGENTS) {
        const std::string peer = "filler_" + std::to_string(filler++);
        TEST_ASSERT(addPeer(engine, peer) == NIXL_SUCCESS,
                    std::string("createAgentConnection succeeded for ") + peer);
    }
    TEST_ASSERT(tableSize(engine) == NIXL_LIBFABRIC_MAX_AGENTS,
                "the table fills exactly to the cap");

    const std::string over_cap = "peer_over_cap";
    TEST_ASSERT(addPeer(engine, over_cap) == NIXL_ERR_NOT_SUPPORTED,
                "a peer past the cap is refused with NIXL_ERR_NOT_SUPPORTED");
    TEST_ASSERT(tableSize(engine) == NIXL_LIBFABRIC_MAX_AGENTS,
                "a refused peer does not grow the table");
    TEST_ASSERT(!isConnected(engine, over_cap), "a refused peer leaves no connection behind");

    return 0;
}

int
nixlLibfabricAgentIndexTest::run() {
    nixl_b_params_t custom_params;
    nixlBackendInitParams init_params;
    init_params.localAgent = LOCAL_AGENT;
    init_params.type = "LIBFABRIC";
    init_params.customParams = &custom_params;
    init_params.enableProgTh = false;
    init_params.pthrDelay = 0;
    init_params.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE;
    init_params.enableTelemetry_ = false;

    nixlLibfabricEngine engine(&init_params);

    if (testSelfTakesIndexZero(engine) != 0) {
        return 1;
    }
    if (testIndicesAreUniqueAndSurviveTheWire(engine) != 0) {
        return 1;
    }
    if (testKnownPeerReusesItsIndex(engine) != 0) {
        return 1;
    }
    if (testDisconnectRetiresTheSlotWithoutRenumbering(engine) != 0) {
        return 1;
    }
    // Runs last: it fills the table to the cap, so no further peer can be added afterwards.
    if (testPeerCapIsRefusedNotAliased(engine) != 0) {
        return 1;
    }

    return 0;
}

int
main() {
    NIXL_INFO << "=== Libfabric Agent Index Test ===";
    NIXL_INFO << "Using mock stubs (__wrap_fi_getinfo, __wrap_fi_fabric, etc.)";

    if (nixlLibfabricAgentIndexTest::run() != 0) {
        return 1;
    }

    NIXL_INFO << "All agent index allocation tests passed";
    return 0;
}
