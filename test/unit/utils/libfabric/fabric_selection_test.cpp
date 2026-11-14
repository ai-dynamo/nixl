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
 * Unit test for EFA fabric selection in the hints of every EFA fi_getinfo call.
 *
 * The EFA provider lists each device under both the "efa-direct" and "efa" fabrics, under one
 * provider name and the same domain names, and returns efa-direct first. The backend needs
 * "efa", so it names the fabric rather than relying on the order of the results.
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <iostream>
#include <set>
#include <string>
#include <vector>

// Number of fake EFA devices to create, per fabric
static const size_t NUM_FAKE_RAILS = 4;

// The hints of every __wrap_fi_getinfo call, in call order
struct RecordedHints {
    std::string fabric_name;
    std::string prov_name;
    enum fi_ep_type ep_type;
};

static std::vector<RecordedHints> recorded_hints;

// --- Unconditional __wrap_* functions (no isTesting gate needed) ---

extern "C" int
__wrap_numa_max_node() {
    return 1;
}

extern "C" int
__wrap_numa_num_configured_nodes() {
    return 2;
}

static void
appendChain(struct fi_info **head, struct fi_info *chain) {
    if (!*head) {
        *head = chain;
        return;
    }
    struct fi_info *tail = *head;
    while (tail->next) {
        tail = tail->next;
    }
    tail->next = chain;
}

// mock_fi_info_chain() builds entries on the "efa" fabric. Relabel a chain as efa-direct,
// which the EFA provider reports under the same provider and domain names.
static struct fi_info *
mockEfaDirectChain(size_t count, uint64_t link_speed) {
    struct fi_info *head = mock_fi_info_chain(count, link_speed);
    for (struct fi_info *cur = head; cur; cur = cur->next) {
        free(cur->fabric_attr->name);
        cur->fabric_attr->name = strdup("efa-direct");
    }
    return head;
}

// Record the hints, then answer like libfabric v2.7.0: every efa-direct device first, then
// every efa one, filtered by the fabric name hint if there is one (an exact match, as in
// ofi_filter_by_names()).
extern "C" int
__wrap_fi_getinfo(uint32_t /*version*/,
                  const char * /*node*/,
                  const char * /*service*/,
                  uint64_t /*flags*/,
                  const struct fi_info *hints,
                  struct fi_info **info) {
    const char *wanted = (hints && hints->fabric_attr && hints->fabric_attr->name) ?
        hints->fabric_attr->name :
        nullptr;
    if (hints) {
        recorded_hints.push_back({wanted ? wanted : "",
                                  hints->fabric_attr && hints->fabric_attr->prov_name ?
                                      hints->fabric_attr->prov_name :
                                      "",
                                  hints->ep_attr ? hints->ep_attr->type : FI_EP_UNSPEC});
    }

    const uint64_t speed = 100ull * NIXL_LIBFABRIC_GIGA;
    *info = nullptr;
    if (!wanted || strcmp(wanted, "efa-direct") == 0) {
        appendChain(info, mockEfaDirectChain(NUM_FAKE_RAILS, speed));
    }
    if (!wanted || strcmp(wanted, "efa") == 0) {
        appendChain(info, mock_fi_info_chain(NUM_FAKE_RAILS, speed));
    }
    return *info ? 0 : -FI_ENODATA;
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

// --- Tests ---

static int
testRailNamesEfaFabric() {
    NIXL_INFO << "  testRailNamesEfaFabric";
    recorded_hints.clear();

    nixlLibfabricRail rail("efa_0", "efa", 0, FI_HMEM_SYSTEM);

    const std::string fabric = rail.getRailInfo()->fabric_attr->name;
    TEST_ASSERT(fabric == "efa", "rail opened the efa fabric, got '" + fabric + "'");
    TEST_ASSERT(!recorded_hints.empty(), "rail called fi_getinfo");
    for (const auto &h : recorded_hints) {
        TEST_ASSERT(h.fabric_name == "efa",
                    "rail hints name the efa fabric, got '" + h.fabric_name + "'");
    }
    return 0;
}

static int
testDiscoveryListsEachDeviceOnce() {
    NIXL_INFO << "  testDiscoveryListsEachDeviceOnce";
    recorded_hints.clear();

    const auto [provider, devices] = LibfabricUtils::getAvailableNetworkDevices();
    TEST_ASSERT(provider == "efa", "discovery picks the efa provider, got '" + provider + "'");
    TEST_ASSERT(devices.size() == NUM_FAKE_RAILS,
                "one device per NIC, not one per fabric: got " + std::to_string(devices.size()));
    const std::set<std::string> unique(devices.begin(), devices.end());
    TEST_ASSERT(unique.size() == devices.size(), "no duplicate devices");

    // Discovery spans every provider, so it must not name a fabric -- that would drop cxi, tcp
    // and sockets, whose fabrics are called something else. It filters the results instead.
    TEST_ASSERT(recorded_hints.size() == 1, "discovery calls fi_getinfo once");
    TEST_ASSERT(recorded_hints[0].fabric_name.empty(), "discovery does not name a fabric");
    return 0;
}

static int
testEveryEfaQueryNamesTheFabric() {
    NIXL_INFO << "  testEveryEfaQueryNamesTheFabric";
    recorded_hints.clear();

    // Builds the topology, including the PCIe mapping, then one rail per device.
    nixlLibfabricRailManager mgr(0);
    TEST_ASSERT(mgr.getNumRails() == NUM_FAKE_RAILS,
                "one rail per NIC, got " + std::to_string(mgr.getNumRails()));

    // Every query for the EFA provider must name the fabric too, so that the PCIe mapping and
    // the rails describe the same one.
    size_t efa_queries = 0;
    for (const auto &h : recorded_hints) {
        if (h.prov_name == "efa") {
            ++efa_queries;
            TEST_ASSERT(h.fabric_name == "efa",
                        "an EFA query named fabric '" + h.fabric_name + "' instead of efa");
        }
    }
    TEST_ASSERT(efa_queries > 0, "the PCIe mapping queried the efa provider");
    return 0;
}

int
main() {
    NIXL_INFO << "=== Fabric Selection Test ===";
    NIXL_INFO << "Using mock stubs (__wrap_fi_getinfo, __wrap_fi_fabric, etc.)";
    cq_ops_stub.size = sizeof(fi_ops_cq);

    int res;
    if ((res = testRailNamesEfaFabric()) != 0) {
        return res;
    }
    if ((res = testDiscoveryListsEachDeviceOnce()) != 0) {
        return res;
    }
    if ((res = testEveryEfaQueryNamesTheFabric()) != 0) {
        return res;
    }

    NIXL_INFO << "=== All fabric selection tests PASSED ===";
    return 0;
}
