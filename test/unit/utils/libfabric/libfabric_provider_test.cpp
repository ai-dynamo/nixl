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
 * Tests for libfabric provider handling: LibfabricUtils::baseProviderName(),
 * LibfabricUtils::selectVerbsDomains(), the provider table and the selection rule built on it, plus
 * a live check that a verbs host ends up with real PCIe information for its NICs.
 *
 * The pure functions are tested against synthetic prov_name → domain maps so the interesting
 * cases can be covered on a host that has no verbs devices at all. The shapes used are the ones
 * libfabric actually produces; they are noted per case.
 *
 * The provider-table assertions look tautological one at a time -- each restates a flag the table
 * sets. What they are worth is collectively: every one of those flags used to be an independent
 * `provider == "..."` comparison in a different file, and the table exists so that adding a provider
 * is one edit rather than seven. These pin the flags to the behaviour each one stands for, so a
 * future edit that drops one fails here rather than on a fabric nobody in CI has.
 */

#include "libfabric/libfabric_common.h"
#include "libfabric/libfabric_topology.h"
#include "libfabric/libfabric_rail_manager.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void
check(bool cond, const std::string &what) {
    if (cond) {
        NIXL_INFO << "  PASS: " << what;
    } else {
        NIXL_ERROR << "  FAIL: " << what;
        ++failures;
    }
}

using ProviderMap = std::unordered_map<std::string, std::vector<std::string>>;

std::string
join(const std::vector<std::string> &values) {
    std::string out;
    for (const auto &value : values) {
        if (!out.empty()) {
            out += ",";
        }
        out += value;
    }
    return out;
}

/****************************************
 * 1. baseProviderName
 *****************************************/

void
testBaseProviderName() {
    NIXL_INFO << "1. Testing baseProviderName()";

    check(LibfabricUtils::baseProviderName("efa") == "efa", "an unlayered name is returned as is");
    check(LibfabricUtils::baseProviderName("verbs;ofi_rxm") == "verbs",
          "'verbs;ofi_rxm' reduces to 'verbs'");
    check(LibfabricUtils::baseProviderName("verbs;ofi_rxm;ofi_hook_perf") == "verbs",
          "a multiply-layered name reduces to its core provider");
    check(LibfabricUtils::baseProviderName("") == "", "the empty name is handled");
    check(LibfabricUtils::baseProviderName(";ofi_rxm") == "",
          "a leading separator yields an empty core name rather than reading past it");
    // The whole point of the function: the plain equality test this replaces would have missed
    // every verbs entry, since verbs never reports its RDM endpoint as plain "verbs".
    check(LibfabricUtils::baseProviderName("verbs;ofi_rxm") != "verbs;ofi_rxm",
          "the layered name is not equal to its core name");
}

/****************************************
 * 2. selectVerbsDomains
 *****************************************/

void
testSelectVerbsDomains() {
    NIXL_INFO << "2. Testing selectVerbsDomains()";

    // Exactly what this host reports for FI_EP_RDM: four entries per NIC from ofi_rxm (one per
    // capability variant) and the same NICs again from ofi_rxd under their DGRAM domains.
    const ProviderMap real_shape = {
        {"verbs;ofi_rxm",
         {"rocep153s0f0",
          "rocep153s0f0",
          "rocep153s0f0",
          "rocep153s0f0",
          "rocep153s0f1",
          "rocep153s0f1",
          "rocep153s0f1",
          "rocep153s0f1"}},
        {"verbs;ofi_rxd", {"rocep153s0f0-dgram", "rocep153s0f1-dgram"}},
        {"tcp", {"ens1f0np0", "lo"}},
        {"psm3", {"rocep153s0f0", "rocep153s0f1"}},
    };
    const std::vector<std::string> selected = LibfabricUtils::selectVerbsDomains(real_shape);
    NIXL_INFO << "   selected: [" << join(selected) << "]";
    check(selected == std::vector<std::string>({"rocep153s0f0", "rocep153s0f1"}),
          "one rail per NIC, deduplicated, sorted, with the ofi_rxd DGRAM domains dropped");

    // A hyphen in the device name itself must not be mistaken for an endpoint-variant suffix.
    // Testing for "the name contains a hyphen" instead of "ends with a known suffix" would drop
    // this NIC entirely.
    const ProviderMap hyphenated = {
        {"verbs;ofi_rxm", {"roce-nic0", "roce-nic0-dgram", "roce-nic0-xrc"}},
    };
    const std::vector<std::string> hyphen_selected = LibfabricUtils::selectVerbsDomains(hyphenated);
    NIXL_INFO << "   selected: [" << join(hyphen_selected) << "]";
    check(hyphen_selected == std::vector<std::string>({"roce-nic0"}),
          "a hyphenated device name survives while its -dgram and -xrc variants are dropped");

    check(LibfabricUtils::selectVerbsDomains({{"verbs;ofi_rxd", {"mlx5_0-dgram"}}}).empty(),
          "a host offering verbs only through ofi_rxd selects nothing");
    check(LibfabricUtils::selectVerbsDomains({{"efa", {"rdmap16s27-rdm"}}, {"tcp", {"lo"}}}).empty(),
          "no verbs entry selects nothing, and efa's own -rdm domain is left alone");
    check(LibfabricUtils::selectVerbsDomains({}).empty(), "an empty map selects nothing");

    // Sort order is what makes rail indices stable: provider_device_map is unordered, so without
    // it the same host could number its rails differently from one run to the next.
    const ProviderMap reversed = {{"verbs;ofi_rxm", {"mlx5_3", "mlx5_1", "mlx5_2", "mlx5_0"}}};
    check(LibfabricUtils::selectVerbsDomains(reversed) ==
              std::vector<std::string>({"mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"}),
          "domains come back in a deterministic order regardless of discovery order");
}

/****************************************
 * 3. The provider table
 *****************************************/

void
testProviderTable() {
    NIXL_INFO << "3. Testing the provider table";

    const auto &providers = LibfabricUtils::knownProviders();

    // Preference order is the table order, and it is load-bearing: on a host with both RoCE NICs
    // and a tcp device, picking tcp would silently give up RDMA.
    std::vector<std::string> order;
    for (const auto *provider : providers) {
        order.push_back(provider->name);
    }
    NIXL_INFO << "   preference order: [" << join(order) << "]";
    check(order == std::vector<std::string>({"cxi", "efa", "verbs", "tcp", "sockets"}),
          "providers are in the documented preference order, RDMA before host-stack");

    // Lookup has to accept a layered prov_name, because that is the only form verbs ever appears
    // in; a plain equality test against "verbs" is exactly the bug baseProviderName() exists for.
    check(LibfabricUtils::findProviderInfo("verbs;ofi_rxm") != nullptr,
          "findProviderInfo() resolves a layered prov_name");
    check(LibfabricUtils::findProviderInfo("verbs;ofi_rxm") ==
              LibfabricUtils::findProviderInfo("verbs"),
          "the layered and core names resolve to the same entry");
    check(LibfabricUtils::findProviderInfo("psm3") == nullptr,
          "an unlisted provider has no table entry");
    check(LibfabricUtils::findProviderInfo("") == nullptr, "the empty name has no table entry");

    // An unlisted provider must still get a usable rail rather than a special code path, which is
    // what defaultProviderInfo() is for.
    const auto &fallback = LibfabricUtils::defaultProviderInfo();
    check((fallback.mr_mode & FI_MR_HMEM) != 0,
          "the fallback entry requests FI_MR_HMEM, so device memory still works on an unlisted "
          "provider");
    check(!fallback.needs_prov_name_hint,
          "the fallback entry does not pin hints to one provider name");

    const auto *verbs = LibfabricUtils::findProviderInfo("verbs");
    const auto *cxi = LibfabricUtils::findProviderInfo("cxi");
    const auto *tcp = LibfabricUtils::findProviderInfo("tcp");

    // Each of these was an independent `provider == "..."` comparison before the table, so these
    // assertions are what stops one of them silently reverting.
    check(verbs != nullptr && verbs->needs_prov_name_hint,
          "verbs names itself in rail hints, so psm3 cannot claim the same domain");
    check(verbs != nullptr && verbs->needs_all_rail_progress,
          "verbs progresses every rail, so the passive side accepts lazy connections");
    check(cxi != nullptr && !cxi->needs_all_rail_progress,
          "a connectionless RDM provider keeps the cheaper active-rails-only progress");
    check(cxi != nullptr && (cxi->extra_caps & FI_RMA_EVENT) != 0, "cxi requests FI_RMA_EVENT");
    check(cxi != nullptr && (cxi->mr_mode & FI_MR_ENDPOINT) != 0,
          "cxi requests FI_MR_ENDPOINT in its mr_mode");
    check(tcp != nullptr && (tcp->mr_mode & (FI_MR_PROV_KEY | FI_MR_VIRT_ADDR)) == 0,
          "tcp asks for neither FI_MR_PROV_KEY nor FI_MR_VIRT_ADDR, which it does not support");

    // Only the host-stack providers collapse to a single rail; multiplying rails there would only
    // multiply host copies, while for an RDMA provider one rail per NIC is the entire point.
    for (const auto *provider : providers) {
        const bool is_host_stack =
            (std::string(provider->name) == "tcp" || std::string(provider->name) == "sockets");
        check(provider->single_domain == is_host_stack,
              std::string(provider->name) + (is_host_stack ? " uses a single rail" : " uses one rail per NIC"));
        check(provider->needs_full_topology == !is_host_stack,
              std::string(provider->name) +
                  (is_host_stack ? " skips hwloc grouping" : " does PCIe-proximity grouping"));
    }

    // The two helpers the rest of the plugin actually calls, rather than reading flags directly.
    check(LibfabricUtils::providerNeedsAllRailProgress("verbs;ofi_rxm"),
          "providerNeedsAllRailProgress() accepts the layered name it will really be given");
    check(!LibfabricUtils::providerNeedsAllRailProgress("efa"),
          "providerNeedsAllRailProgress() is false for efa");
    check(!LibfabricUtils::providerNeedsAllRailProgress("psm3"),
          "an unlisted provider does not get all-rail progress");
    check(LibfabricUtils::providerNeedsFullTopology("efa") &&
              LibfabricUtils::providerNeedsFullTopology("cxi") &&
              LibfabricUtils::providerNeedsFullTopology("verbs"),
          "providerNeedsFullTopology() is true for every RDMA provider");
    check(!LibfabricUtils::providerNeedsFullTopology("tcp") &&
              !LibfabricUtils::providerNeedsFullTopology("sockets"),
          "providerNeedsFullTopology() is false for the host-stack providers");
    check(!LibfabricUtils::providerNeedsFullTopology("none"),
          "the no-provider sentinel does not request topology discovery");
}

/****************************************
 * 4. Provider selection over synthetic maps
 *****************************************/

void
testProviderSelection() {
    NIXL_INFO << "4. Testing getAvailableNetworkDevices()'s selection rule";

    /*
     * getAvailableNetworkDevices() itself needs a real fabric, but the rule it applies is the table
     * walk plus each entry's select_domains hook, and those are testable. Reproduced here rather
     * than exported, because what is worth pinning down is the outcome, not the loop.
     */
    auto select = [](const ProviderMap &map) -> std::pair<std::string, std::vector<std::string>> {
        for (const auto *provider : LibfabricUtils::knownProviders()) {
            std::vector<std::string> domains;
            if (provider->select_domains != nullptr) {
                domains = provider->select_domains(map);
            } else {
                domains = LibfabricUtils::selectDomainsGeneric(map, provider->name);
            }
            if (domains.empty()) {
                continue;
            }
            if (provider->single_domain && domains.size() > 1) {
                domains.resize(1);
            }
            return {provider->name, domains};
        }
        return {"none", {}};
    };

    // A RoCE host that also has tcp: verbs must win, which is the behaviour change the table order
    // encodes. Picking tcp here would silently drop to the host stack on real RDMA hardware.
    const auto roce = select({
        {"verbs;ofi_rxm", {"rocep153s0f0", "rocep153s0f1"}},
        {"verbs;ofi_rxd", {"rocep153s0f0-dgram"}},
        {"tcp", {"ens1f0np0", "lo"}},
        {"psm3", {"rocep153s0f0"}},
    });
    check(roce.first == "verbs", "verbs is preferred over tcp on a RoCE host");
    check(roce.second == std::vector<std::string>({"rocep153s0f0", "rocep153s0f1"}),
          "the verbs hook's filtered domain list is the one used");

    // efa outranks verbs, and its domains come through the generic rule untouched.
    const auto efa = select({
        {"efa", {"rdmap16s27-rdm", "rdmap32s27-rdm"}},
        {"verbs;ofi_rxm", {"mlx5_0"}},
        {"tcp", {"lo"}},
    });
    check(efa.first == "efa", "efa outranks verbs");
    check(efa.second == std::vector<std::string>({"rdmap16s27-rdm", "rdmap32s27-rdm"}),
          "the generic rule preserves fi_getinfo order, so existing efa rail numbering is unchanged");

    check(select({{"cxi", {"cxi0"}}, {"efa", {"rdmap16s27-rdm"}}}).first == "cxi",
          "cxi outranks efa");

    // The generic rule has to deduplicate: fi_getinfo returns one entry per capability variant, and
    // without this each would become its own rail on the same NIC.
    const auto dupes = select({{"efa", {"rdmap16s27-rdm", "rdmap16s27-rdm", "rdmap32s27-rdm"}}});
    check(dupes.second == std::vector<std::string>({"rdmap16s27-rdm", "rdmap32s27-rdm"}),
          "the generic rule deduplicates repeated domains");

    // ...and take exactly the provider's own key: a layered variant reaches the same NICs through
    // slower emulation, so its domains must not become extra rails, whatever the map's hash order.
    const auto layered = select({
        {"efa;ofi_rxd", {"rdmap16s27-dgrm", "rdmap32s27-dgrm"}},
        {"efa", {"rdmap16s27-rdm", "rdmap32s27-rdm"}},
    });
    check(layered.first == "efa", "efa is selected when it also has layered variants");
    check(layered.second == std::vector<std::string>({"rdmap16s27-rdm", "rdmap32s27-rdm"}),
          "the generic rule ignores layered variants such as efa;ofi_rxd");

    // tcp and sockets collapse to one rail however many domains they report.
    const auto tcp = select({{"tcp", {"ens1f0np0", "lo", "docker0"}}});
    check(tcp.first == "tcp" && tcp.second.size() == 1,
          "tcp collapses to a single rail regardless of how many domains it offers");
    check(select({{"sockets", {"a", "b"}}}).second.size() == 1, "sockets collapses too");

    check(select({}).first == "none", "an empty map selects no provider");
    check(select({{"psm3", {"rocep153s0f0"}}}).first == "none",
          "a host offering only an unlisted provider selects nothing rather than mis-claiming it");
}

/****************************************
 * 5. Live topology on a verbs host
 *****************************************/

void
testLiveVerbsTopology() {
    NIXL_INFO << "5. Testing live topology";

    nixlLibfabricTopology topology;
    const std::string provider = topology.getProviderName();
    NIXL_INFO << "   provider: " << provider << ", " << topology.getAllDevices().size()
              << " device(s), " << topology.getTotalNicCount() << " NIC(s) with PCIe info";

    if (provider != "verbs") {
        NIXL_INFO << "   not a verbs host, skipping the verbs-specific assertions";
        return;
    }

    /*
     * The real subject here is the hwloc fallback in buildPcieToLibfabricMapping(): verbs reports
     * bus_type FI_BUS_UNKNOWN, so every one of these NICs would have no PCIe address and
     * nic_info_map would be empty. That failure is silent -- grouping would fail, the fallback
     * mapping would hand every accelerator every rail, and nothing would report an error.
     */
    check(topology.getTotalNicCount() == topology.getAllDevices().size(),
          "every selected verbs device resolved to a PCIe address");
    check(topology.hasPcieDevices(), "hasPcieDevices() reflects the resolved addresses");

    /*
     * Device discovery and the PCIe mapping used to be two separate fi_getinfo() calls with
     * different hints, which meant they could return different domain sets. They are one call now,
     * so the two views cannot disagree -- but "cannot" is worth an assertion, because the failure
     * mode of the old arrangement was silent: a domain in one view and not the other left grouping
     * pointing at a NIC that never became a rail.
     */
    for (const auto &device : topology.getAllDevices()) {
        check(!topology.getPcieAddress(device).empty(),
              "selected device " + device + " has a PCIe address from the single discovery query");
    }

    for (const auto &device : topology.getAllDevices()) {
        check(device.find("-dgram") == std::string::npos &&
                  device.find("-xrc") == std::string::npos,
              "selected device " + device + " is an RDM-capable domain");
        check(topology.getDeviceNumaNode(device) != nixlLibfabricTopology::INVALID_NUMA_NODE_ID,
              "selected device " + device + " resolved to a NUMA node");
    }

    /*
     * Not asserted: that getNicsForPci(<accelerator BDF>) returns a proximity group rather than
     * the all-devices fallback. On a host with one accelerator the group is the whole NIC set, so
     * the two answers are identical and the assertion would prove nothing. The NIC count above is
     * the part that actually distinguishes them.
     */
}

} // namespace

/*
 * Peer-rail assignment for the rails whose fabric-matched peer rail is unusable. Each case is a
 * scenario a code review raised against an earlier version of this rule.
 */
void
testFallbackPeerRails() {
    NIXL_INFO << "Testing fallback peer rail assignment";
    using V = std::vector<size_t>;
    const size_t none = NIXL_LF_NO_PEER_RAIL;
    const auto assign = nixlLibfabricRailManager::assignFallbackPeerRails;

    // No fabric information: same id wins, even when it is not the rank pairing. On a rail-optimised
    // fabric local rail 2 only reaches peer rail 2, so rank pairing (2->0, 3->1) would hang.
    check(assign({2, 3}, {0, 1, 2, 3}, {}) == V({2, 3}),
          "without fabric ids, a registered same-id peer rail is preferred");

    // Policy order of selected_rails must not change the pairing, and it stays one-to-one.
    check(assign({2, 0}, {0, 1}, {}) == V({1, 0}),
          "without a same-id peer rail, rails pair by rank in rail-id order, one-to-one");
    check(assign({1, 0}, {0, 1}, {}) == V({1, 0}), "policy order does not change the pairing");

    // A fabric match claims its peer rail; the unmatched rail must not pile onto it.
    check(assign({0, 1}, {0, 1}, {1, none}) == V({1, 0}),
          "an unmatched rail takes an unclaimed peer rail, not one a fabric match holds");

    // Matched peer rail did not register the buffer: same id is known to be on another segment,
    // so the rail takes an unclaimed registered peer rail instead.
    check(assign({0, 1}, {0, 2}, {2, 3}) == V({2, 0}),
          "a rail whose matched peer is unregistered skips the same-id rail");

    // More local rails than registered peer rails: sharing only once all are claimed.
    check(assign({0, 1, 2}, {0}, {}) == V({0, 0, 0}),
          "with one registered peer rail, every local rail shares it");
    check(assign({0, 1, 2, 3}, {0, 1}, {}) == V({0, 1, 0, 1}),
          "sharing only starts once every registered peer rail is claimed");

    check(assign({0, 1}, {}, {}) == V({none, none}),
          "no registered peer rail leaves the rails unassigned rather than indexing an empty list");
}

int
main() {
    NIXL_INFO << "=== Testing Libfabric verbs provider selection ===";

    testBaseProviderName();
    testSelectVerbsDomains();
    testProviderTable();
    testProviderSelection();
    testFallbackPeerRails();
    testLiveVerbsTopology();

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
