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
 * Unit test for source-address based sender resolution (NIXL-150 / NIXL-166).
 *
 * The receiver used to learn which peer sent a notification or a remote write from an
 * 8-bit agent index in the RDMA immediate data, which capped the number of peers at 256.
 * It now resolves the sender from the source address libfabric reports on each completion
 * (FI_SOURCE + fi_cq_readfrom). These tests cover the resolution table and the two
 * completion handlers that consume it, with libfabric mocked out.
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

// Number of fake EFA devices to create
static const size_t NUM_FAKE_RAILS = 4;

// --- Unconditional __wrap_* functions ---

extern "C" int
__wrap_numa_max_node() {
    return 1;
}

extern "C" int
__wrap_numa_num_configured_nodes() {
    return 2;
}

// When set, behave like a provider without FI_SOURCE: a requested secondary capability it
// cannot support makes fi_getinfo fail with FI_ENODATA.
static bool mock_reject_fi_source = false;

// When set, behave like a provider without FI_HMEM, so the rail takes its retry path.
static bool mock_reject_fi_hmem = false;

// When set, every fi_getinfo call appends hints->caps here.
static bool record_hint_caps = false;
static std::vector<uint64_t> recorded_hint_caps;

extern "C" int
__wrap_fi_getinfo(uint32_t /*version*/,
                  const char * /*node*/,
                  const char * /*service*/,
                  uint64_t /*flags*/,
                  const struct fi_info *hints,
                  struct fi_info **info) {
    if (record_hint_caps && hints) {
        recorded_hint_caps.push_back(hints->caps);
    }
    if (mock_reject_fi_hmem && hints && (hints->caps & FI_HMEM)) {
        *info = nullptr;
        return -FI_ENODATA;
    }
    if (mock_reject_fi_source && hints && (hints->caps & FI_SOURCE)) {
        *info = nullptr;
        return -FI_ENODATA;
    }
    *info = mock_fi_info_chain(NUM_FAKE_RAILS, 100ull * NIXL_LIBFABRIC_GIGA);
    return 0;
}

extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr * /*attr*/, struct fid_fabric **fabric, void * /*context*/) {
    *fabric = mock_fabric_create();
    return 0;
}

// --- Fake completion queue ---
//
// Tests push completions plus their source addresses, then call
// progressCompletionQueue() and inspect what the rail's callbacks saw.

struct FakeCompletion {
    struct fi_cq_data_entry entry;
    fi_addr_t src_addr;
};

static std::vector<FakeCompletion> pending_completions;
static size_t completions_consumed = 0;

static ssize_t
fi_cq_readfrom_stub(struct fid_cq * /*cq*/, void *buf, size_t count, fi_addr_t *src_addr) {
    if (completions_consumed >= pending_completions.size()) {
        return -FI_EAGAIN;
    }
    auto *entries = static_cast<struct fi_cq_data_entry *>(buf);
    size_t n = 0;
    while (n < count && completions_consumed < pending_completions.size()) {
        const FakeCompletion &fc = pending_completions[completions_consumed++];
        entries[n] = fc.entry;
        // A caller that forgets to size the src_addr array to `count`, or that mixes up the
        // two arrays, shows up here.
        if (src_addr != nullptr) {
            src_addr[n] = fc.src_addr;
        }
        ++n;
    }
    return static_cast<ssize_t>(n);
}

static ssize_t
fi_cq_readerr_stub(struct fid_cq * /*cq*/, struct fi_cq_err_entry * /*buf*/, uint64_t /*flags*/) {
    return -FI_EAGAIN;
}

static void
queue_reset() {
    pending_completions.clear();
    completions_consumed = 0;
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

// Distinct fake endpoint-name blobs, one per (agent, remote rail) pair.
static std::vector<std::array<char, LF_EP_NAME_MAX_LEN>>
makeEndpoints(const std::string &agent, size_t num_rails) {
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> eps(num_rails);
    for (size_t r = 0; r < num_rails; ++r) {
        eps[r].fill(0);
        const std::string name = agent + ":rail" + std::to_string(r);
        std::memcpy(eps[r].data(), name.c_str(), std::min(name.size(), eps[r].size() - 1));
    }
    return eps;
}

// --- Tests ---

static int
testInsertAndResolve(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testInsertAndResolve";
    mock_av_reset();

    const nixlLibfabricRail &rail = mgr.getRail(0);
    const auto eps = makeEndpoints("agentA", 1);

    fi_addr_t fi_addr = FI_ADDR_UNSPEC;
    nixl_status_t st = rail.insertAddress(eps[0].data(), 7, &fi_addr);
    TEST_ASSERT(st == NIXL_SUCCESS, "insertAddress succeeded");
    TEST_ASSERT(rail.resolveSourceAddress(fi_addr) == 7u, "resolves to the agent index given");

    return 0;
}

static int
testUnresolvableAddresses(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testUnresolvableAddresses";
    mock_av_reset();

    const nixlLibfabricRail &rail = mgr.getRail(1);
    const auto eps = makeEndpoints("agentB", 1);
    fi_addr_t fi_addr = FI_ADDR_UNSPEC;
    TEST_ASSERT(rail.insertAddress(eps[0].data(), 3, &fi_addr) == NIXL_SUCCESS, "insert ok");

    // FI_ADDR_NOTAVAIL is what the provider reports when it cannot identify the sender --
    // for instance a peer that is only in EFA's implicit AV. It must never resolve.
    TEST_ASSERT(rail.resolveSourceAddress(FI_ADDR_NOTAVAIL) == nixlLibfabricRail::kUnknownAgentIdx,
                "FI_ADDR_NOTAVAIL does not resolve");

    // Past the end of the table.
    TEST_ASSERT(rail.resolveSourceAddress(fi_addr + 1000) == nixlLibfabricRail::kUnknownAgentIdx,
                "out-of-range fi_addr does not resolve");

    return 0;
}

static int
testSparseAndHighAgentIndex(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testSparseAndHighAgentIndex";
    mock_av_reset();

    const nixlLibfabricRail &rail = mgr.getRail(2);

    // The whole point of the change: an agent index far above the old 8-bit ceiling, and
    // above 16 bits too, must round-trip.
    const uint32_t big_idx = 70000;
    const auto eps = makeEndpoints("agentBig", 1);
    fi_addr_t fi_addr = FI_ADDR_UNSPEC;
    TEST_ASSERT(rail.insertAddress(eps[0].data(), big_idx, &fi_addr) == NIXL_SUCCESS, "insert ok");
    TEST_ASSERT(rail.resolveSourceAddress(fi_addr) == big_idx, "large agent index round-trips");

    // Table growth: insert enough distinct endpoints that the vector has to resize past its
    // initial size, and check both the first and the last mapping survive.
    const size_t kMany = 600;
    std::vector<fi_addr_t> addrs;
    for (size_t i = 0; i < kMany; ++i) {
        const auto e = makeEndpoints("agentMany" + std::to_string(i), 1);
        fi_addr_t a = FI_ADDR_UNSPEC;
        TEST_ASSERT(rail.insertAddress(e[0].data(), static_cast<uint32_t>(1000 + i), &a) ==
                        NIXL_SUCCESS,
                    "bulk insert ok");
        addrs.push_back(a);
    }
    for (size_t i = 0; i < kMany; ++i) {
        TEST_ASSERT(rail.resolveSourceAddress(addrs[i]) == static_cast<uint32_t>(1000 + i),
                    "every bulk mapping survives table growth");
    }
    TEST_ASSERT(rail.resolveSourceAddress(fi_addr) == big_idx,
                "earlier mapping survives table growth");

    return 0;
}

static int
testRemapAndRemove(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testRemapAndRemove";
    mock_av_reset();

    const nixlLibfabricRail &rail = mgr.getRail(3);
    const auto eps = makeEndpoints("agentC", 1);

    fi_addr_t first = FI_ADDR_UNSPEC;
    TEST_ASSERT(rail.insertAddress(eps[0].data(), 11, &first) == NIXL_SUCCESS, "first insert ok");

    // Re-inserting the same endpoint returns the same fi_addr (what EFA does), which is the
    // reconnect path: the agent got a fresh index. Last writer wins.
    fi_addr_t second = FI_ADDR_UNSPEC;
    TEST_ASSERT(rail.insertAddress(eps[0].data(), 12, &second) == NIXL_SUCCESS, "re-insert ok");
    TEST_ASSERT(first == second, "duplicate insert returns the same fi_addr");
    TEST_ASSERT(rail.resolveSourceAddress(first) == 12u, "re-map takes the new agent index");

    // Removing the address must drop the mapping, otherwise a later completion whose
    // fi_addr the provider recycles would be credited to the departed agent.
    TEST_ASSERT(rail.removeAddress(first) == NIXL_SUCCESS, "removeAddress ok");
    TEST_ASSERT(rail.resolveSourceAddress(first) == nixlLibfabricRail::kUnknownAgentIdx,
                "mapping cleared after removeAddress");

    return 0;
}

static int
testInsertAllAddressesCrossProduct(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testInsertAllAddressesCrossProduct";
    mock_av_reset();

    // Every endpoint of a peer lands in every local rail's AV. That cross product is what
    // makes a write from any of the peer's rails, arriving on any of ours, resolvable.
    const size_t num_rails = mgr.getNumRails();
    const auto eps = makeEndpoints("peerX", num_rails);

    std::unordered_map<size_t, std::vector<fi_addr_t>> fi_addrs;
    std::vector<char *> ep_names;
    nixl_status_t st = mgr.insertAllAddresses(eps, 4242, fi_addrs, ep_names);
    TEST_ASSERT(st == NIXL_SUCCESS, "insertAllAddresses succeeded");
    TEST_ASSERT(fi_addrs.size() == num_rails, "one address list per local rail");

    for (size_t rail_id = 0; rail_id < num_rails; ++rail_id) {
        TEST_ASSERT(fi_addrs.at(rail_id).size() == num_rails,
                    "every remote endpoint inserted on every local rail");
        for (fi_addr_t a : fi_addrs.at(rail_id)) {
            TEST_ASSERT(mgr.getRail(rail_id).resolveSourceAddress(a) == 4242u,
                        "each of the peer's endpoints resolves to the peer on each rail");
        }
    }

    return 0;
}

static int
testRemoteWriteCompletionAttribution(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testRemoteWriteCompletionAttribution";
    mock_av_reset();
    queue_reset();

    nixlLibfabricRail &rail = mgr.getRail(0);

    // Two peers, so a mis-resolution shows up as the wrong index rather than as a crash.
    const auto eps_p = makeEndpoints("peerP", 1);
    const auto eps_q = makeEndpoints("peerQ", 1);
    fi_addr_t addr_p = FI_ADDR_UNSPEC, addr_q = FI_ADDR_UNSPEC;
    TEST_ASSERT(rail.insertAddress(eps_p[0].data(), 300, &addr_p) == NIXL_SUCCESS, "insert P");
    TEST_ASSERT(rail.insertAddress(eps_q[0].data(), 90000, &addr_q) == NIXL_SUCCESS, "insert Q");

    std::vector<std::pair<uint16_t, uint32_t>> seen; // (xfer_id, agent_idx)
    rail.setXferIdCallback([&seen](uint64_t imm_data, uint32_t agent_idx) {
        seen.emplace_back(static_cast<uint16_t>(NIXL_GET_XFER_ID_FROM_IMM(imm_data)), agent_idx);
    });

    auto make_write = [](uint16_t xfer_id, fi_addr_t src) {
        FakeCompletion fc{};
        fc.entry.flags = FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA | FI_RMA;
        fc.entry.len = 4096;
        fc.entry.data = NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER, xfer_id, 0);
        fc.src_addr = src;
        return fc;
    };

    pending_completions.push_back(make_write(11, addr_p));
    pending_completions.push_back(make_write(22, addr_q));
    pending_completions.push_back(make_write(33, addr_p));

    nixl_status_t st = rail.progressCompletionQueue();
    TEST_ASSERT(st == NIXL_SUCCESS, "progressCompletionQueue processed the batch");
    TEST_ASSERT(seen.size() == 3, "all three write completions reached the callback");
    TEST_ASSERT(seen[0].first == 11 && seen[0].second == 300u, "first credited to P");
    TEST_ASSERT(seen[1].first == 22 && seen[1].second == 90000u,
                "second credited to Q, with an index far above the old 8-bit ceiling");
    TEST_ASSERT(seen[2].first == 33 && seen[2].second == 300u, "third credited to P");

    // The reserved immediate-data field must go out as zero now that the agent index is
    // gone from the wire.
    TEST_ASSERT(NIXL_GET_IMM_RESERVED(pending_completions[0].entry.data) == 0,
                "reserved imm_data field is zero");

    // An unknown source address must not be attributed to anybody.
    queue_reset();
    seen.clear();
    pending_completions.push_back(make_write(44, FI_ADDR_NOTAVAIL));
    st = rail.progressCompletionQueue();
    // Dropped locally; reporting it would fail whichever API call progressed the CQ.
    TEST_ASSERT(st == NIXL_SUCCESS, "unattributable write completion is dropped, not an error");
    TEST_ASSERT(seen.empty(), "unattributable write completion is not credited to any peer");

    rail.setXferIdCallback(nullptr);
    return 0;
}

static int
testRecvCompletionAttribution(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testRecvCompletionAttribution";
    mock_av_reset();
    queue_reset();

    nixlLibfabricRail &rail = mgr.getRail(1);

    const auto eps = makeEndpoints("peerR", 1);
    fi_addr_t addr_r = FI_ADDR_UNSPEC;
    TEST_ASSERT(rail.insertAddress(eps[0].data(), 65537, &addr_r) == NIXL_SUCCESS, "insert R");

    std::vector<uint32_t> notif_senders;
    std::vector<std::string> handshakes;
    rail.setNotificationCallback(
        [&notif_senders](const std::string &, uint32_t idx) { notif_senders.push_back(idx); });
    rail.setHandshakeCallback(
        [&handshakes](const std::string &payload) { handshakes.push_back(payload); });

    // A receive completion refers to a posted control request by its context.
    auto make_recv = [&rail](uint64_t msg_type, fi_addr_t src, const std::string &payload) {
        nixlLibfabricReq *req = rail.allocateControlRequest(NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE,
                                                            /*req_id=*/1);
        assert(req != nullptr);
        std::memcpy(req->buffer, payload.data(), payload.size());
        FakeCompletion fc{};
        fc.entry.flags = FI_RECV | FI_MSG;
        fc.entry.len = payload.size();
        fc.entry.op_context = &req->ctx;
        fc.entry.data = NIXL_MAKE_IMM_DATA(msg_type, /*xfer_id=*/5, 0);
        fc.src_addr = src;
        return fc;
    };

    pending_completions.push_back(
        make_recv(NIXL_LIBFABRIC_MSG_NOTIFICTION, addr_r, std::string(64, 'n')));
    nixl_status_t st = rail.progressCompletionQueue();
    TEST_ASSERT(st == NIXL_SUCCESS, "notification processed");
    TEST_ASSERT(notif_senders.size() == 1 && notif_senders[0] == 65537u,
                "notification attributed to the resolved sender");

    // A handshake is the one control message that may arrive from a peer we have not
    // inserted yet -- it is what triggers the insert -- so it must be processed even
    // though its source address does not resolve.
    queue_reset();
    pending_completions.push_back(
        make_recv(NIXL_LIBFABRIC_MSG_HANDSHAKE, FI_ADDR_NOTAVAIL, std::string(32, 'h')));
    st = rail.progressCompletionQueue();
    TEST_ASSERT(st == NIXL_SUCCESS, "handshake from an unknown source is still processed");
    TEST_ASSERT(handshakes.size() == 1, "handshake callback ran without source resolution");

    // A notification from an unknown source cannot be correlated, so it is dropped.
    queue_reset();
    notif_senders.clear();
    pending_completions.push_back(
        make_recv(NIXL_LIBFABRIC_MSG_NOTIFICTION, FI_ADDR_NOTAVAIL, std::string(64, 'n')));
    st = rail.progressCompletionQueue();
    TEST_ASSERT(st == NIXL_SUCCESS, "unattributable notification is dropped, not an error");
    TEST_ASSERT(notif_senders.empty(), "unattributable notification is not delivered");

    rail.setNotificationCallback(nullptr);
    rail.setHandshakeCallback(nullptr);
    return 0;
}

static int
testProviderWithoutFiSource() {
    NIXL_INFO << "  testProviderWithoutFiSource";

    // Both fi_getinfo attempts fail with FI_ENODATA. The rail must name FI_SOURCE as the
    // cause instead of reporting a generic fi_getinfo failure.
    mock_reject_fi_source = true;
    std::string what;
    try {
        nixlLibfabricRail rail("efa_0", "efa", 0, FI_HMEM_SYSTEM);
    }
    catch (const std::runtime_error &e) {
        what = e.what();
    }
    mock_reject_fi_source = false;

    TEST_ASSERT(what.find("FI_SOURCE not supported") != std::string::npos,
                "provider without FI_SOURCE is reported as such, got: '" + what + "'");
    return 0;
}

// Builds a rail and returns the hints->caps of every fi_getinfo call it made.
static std::vector<uint64_t>
hintCapsOfRailInit() {
    recorded_hint_caps.clear();
    record_hint_caps = true;
    nixlLibfabricRail rail("efa_0", "efa", 0, FI_HMEM_SYSTEM);
    record_hint_caps = false;
    return recorded_hint_caps;
}

static int
testFiSourceRequested() {
    NIXL_INFO << "  testFiSourceRequested";

    // Both fi_getinfo paths must request FI_SOURCE. If one drops it, a provider can return
    // info without it, and every completion then reports FI_ADDR_NOTAVAIL as its source.
    std::vector<uint64_t> caps = hintCapsOfRailInit();
    TEST_ASSERT(caps.size() == 1, "FI_HMEM path: expected 1 fi_getinfo call");
    TEST_ASSERT((caps[0] & FI_HMEM) && (caps[0] & FI_SOURCE),
                "FI_HMEM path requests FI_HMEM and FI_SOURCE");

    mock_reject_fi_hmem = true;
    caps = hintCapsOfRailInit();
    mock_reject_fi_hmem = false;
    TEST_ASSERT(caps.size() == 2, "retry path: expected 2 fi_getinfo calls");
    TEST_ASSERT(!(caps[1] & FI_HMEM) && (caps[1] & FI_SOURCE),
                "retry path requests FI_SOURCE without FI_HMEM");
    return 0;
}

int
main() {
    NIXL_INFO << "=== Source Address Resolution Test ===";
    NIXL_INFO << "Using mock stubs (__wrap_fi_getinfo, fi_av_insert, fi_cq_readfrom, ...)";

    // Route CQ reads to the fake queue. Done before the manager is built so that the
    // rail's own initialisation sees a consistent CQ.
    cq_ops_stub.size = sizeof(fi_ops_cq);
    cq_ops_stub.readfrom = fi_cq_readfrom_stub;
    cq_ops_stub.readerr = fi_cq_readerr_stub;

    nixlLibfabricRailManager mgr(0);
    TEST_ASSERT(mgr.getNumRails() == NUM_FAKE_RAILS,
                "expected " + std::to_string(NUM_FAKE_RAILS) + " rails");

    int res;
    if ((res = testInsertAndResolve(mgr)) != 0) {
        return res;
    }
    if ((res = testUnresolvableAddresses(mgr)) != 0) {
        return res;
    }
    if ((res = testSparseAndHighAgentIndex(mgr)) != 0) {
        return res;
    }
    if ((res = testRemapAndRemove(mgr)) != 0) {
        return res;
    }
    if ((res = testInsertAllAddressesCrossProduct(mgr)) != 0) {
        return res;
    }
    if ((res = testRemoteWriteCompletionAttribution(mgr)) != 0) {
        return res;
    }
    if ((res = testRecvCompletionAttribution(mgr)) != 0) {
        return res;
    }
    if ((res = testFiSourceRequested()) != 0) {
        return res;
    }
    if ((res = testProviderWithoutFiSource()) != 0) {
        return res;
    }

    NIXL_INFO << "=== All source address resolution tests PASSED ===";
    return 0;
}
