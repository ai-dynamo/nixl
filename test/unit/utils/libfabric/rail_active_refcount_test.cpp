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
 * Unit test for active rail reference counting in nixlLibfabricRailManager.
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <cassert>
#include <iostream>
#include <set>

// Number of fake EFA devices to create
static const size_t NUM_FAKE_RAILS = 4;

// --- Unconditional __wrap_* functions (no isTesting gate needed) ---

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
    // Build a linked list of NUM_FAKE_RAILS fake EFA devices
    *info = mock_fi_info_chain(NUM_FAKE_RAILS, 100ull * NIXL_LIBFABRIC_GIGA);
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

// --- Tests ---

static int
testIncDecBasic(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testIncDecBasic";

    // inc rail 0 twice
    mgr.incRailActive(0);
    mgr.incRailActive(0);
    TEST_ASSERT(mgr.getActiveRailCount() == 1, "one rail active after two incs on same rail");

    // dec once — refcount drops to 1, rail still active
    mgr.decRailActive(0);
    TEST_ASSERT(mgr.getActiveRailCount() == 1, "rail still active after one dec");

    // dec again — refcount drops to 0, rail removed
    mgr.decRailActive(0);
    TEST_ASSERT(mgr.getActiveRailCount() == 0, "rail removed after second dec");

    return 0;
}

static int
testDecNonExistent(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testDecNonExistent";

    // dec on a rail that was never inc'd — should not crash
    mgr.decRailActive(1);
    TEST_ASSERT(mgr.getActiveRailCount() == 0, "count unchanged after dec on non-existent rail");

    return 0;
}

static int
testIncInvalidRailId(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testIncInvalidRailId";

    // inc with rail_id >= rails_.size() — should be a no-op
    mgr.incRailActive(NUM_FAKE_RAILS + 10);
    TEST_ASSERT(mgr.getActiveRailCount() == 0, "no-op for invalid rail id");

    return 0;
}

static int
testClearActiveRails(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testClearActiveRails";

    mgr.incRailActive(0);
    mgr.incRailActive(1);
    mgr.incRailActive(2);
    TEST_ASSERT(mgr.getActiveRailCount() == 3, "three rails active");

    mgr.clearActiveRails();
    TEST_ASSERT(mgr.getActiveRailCount() == 0, "all rails cleared");

    return 0;
}

static int
testMultipleRailsIndependent(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testMultipleRailsIndependent";

    // rail 0: refcount 3, rail 1: refcount 1, rail 2: refcount 2
    mgr.incRailActive(0);
    mgr.incRailActive(0);
    mgr.incRailActive(0);
    mgr.incRailActive(1);
    mgr.incRailActive(2);
    mgr.incRailActive(2);
    TEST_ASSERT(mgr.getActiveRailCount() == 3, "three distinct rails active");

    // remove rail 1 entirely
    mgr.decRailActive(1);
    TEST_ASSERT(mgr.getActiveRailCount() == 2, "rail 1 removed, two rails remain");

    // dec rail 0 once — still active (refcount 2)
    mgr.decRailActive(0);
    TEST_ASSERT(mgr.getActiveRailCount() == 2, "rail 0 still active after one dec");

    // clean up
    mgr.clearActiveRails();
    return 0;
}

static int
testRegisterDeregisterRefcount(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testRegisterDeregisterRefcount";

    // Allocate a DRAM buffer to register
    char buf[4096] = {};
    std::vector<struct fid_mr *> mr_list;
    std::vector<uint64_t> key_list;
    std::vector<size_t> selected_rails;

    TEST_ASSERT(mgr.getActiveRailCount() == 0, "no active rails before register");

    nixl_status_t st =
        mgr.registerMemory(buf, sizeof(buf), DRAM_SEG, 0, "", mr_list, key_list, selected_rails);
    TEST_ASSERT(st == NIXL_SUCCESS, "registerMemory succeeded");
    TEST_ASSERT(mgr.getActiveRailCount() == NUM_FAKE_RAILS, "all rails active after DRAM register");

    // Register a second time — refcounts should increase but active count stays the same
    std::vector<struct fid_mr *> mr_list2;
    std::vector<uint64_t> key_list2;
    std::vector<size_t> selected_rails2;
    st =
        mgr.registerMemory(buf, sizeof(buf), DRAM_SEG, 0, "", mr_list2, key_list2, selected_rails2);
    TEST_ASSERT(st == NIXL_SUCCESS, "second registerMemory succeeded");
    TEST_ASSERT(mgr.getActiveRailCount() == NUM_FAKE_RAILS,
                "active count unchanged after second register");

    // Deregister first — rails still active (refcount > 0)
    st = mgr.deregisterMemory(selected_rails, mr_list);
    TEST_ASSERT(st == NIXL_SUCCESS, "first deregisterMemory succeeded");
    TEST_ASSERT(mgr.getActiveRailCount() == NUM_FAKE_RAILS,
                "all rails still active after first deregister");

    // Deregister second — refcounts drop to 0, rails removed
    st = mgr.deregisterMemory(selected_rails2, mr_list2);
    TEST_ASSERT(st == NIXL_SUCCESS, "second deregisterMemory succeeded");
    TEST_ASSERT(mgr.getActiveRailCount() == 0, "no active rails after full deregister");

    return 0;
}

static int
fi_mr_close_fail(struct fid * /*fid*/) {
    return -FI_EIO;
}

static int
testDeregisterFailureKeepsRailActive(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testDeregisterFailureKeepsRailActive";

    char buf[4096] = {};
    std::vector<struct fid_mr *> mr_list;
    std::vector<uint64_t> key_list;
    std::vector<size_t> selected_rails;

    nixl_status_t st =
        mgr.registerMemory(buf, sizeof(buf), DRAM_SEG, 0, "", mr_list, key_list, selected_rails);
    TEST_ASSERT(st == NIXL_SUCCESS, "registerMemory succeeded");
    TEST_ASSERT(mgr.getActiveRailCount() == NUM_FAKE_RAILS, "all rails active after register");

    // Make fi_close fail so deregisterMemory fails
    fi_mr_self_ops_stub.close = fi_mr_close_fail;
    st = mgr.deregisterMemory(selected_rails, mr_list);
    fi_mr_self_ops_stub.close = fi_mr_close_stub;

    TEST_ASSERT(st != NIXL_SUCCESS, "deregisterMemory should fail");
    TEST_ASSERT(mgr.getActiveRailCount() == NUM_FAKE_RAILS,
                "all rails still active after failed deregister");

    // Clean up: successful deregister to reset state
    st = mgr.deregisterMemory(selected_rails, mr_list);
    mgr.clearActiveRails();
    return 0;
}

// --- Same address, different registration parameters ---

// Closed MRs are recorded instead of freed, so a second close of the same MR is caught
// rather than touching freed memory.
static std::set<struct fid *> closed_mrs;
static bool double_close = false;

static int
fi_mr_close_recording(struct fid *fid) {
    if (!closed_mrs.insert(fid).second) {
        double_close = true;
    }
    return 0;
}

static int
testSameAddressDifferentLength(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testSameAddressDifferentLength";

    char buf[4096] = {};
    std::vector<struct fid_mr *> mr_full, mr_half, mr_full2;
    std::vector<uint64_t> key_full, key_half, key_full2;
    std::vector<size_t> rails_full, rails_half, rails_full2;

    nixl_status_t st =
        mgr.registerMemory(buf, sizeof(buf), DRAM_SEG, 0, "", mr_full, key_full, rails_full);
    TEST_ASSERT(st == NIXL_SUCCESS, "registerMemory(buf, 4096) succeeded");

    // Same address, different length, while the first registration is held
    st = mgr.registerMemory(buf, sizeof(buf) / 2, DRAM_SEG, 0, "", mr_half, key_half, rails_half);
    TEST_ASSERT(st == NIXL_SUCCESS, "registerMemory(buf, 2048) at a held address succeeded");
    TEST_ASSERT(rails_half == rails_full && mr_half.size() == mr_full.size(),
                "both registrations selected the same rails");
    for (size_t i = 0; i < mr_full.size(); ++i) {
        TEST_ASSERT(mr_half[i] != mr_full[i], "different-length registration got its own MR");
    }

    // Same address and length while held: the first MR is shared
    st = mgr.registerMemory(buf, sizeof(buf), DRAM_SEG, 0, "", mr_full2, key_full2, rails_full2);
    TEST_ASSERT(st == NIXL_SUCCESS, "second registerMemory(buf, 4096) succeeded");
    TEST_ASSERT(mr_full2 == mr_full, "same-length registration shares the first MR");

    // Deregister all three: every call succeeds and each distinct MR is closed exactly once
    closed_mrs.clear();
    double_close = false;
    fi_mr_self_ops_stub.close = fi_mr_close_recording;
    const nixl_status_t st_full = mgr.deregisterMemory(rails_full, mr_full);
    const nixl_status_t st_half = mgr.deregisterMemory(rails_half, mr_half);
    const nixl_status_t st_full2 = mgr.deregisterMemory(rails_full2, mr_full2);
    fi_mr_self_ops_stub.close = fi_mr_close_stub;
    const size_t num_closed = closed_mrs.size();
    for (struct fid *fid : closed_mrs) {
        fi_mr_close_stub(fid);
    }
    closed_mrs.clear();

    TEST_ASSERT(st_full == NIXL_SUCCESS && st_half == NIXL_SUCCESS && st_full2 == NIXL_SUCCESS,
                "all three deregisterMemory calls succeeded");
    TEST_ASSERT(!double_close, "no MR closed twice");
    TEST_ASSERT(num_closed == 2 * mr_full.size(), "both distinct MRs closed on every rail");
    TEST_ASSERT(mgr.getActiveRailCount() == 0, "no active rails after full deregister");
    return 0;
}

// Deterministic stand-in for two threads registering one address with different lengths:
// the outer registration's fi_mr_regattr() first runs the competing registration on the same
// rail, so it lands between the outer registration's cache check and its insert.
static const nixlLibfabricRail *race_rail = nullptr;
static void *race_buf = nullptr;
static size_t race_len = 0;
static struct fid_mr *race_mr = nullptr;
static uint64_t race_key = 0;
static nixl_status_t race_status = NIXL_ERR_BACKEND;
static struct fid_mr *race_outer_mr = nullptr; // fresh MR fi_mr_regattr gave the outer call

static int
fi_mr_regattr_racing(struct fid *fid,
                     const struct fi_mr_attr *attr,
                     uint64_t flags,
                     struct fid_mr **mr) {
    if (race_rail != nullptr) {
        const nixlLibfabricRail *rail = race_rail;
        race_rail = nullptr; // the competing registration takes the plain stub
        race_status = rail->registerMemory(
            race_buf, race_len, DRAM_SEG, 0, FI_HMEM_SYSTEM, &race_mr, &race_key);
        const int ret = fi_mr_regattr_stub(fid, attr, flags, mr);
        race_outer_mr = *mr;
        return ret;
    }
    return fi_mr_regattr_stub(fid, attr, flags, mr);
}

static int
testConcurrentDifferentLengthRegistration(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testConcurrentDifferentLengthRegistration";

    // static: an address no other case registers, so a registration left behind by a failed
    // case cannot turn the outer registration into a cache hit
    static char buf[4096] = {};
    const nixlLibfabricRail &rail = mgr.getRail(0);
    race_rail = &rail;
    race_buf = buf;
    race_len = sizeof(buf) / 2;

    struct fid_mr *mr = nullptr;
    uint64_t key = 0;
    fi_mr_ops_stub.regattr = fi_mr_regattr_racing;
    const nixl_status_t st =
        rail.registerMemory(buf, sizeof(buf), DRAM_SEG, 0, FI_HMEM_SYSTEM, &mr, &key);
    fi_mr_ops_stub.regattr = fi_mr_regattr_stub;

    TEST_ASSERT(race_rail == nullptr && race_status == NIXL_SUCCESS,
                "competing registration (buf, 2048) ran and succeeded");
    TEST_ASSERT(st == NIXL_SUCCESS, "registration (buf, 4096) succeeded");
    TEST_ASSERT(mr != race_mr, "registration that lost the race kept its own MR");

    TEST_ASSERT(rail.deregisterMemory(race_mr) == NIXL_SUCCESS, "deregister the competing MR");
    TEST_ASSERT(rail.deregisterMemory(mr) == NIXL_SUCCESS, "deregister the outer MR");
    return 0;
}

// Same-parameter race: the competing registration matches the outer one, so the outer call
// must close its own fresh MR exactly once and share the competitor's tracked MR.
static int
testConcurrentSameParameterRegistration(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testConcurrentSameParameterRegistration";

    // static: an address no other case registers
    static char buf[4096] = {};
    const nixlLibfabricRail &rail = mgr.getRail(0);
    race_rail = &rail;
    race_buf = buf;
    race_len = sizeof(buf);
    race_mr = nullptr;
    race_outer_mr = nullptr;
    race_status = NIXL_ERR_BACKEND;

    struct fid_mr *mr = nullptr;
    uint64_t key = 0;
    closed_mrs.clear();
    double_close = false;
    fi_mr_self_ops_stub.close = fi_mr_close_recording;
    fi_mr_ops_stub.regattr = fi_mr_regattr_racing;
    const nixl_status_t st =
        rail.registerMemory(buf, sizeof(buf), DRAM_SEG, 0, FI_HMEM_SYSTEM, &mr, &key);
    fi_mr_ops_stub.regattr = fi_mr_regattr_stub;
    const bool hook_ran = race_rail == nullptr && race_status == NIXL_SUCCESS &&
        race_mr != nullptr && race_outer_mr != nullptr;
    const bool outer_closed_once =
        hook_ran && closed_mrs.size() == 1 && closed_mrs.count(&race_outer_mr->fid) == 1;

    const nixl_status_t st_first = rail.deregisterMemory(mr);
    const bool open_after_first = hook_ran && closed_mrs.count(&race_mr->fid) == 0;
    const nixl_status_t st_second = rail.deregisterMemory(race_mr);
    const bool closed_after_second = hook_ran && closed_mrs.count(&race_mr->fid) == 1;
    fi_mr_self_ops_stub.close = fi_mr_close_stub;
    const bool no_double_close = !double_close;
    for (struct fid *fid : closed_mrs) {
        fi_mr_close_stub(fid);
    }
    closed_mrs.clear();

    TEST_ASSERT(hook_ran, "competing registration (buf, 4096) ran and succeeded");
    TEST_ASSERT(st == NIXL_SUCCESS, "registration (buf, 4096) succeeded");
    TEST_ASSERT(mr == race_mr, "registration that lost the race returns the competitor's MR");
    TEST_ASSERT(outer_closed_once, "the outer registration's fresh MR is closed exactly once");
    TEST_ASSERT(st_first == NIXL_SUCCESS && open_after_first,
                "first deregister succeeds and leaves the shared MR open");
    TEST_ASSERT(st_second == NIXL_SUCCESS && closed_after_second,
                "second deregister succeeds and closes the shared MR");
    TEST_ASSERT(no_double_close, "no MR closed twice");
    return 0;
}

int
main() {
    NIXL_INFO << "=== Rail Active Refcount Test ===";
    NIXL_INFO << "Using mock stubs (__wrap_fi_getinfo, __wrap_fi_fabric, etc.)";

    // Construct rail manager with mocked hardware (NUM_FAKE_RAILS rails)
    nixlLibfabricRailManager mgr(0);
    TEST_ASSERT(mgr.getNumRails() == NUM_FAKE_RAILS,
                "expected " + std::to_string(NUM_FAKE_RAILS) + " rails");

    int res;
    if ((res = testIncDecBasic(mgr)) != 0) return res;
    if ((res = testDecNonExistent(mgr)) != 0) return res;
    if ((res = testIncInvalidRailId(mgr)) != 0) return res;
    if ((res = testClearActiveRails(mgr)) != 0) return res;
    if ((res = testMultipleRailsIndependent(mgr)) != 0) return res;
    if ((res = testRegisterDeregisterRefcount(mgr)) != 0) return res;
    // These cases all run, so one failing does not hide another
    const int diff_len_res = testSameAddressDifferentLength(mgr);
    const int race_res = testConcurrentDifferentLengthRegistration(mgr);
    const int same_race_res = testConcurrentSameParameterRegistration(mgr);
    if (diff_len_res != 0 || race_res != 0 || same_race_res != 0) {
        return 1;
    }
    if ((res = testDeregisterFailureKeepsRailActive(mgr)) != 0) return res;

    NIXL_INFO << "=== All rail active refcount tests PASSED ===";
    return 0;
}
