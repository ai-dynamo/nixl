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
    st = mgr.deregisterMemory(selected_rails, mr_list, DRAM_SEG, 0);
    TEST_ASSERT(st == NIXL_SUCCESS, "first deregisterMemory succeeded");
    TEST_ASSERT(mgr.getActiveRailCount() == NUM_FAKE_RAILS,
                "all rails still active after first deregister");

    // Deregister second — refcounts drop to 0, rails removed
    st = mgr.deregisterMemory(selected_rails2, mr_list2, DRAM_SEG, 0);
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
    st = mgr.deregisterMemory(selected_rails, mr_list, DRAM_SEG, 0);
    fi_mr_self_ops_stub.close = fi_mr_close_stub;

    TEST_ASSERT(st != NIXL_SUCCESS, "deregisterMemory should fail");
    TEST_ASSERT(mgr.getActiveRailCount() == NUM_FAKE_RAILS,
                "all rails still active after failed deregister");

    // Clean up: successful deregister to reset state
    st = mgr.deregisterMemory(selected_rails, mr_list, DRAM_SEG, 0);
    mgr.clearActiveRails();
    return 0;
}

static int
testVramDeviceSelection(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testVramDeviceSelection";

    nixlLibfabricRail &rail = mgr.getRail(0);
    const int none = nixlLibfabricRail::VRAM_DEVICE_NONE;
    TEST_ASSERT(rail.getVramDevice() == none, "no VRAM device before any registration");

    rail.addVramDevice(-1);
    TEST_ASSERT(rail.getVramDevice() == none, "negative device id ignored");

    rail.addVramDevice(3);
    TEST_ASSERT(rail.getVramDevice() == 3, "single device selected");

    rail.addVramDevice(1);
    TEST_ASSERT(rail.getVramDevice() == 1, "tie between devices 1 and 3 keeps the lower id");

    rail.addVramDevice(3);
    TEST_ASSERT(rail.getVramDevice() == 3, "device with more live registrations wins");

    rail.removeVramDevice(3);
    TEST_ASSERT(rail.getVramDevice() == 1, "back to a tie after a deregistration: lower id");

    rail.removeVramDevice(1);
    TEST_ASSERT(rail.getVramDevice() == 3, "device 1 fully deregistered: only device 3 left");

    rail.removeVramDevice(7);
    TEST_ASSERT(rail.getVramDevice() == 3, "removing an unknown device is a no-op");

    rail.removeVramDevice(3);
    TEST_ASSERT(rail.getVramDevice() == none, "no VRAM device after the last deregistration");

    rail.addVramDevice(2);
    TEST_ASSERT(rail.getVramDevice() == 2, "a new device is picked up after earlier ones are gone");
    rail.removeVramDevice(2);
    TEST_ASSERT(rail.getVramDevice() == none, "state cleared for later tests");

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
    if ((res = testDeregisterFailureKeepsRailActive(mgr)) != 0) return res;
    if ((res = testVramDeviceSelection(mgr)) != 0) {
        return res;
    }

    NIXL_INFO << "=== All rail active refcount tests PASSED ===";
    return 0;
}
