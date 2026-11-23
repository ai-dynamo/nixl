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
 * Unit test for the striping threshold of nixlLibfabricRailManager: the default, a threshold
 * set through setStripingThreshold(), rejected values, and the striping_threshold parameter
 * with its NIXL_LIBFABRIC_STRIPING_THRESHOLD environment override, each checked through
 * usesStriping().
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <cstdlib>
#include <iostream>
#include <string>

// Number of fake EFA devices to create
static const size_t NUM_FAKE_RAILS = 4;

static const char *THRESHOLD_ENV = "NIXL_LIBFABRIC_STRIPING_THRESHOLD";

// --- Unconditional __wrap_* functions ---

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

// True when usesStriping() switches from one rail to striping exactly at threshold
static bool
stripesFrom(const nixlLibfabricRailManager &mgr, size_t threshold) {
    return !mgr.usesStriping(threshold - 1, NUM_FAKE_RAILS) &&
        mgr.usesStriping(threshold, NUM_FAKE_RAILS);
}

// Loads the threshold the way nixlLibfabricEngine's constructor does
static void
applyConfiguredThreshold(nixlLibfabricRailManager &mgr, const nixl_b_params_t &params) {
    size_t threshold = NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD;
    LibfabricUtils::getCustomIntParam(params, "striping_threshold", threshold);
    mgr.setStripingThreshold(threshold);
}

// --- Tests ---

static int
testDefault(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testDefault";

    TEST_ASSERT(stripesFrom(mgr, NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD),
                "default threshold stripes from 128 KiB");
    TEST_ASSERT(!mgr.usesStriping(16ull << 20, 1), "a single rail never stripes");

    return 0;
}

static int
testSetThreshold(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testSetThreshold";

    mgr.setStripingThreshold(64 * 1024);
    TEST_ASSERT(stripesFrom(mgr, 64 * 1024), "custom threshold stripes from 64 KiB");

    mgr.setStripingThreshold(NUM_FAKE_RAILS);
    TEST_ASSERT(stripesFrom(mgr, NUM_FAKE_RAILS), "threshold equal to the rail count accepted");

    mgr.setStripingThreshold(NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD);
    return 0;
}

static int
testRejectBelowRailCount(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testRejectBelowRailCount";

    mgr.setStripingThreshold(1024 * 1024);

    mgr.setStripingThreshold(0);
    TEST_ASSERT(stripesFrom(mgr, 1024 * 1024), "threshold 0 rejected, 1 MiB kept");

    mgr.setStripingThreshold(NUM_FAKE_RAILS - 1);
    TEST_ASSERT(stripesFrom(mgr, 1024 * 1024), "threshold below the rail count rejected");

    mgr.setStripingThreshold(NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD);
    return 0;
}

static int
testConfiguredThreshold(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testConfiguredThreshold";

    unsetenv(THRESHOLD_ENV);
    const nixl_b_params_t no_params;
    const nixl_b_params_t params = {{"striping_threshold", "65536"}};

    applyConfiguredThreshold(mgr, no_params);
    TEST_ASSERT(stripesFrom(mgr, NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD),
                "no parameter and no env: default threshold");

    applyConfiguredThreshold(mgr, params);
    TEST_ASSERT(stripesFrom(mgr, 65536), "backend parameter applied");

    TEST_ASSERT(setenv(THRESHOLD_ENV, "1048576", 1) == 0, "setenv");
    applyConfiguredThreshold(mgr, params);
    TEST_ASSERT(stripesFrom(mgr, 1048576), "env var overrides the backend parameter");

    TEST_ASSERT(setenv(THRESHOLD_ENV, "64KiB", 1) == 0, "setenv");
    applyConfiguredThreshold(mgr, params);
    TEST_ASSERT(stripesFrom(mgr, NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD),
                "malformed env value: default threshold");

    // The threshold is the default again here, as in a newly constructed engine
    TEST_ASSERT(setenv(THRESHOLD_ENV, "0", 1) == 0, "setenv");
    applyConfiguredThreshold(mgr, params);
    TEST_ASSERT(stripesFrom(mgr, NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD),
                "env value 0 rejected: default threshold kept");

    unsetenv(THRESHOLD_ENV);
    mgr.setStripingThreshold(NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD);
    return 0;
}

int
main() {
    NIXL_INFO << "=== Striping Threshold Test ===";
    NIXL_INFO << "Using mock stubs (__wrap_fi_getinfo, __wrap_fi_fabric, etc.)";

    // Construct rail manager with mocked hardware (NUM_FAKE_RAILS rails), as the engine does
    nixlLibfabricRailManager mgr(NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD);
    TEST_ASSERT(mgr.getNumRails() == NUM_FAKE_RAILS,
                "expected " + std::to_string(NUM_FAKE_RAILS) + " rails");

    int res;
    if ((res = testDefault(mgr)) != 0) {
        return res;
    }
    if ((res = testSetThreshold(mgr)) != 0) {
        return res;
    }
    if ((res = testRejectBelowRailCount(mgr)) != 0) {
        return res;
    }
    if ((res = testConfiguredThreshold(mgr)) != 0) {
        return res;
    }

    NIXL_INFO << "=== All striping threshold tests PASSED ===";
    return 0;
}
