/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_oob_endpoint.h"

#include <cassert>
#include <limits>
#include <stdexcept>

template<typename Fn>
void
assertInvalid(Fn &&fn) {
    bool rejected = false;
    try {
        fn();
    }
    catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

int
main() {
    assert(!isValidGpunetioAgentNameSize(0));
    assert(isValidGpunetioAgentNameSize(1));
    assert(isValidGpunetioAgentNameSize(GPUNETIO_MAX_AGENT_NAME_SIZE));
    assert(!isValidGpunetioAgentNameSize(GPUNETIO_MAX_AGENT_NAME_SIZE + 1));
    assert(!isValidGpunetioAgentNameSize(std::numeric_limits<std::size_t>::max()));

    assert(parseGpunetioOobPort("") == GPUNETIO_DEFAULT_OOB_PORT);
    assert(parseGpunetioOobPort("1") == 1);
    assert(parseGpunetioOobPort("65535") == 65535);

    const auto legacy = parseGpunetioOobEndpoint("192.0.2.1");
    assert(legacy.ipv4 == "192.0.2.1");
    assert(legacy.port == GPUNETIO_DEFAULT_OOB_PORT);
    assert(formatGpunetioOobEndpoint(legacy.ipv4, legacy.port) == "192.0.2.1");

    const auto configured = parseGpunetioOobEndpoint("198.51.100.2:16544");
    assert(configured.ipv4 == "198.51.100.2");
    assert(configured.port == 16544);
    assert(formatGpunetioOobEndpoint(configured.ipv4, configured.port) == "198.51.100.2:16544");

    for (const auto *value : {"0", "-1", "+1", "65536", "12x", " 6544", "6544 "}) {
        assertInvalid([=] { parseGpunetioOobPort(value); });
    }

    for (const auto *value : {"",
                              "192.0.2",
                              "256.0.0.1",
                              "192.0.2.1:",
                              "192.0.2.1:0",
                              "192.0.2.1:65536",
                              "192.0.2.1:12x",
                              "192.0.2.1:6544:1",
                              "[::1]:6544"}) {
        assertInvalid([=] { parseGpunetioOobEndpoint(value); });
    }
    assertInvalid([] { parseGpunetioOobEndpoint(std::string("192.0.2.1\0:6544", 16)); });

    return 0;
}
