// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "test_support.h"
#include "ucx/memory_policy.h"

#include <array>
#include <limits>

int
main() {
    using namespace nixl::ucx;
    const std::array<const char *, 5> names = {"host", "cuda", "rocm", "musa", "unknown"};
    NIXL_TEST_CHECK(findDeviceMemoryType(names, "musa", 1ULL << 3, 0) == 3);
    NIXL_TEST_CHECK(!findDeviceMemoryType(names, "musa", 1ULL << 2, 0));
    NIXL_TEST_CHECK(!findDeviceMemoryType(names, "absent", ~0ULL, 0));
    NIXL_TEST_CHECK(!findDeviceMemoryType(names, "host", ~0ULL, 0));
    NIXL_TEST_CHECK(!findDeviceMemoryType(names, "", ~0ULL, 0));
    NIXL_TEST_CHECK(!findDeviceMemoryType(names, "unknown", ~0ULL, 0));
    std::array<const char *, 65> too_many{};
    too_many[64] = "musa";
    NIXL_TEST_CHECK(!findDeviceMemoryType(too_many, "musa", ~0ULL, 0));
    NIXL_TEST_CHECK(registeredMemoryTypeMatches(DRAM_SEG, 0, 0, 3));
    NIXL_TEST_CHECK(!registeredMemoryTypeMatches(DRAM_SEG, 3, 0, 3));
    NIXL_TEST_CHECK(registeredMemoryTypeMatches(VRAM_SEG, 3, 0, 3));
    NIXL_TEST_CHECK(!registeredMemoryTypeMatches(VRAM_SEG, 0, 0, 3));
    NIXL_TEST_CHECK(!registeredMemoryTypeMatches(VRAM_SEG, 1, 0, 3));
    NIXL_TEST_CHECK(!registeredMemoryTypeMatches(VRAM_SEG, 2, 0, 3));
    NIXL_TEST_CHECK(!registeredMemoryTypeMatches(VRAM_SEG, 4, 0, 3));
    NIXL_TEST_CHECK(!registeredMemoryTypeMatches(FILE_SEG, 3, 0, 3));
    NIXL_TEST_CHECK(validMemoryRange(4096, 64));
    NIXL_TEST_CHECK(!validMemoryRange(0, 64));
    NIXL_TEST_CHECK(!validMemoryRange(4096, 0));
    NIXL_TEST_CHECK(!validMemoryRange(std::numeric_limits<uintptr_t>::max() - 3, 8));
    nixlBlobDesc remote;
    remote.addr = 4096;
    remote.len = 64;
    // Remote device ordinals must not be checked against the local runtime.
    remote.devId = std::numeric_limits<uint64_t>::max();
    NIXL_TEST_CHECK(validateMemoryDescriptor(remote, VRAM_SEG) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(validateMemoryDescriptor(remote, DRAM_SEG) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(validateMemoryDescriptor(remote, FILE_SEG) == NIXL_ERR_NOT_SUPPORTED);
    NIXL_TEST_CHECK(validateMemoryDescriptor(remote, OBJ_SEG) == NIXL_ERR_NOT_SUPPORTED);
    NIXL_TEST_CHECK(validateMemoryDescriptor(remote, BLK_SEG) == NIXL_ERR_NOT_SUPPORTED);
    remote.addr = 0;
    NIXL_TEST_CHECK(validateMemoryDescriptor(remote, VRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    remote.addr = std::numeric_limits<uintptr_t>::max() - 3;
    NIXL_TEST_CHECK(validateMemoryDescriptor(remote, DRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    std::cout << "memory policy checks passed\n";
}
