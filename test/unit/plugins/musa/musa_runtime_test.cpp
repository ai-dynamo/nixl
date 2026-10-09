// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "test_support.h"
#include "musa/musa_runtime.h"
#include "musa/musa_policy.h"
#include "musa_runtime_api.h"

int
main() {
    auto runtime = nixl::musa::makeRuntime();
    nixl::musa::MemoryPolicy policy(runtime);
    NIXL_TEST_CHECK(runtime->deviceCount() == 2);
    auto attributes = runtime->pointerAttributes(4096);
    NIXL_TEST_CHECK(attributes && attributes->device_memory && attributes->device == 1);
    fake_attributes.type = musaMemoryTypeHost;
    NIXL_TEST_CHECK(!runtime->pointerAttributes(4096)->device_memory);
    fake_attributes.type = musaMemoryTypeManaged;
    NIXL_TEST_CHECK(!runtime->pointerAttributes(4096)->device_memory);
    fake_status = musaErrorInvalidValue;
    NIXL_TEST_CHECK(!runtime->pointerAttributes(4096));
    nixlBlobDesc invalid;
    invalid.addr = 4096;
    invalid.len = 64;
    invalid.devId = 1;
    NIXL_TEST_CHECK(policy.validateBeforeMap(invalid, VRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    fake_status = musaSuccess;
    fake_device_count = 0;
    NIXL_TEST_CHECK(runtime->deviceCount() == 0);
    fake_status = musaErrorUnknown;
    checkThrows([&] { (void)runtime->deviceCount(); });
    checkThrows([&] { (void)runtime->pointerAttributes(4096); });
    std::cout << "MUSA runtime adapter + fake SDK checks passed (not hardware)\n";
}
