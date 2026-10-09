// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "test_support.h"
#include "musa/musa_policy.h"

#include <limits>

class FakeRuntime final : public nixl::musa::Runtime {
public:
    int count = 2;
    nixl::musa::PointerAttributes attributes{true, 1};
    bool fail = false;
    mutable int queries = 0;

    int
    deviceCount() const override {
        return count;
    }

    std::optional<nixl::musa::PointerAttributes>
    pointerAttributes(uintptr_t) const override {
        ++queries;
        if (fail) {
            throw std::runtime_error("runtime query failed");
        }
        return attributes;
    }
};

int
main() {
    auto runtime = std::make_shared<FakeRuntime>();
    nixl::musa::MemoryPolicy policy(runtime);
    nixlBlobDesc desc;
    desc.addr = 4096;
    desc.len = 64;
    desc.devId = 1;
    NIXL_TEST_CHECK(policy.deviceMemoryType() == "musa");
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, VRAM_SEG) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, DRAM_SEG) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(runtime->queries == 1);
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, FILE_SEG) == NIXL_ERR_NOT_SUPPORTED);
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, OBJ_SEG) == NIXL_ERR_NOT_SUPPORTED);
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, BLK_SEG) == NIXL_ERR_NOT_SUPPORTED);
    desc.devId = 0;
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, VRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    desc.devId = std::numeric_limits<uint64_t>::max();
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, VRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    desc.devId = 1;
    runtime->attributes.device_memory = false;
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, VRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    runtime->attributes = {true, -1};
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, VRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    runtime->attributes = {true, 1};
    runtime->fail = true;
    checkThrows([&] { (void)policy.validateBeforeMap(desc, VRAM_SEG); });
    runtime->fail = false;
    desc.len = 0;
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, VRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    desc.len = 64;
    desc.addr = std::numeric_limits<uintptr_t>::max() - 3;
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, VRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    desc.addr = 0;
    NIXL_TEST_CHECK(policy.validateBeforeMap(desc, DRAM_SEG) == NIXL_ERR_INVALID_PARAM);
    runtime->count = 0;
    checkThrows([&] { nixl::musa::MemoryPolicy no_device(runtime); });
    checkThrows([] { nixl::musa::MemoryPolicy no_runtime(nullptr); });
    std::cout << "MUSA policy checks passed\n";
}
