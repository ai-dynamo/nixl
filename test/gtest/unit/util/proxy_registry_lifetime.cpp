/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <atomic>
#include <array>
#include <cstring>
#include <thread>
#include <vector>
#include "device/proxy/proxy_registry.h"
#include "mocks/proxy_mocks.h"

using namespace gtest::proxy_mocks;

TEST(ProxyRegistryLifetimeTest, ConcurrentGrowthAndUnusedRetirementPreserveLiveTokens) {
    MockDeviceOps allocator;
    DummyBackendMD md;
    nixlProxyDeviceContextData context;
    nixl::proxyMemViewRegistry registry(allocator, &context);
    auto local = makeLocalDlist(0x1000, 64, 0, &md);
    nixl_remote_meta_dlist_t remote(VRAM_SEG);
    remote.addDesc(makeRemoteDesc("peer", 0x2000, 64, 0, &md));
    nixlMemViewH src = nullptr, dst = nullptr;
    ASSERT_EQ(registry.prepLocal(local, src), NIXL_SUCCESS);
    ASSERT_EQ(registry.prepRemote(remote, {}, dst), NIXL_SUCCESS);
    nixlProxySubmission record;
    record.src_view = static_cast<nixlProxyDeviceMemView *>(src)->host_view;
    record.dst_view = static_cast<nixlProxyDeviceMemView *>(dst)->host_view;
    record.operand = 3;
    record.dst_offset = 7;
    record.size = 8;
    std::atomic<bool> stop{false};
    std::atomic<unsigned> ready{0}, errors{0};
    std::array<uint64_t, 4> counts{};
    std::vector<std::thread> readers;
    for (size_t i = 0; i < counts.size(); ++i) {
        readers.emplace_back([&, i] {
            ++ready;
            while (!stop.load(std::memory_order_acquire)) {
                nixl::proxyBackendSubmission prepared;
                if (registry.prepareSubmission(record, prepared) != NIXL_SUCCESS ||
                    prepared.local.desc.addr != 0x1003 || prepared.remote.desc.addr != 0x2007 ||
                    prepared.remote.desc.metadataP != &md) {
                    ++errors;
                }
                ++counts[i];
            }
        });
    }
    while (ready.load() != readers.size()) {
        std::this_thread::yield();
    }
    std::vector<nixlMemViewH> added;
    for (size_t i = 0; i < 5000; ++i) {
        nixlMemViewH handle = nullptr;
        EXPECT_EQ(registry.prepRemote(remote, {nullptr}, handle), NIXL_SUCCESS);
        added.push_back(handle);
    }
    for (auto handle : added) {
        EXPECT_EQ(registry.unregister(handle), NIXL_SUCCESS);
    }
    stop.store(true, std::memory_order_release);
    for (auto &reader : readers) {
        reader.join();
    }
    EXPECT_EQ(errors.load(), 0u);
    for (auto count : counts) {
        EXPECT_GT(count, 0u);
    }
    EXPECT_EQ(allocator.liveAllocations(), 2u);
    EXPECT_EQ(registry.unregister(src), NIXL_SUCCESS);
    EXPECT_EQ(registry.unregister(dst), NIXL_SUCCESS);
    EXPECT_EQ(allocator.liveAllocations(), 0u);
}

TEST(ProxyRegistryLifetimeTest, RepeatedViewReplacementReclaimsAllocations) {
    MockDeviceOps allocator;
    DummyBackendMD md;
    nixlProxyDeviceContextData context;
    nixl::proxyMemViewRegistry registry(allocator, &context);
    for (uint64_t i = 0; i < 500; ++i) {
        nixl_remote_meta_dlist_t remote(VRAM_SEG);
        remote.addDesc(makeRemoteDesc("peer", 0x2000 + i * 128, 64, 0, &md));
        nixlMemViewH handle = nullptr;
        ASSERT_EQ(registry.prepRemote(remote, {}, handle), NIXL_SUCCESS);
        nixlProxySubmission record;
        record.dst_view = static_cast<nixlProxyDeviceMemView *>(handle)->host_view;
        record.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
        record.operand = (uint64_t{1} << 40) + i;
        nixl::proxyBackendSubmission prepared;
        ASSERT_EQ(registry.prepareSubmission(record, prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.remote.desc.addr, 0x2000 + i * 128);
        EXPECT_EQ(prepared.value, record.operand);
        ASSERT_EQ(registry.unregister(handle), NIXL_SUCCESS);
        EXPECT_TRUE(allocator.wasFreed(handle));
        EXPECT_EQ(registry.unregister(handle), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(allocator.liveAllocations(), 0u);
    }
}
