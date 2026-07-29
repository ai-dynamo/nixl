/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <ucp/api/device/ucp_device_types.h>

#include "device_proxy/proxy_memview_store.h"

static_assert(offsetof(nixlProxyDeviceMemView, version) ==
              offsetof(ucp_device_local_mem_list_t, version));
static_assert(offsetof(nixlProxyDeviceMemView, version) ==
              offsetof(ucp_device_remote_mem_list_t, version));
static_assert(offsetof(nixlProxyDeviceMemView, length) ==
              offsetof(ucp_device_local_mem_list_t, length));
static_assert(offsetof(nixlProxyDeviceMemView, length) ==
              offsetof(ucp_device_remote_mem_list_t, length));
static_assert((UCP_DEVICE_MEM_LIST_VERSION_V1 & NIXL_PROXY_MEM_LIST_NAMESPACE) == 0);
static_assert(sizeof(nixlProxyDeviceMemView) ==
              offsetof(nixlProxyDeviceMemView, mem_elements));
static_assert(offsetof(nixlProxyDeviceMemView, mem_elements) % alignof(void *) == 0);

namespace gtest::proxy_memview_store {

class DummyBackendMD : public nixlBackendMD {
public:
    DummyBackendMD() : nixlBackendMD(false) {}
};

class ProxyMemViewStoreTest : public testing::Test {
protected:
    void
    SetUp() override {
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) {
            GTEST_SKIP() << "No CUDA-capable GPU";
        }
        store_ = std::make_unique<nixlProxyMemViewStore>(device);
    }

    nixl_meta_dlist_t
    local(uintptr_t address = 0x1000, size_t length = 64, uint64_t device_id = 7) {
        nixl_meta_dlist_t dlist(DRAM_SEG);
        dlist.addDesc(nixlMetaDesc(address, length, device_id, &local_md_));
        return dlist;
    }

    nixl_remote_meta_dlist_t
    remote(uintptr_t address = 0x2000, size_t length = 64, uint64_t device_id = 11) {
        nixl_remote_meta_dlist_t dlist(DRAM_SEG);
        nixlRemoteMetaDesc desc("peer");
        desc.addr = address;
        desc.len = length;
        desc.devId = device_id;
        desc.metadataP = &remote_md_;
        dlist.addDesc(desc);
        return dlist;
    }

    static nixlProxyDeviceMemView
    copyHeader(nixlMemViewH handle) {
        nixlProxyDeviceMemView result{};
        EXPECT_EQ(cudaMemcpy(&result, handle, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
        return result;
    }

    nixlProxySubmission
    put(const nixlPreparedProxyMemView &src,
        const nixlPreparedProxyMemView &dst,
        size_t size = 16) {
        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = src.id;
        submission.dst_proxy_memview_id = dst.id;
        submission.size = size;
        return submission;
    }

    std::unique_ptr<nixlProxyMemViewStore> store_;
    DummyBackendMD local_md_;
    DummyBackendMD remote_md_;
};

TEST_F(ProxyMemViewStoreTest, LocalHandleHasCompatiblePrefixAndNullElement) {
    nixlPreparedProxyMemView prepared;
    ASSERT_EQ(store_->createLocal(local(), prepared), NIXL_SUCCESS);
    ASSERT_NE(prepared.handle, nullptr);
    ASSERT_NE(prepared.id, 0u);

    const auto header = copyHeader(prepared.handle);
    EXPECT_EQ(header.version, NIXL_PROXY_MEM_LIST_VERSION_V1);
    EXPECT_EQ(header.length, 1u);
    EXPECT_EQ(header.proxy_memview_id, prepared.id);
    EXPECT_EQ(header.kind, nixlProxyMemViewKind::LOCAL);

    nixlProxyDeviceMemViewElem element{};
    ASSERT_EQ(cudaMemcpy(&element,
                         static_cast<const nixlProxyDeviceMemView *>(prepared.handle)->mem_elements,
                         sizeof(element),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(element.direct_ptr, nullptr);
}

TEST_F(ProxyMemViewStoreTest, RemoteHandlePreservesOneElementPerDescriptor) {
    auto dlist = remote();
    nixlRemoteMetaDesc second("other-peer");
    second.addr = 0x3000;
    second.len = 128;
    second.metadataP = &remote_md_;
    dlist.addDesc(second);
    void *first = reinterpret_cast<void *>(uintptr_t{0xfeed0000});

    nixlPreparedProxyMemView prepared;
    ASSERT_EQ(store_->createRemote(dlist, {first, nullptr}, prepared), NIXL_SUCCESS);
    const auto header = copyHeader(prepared.handle);
    EXPECT_EQ(header.length, 2u);
    EXPECT_EQ(header.kind, nixlProxyMemViewKind::REMOTE);

    nixlProxyDeviceMemViewElem elements[2]{};
    ASSERT_EQ(cudaMemcpy(elements,
                         static_cast<const nixlProxyDeviceMemView *>(prepared.handle)->mem_elements,
                         sizeof(elements),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(elements[0].direct_ptr, first);
    EXPECT_EQ(elements[1].direct_ptr, nullptr);
}

TEST_F(ProxyMemViewStoreTest, RemoteNullFallbackStillMatchesDescriptorCount) {
    auto dlist = remote();
    dlist.addDesc(nixlRemoteMetaDesc(nixl_null_agent));
    nixlPreparedProxyMemView prepared;
    ASSERT_EQ(store_->createRemote(dlist, {}, prepared), NIXL_SUCCESS);
    EXPECT_EQ(copyHeader(prepared.handle).length, 2u);
}

TEST_F(ProxyMemViewStoreTest, RejectsMismatchedDirectPointerCountWithoutPublishingHandle) {
    nixlPreparedProxyMemView prepared{reinterpret_cast<void *>(uintptr_t{1}), 9};
    EXPECT_EQ(store_->createRemote(remote(), {nullptr, nullptr}, prepared), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(prepared.handle, nullptr);
    EXPECT_EQ(prepared.id, 0u);
}

TEST_F(ProxyMemViewStoreTest, PrepareSubmissionCopiesImmutableMetadata) {
    nixlPreparedProxyMemView src;
    nixlPreparedProxyMemView dst;
    ASSERT_EQ(store_->createLocal(local(), src), NIXL_SUCCESS);
    ASSERT_EQ(store_->createRemote(remote(), {}, dst), NIXL_SUCCESS);

    auto submission = put(src, dst);
    submission.op_idx = 17;
    submission.channel_id = 3;
    submission.src_offset = 5;
    submission.dst_offset = 9;
    nixlBackendProxySubmission prepared{};
    ASSERT_EQ(store_->prepareSubmission(submission, prepared), NIXL_SUCCESS);
    EXPECT_EQ(prepared.op_idx, 17u);
    EXPECT_EQ(prepared.channel_id, 3u);
    EXPECT_EQ(prepared.local.mem_type, DRAM_SEG);
    EXPECT_EQ(prepared.local.desc.addr, 0x1005u);
    EXPECT_EQ(prepared.local.desc.len, 16u);
    EXPECT_EQ(prepared.local.desc.devId, 7u);
    EXPECT_EQ(prepared.local.desc.metadataP, &local_md_);
    EXPECT_EQ(prepared.remote.desc.addr, 0x2009u);
    EXPECT_EQ(prepared.remote.desc.devId, 11u);
    EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md_);
}

TEST_F(ProxyMemViewStoreTest, RejectsWrongKindsIndicesAndOverflowSafeRanges) {
    nixlPreparedProxyMemView local_view;
    nixlPreparedProxyMemView remote_view;
    ASSERT_EQ(store_->createLocal(local(), local_view), NIXL_SUCCESS);
    ASSERT_EQ(store_->createRemote(remote(), {}, remote_view), NIXL_SUCCESS);
    nixlBackendProxySubmission prepared{};

    auto wrong_kinds = put(remote_view, local_view);
    EXPECT_EQ(store_->prepareSubmission(wrong_kinds, prepared), NIXL_ERR_INVALID_PARAM);

    auto bad_index = put(local_view, remote_view);
    bad_index.dst_index = 1;
    EXPECT_EQ(store_->prepareSubmission(bad_index, prepared), NIXL_ERR_INVALID_PARAM);

    auto bad_range = put(local_view, remote_view, 8);
    bad_range.dst_offset = 60;
    EXPECT_EQ(store_->prepareSubmission(bad_range, prepared), NIXL_ERR_INVALID_PARAM);

    auto overflowing_address = local(std::numeric_limits<uintptr_t>::max() - 3, 64);
    nixlPreparedProxyMemView overflow_src;
    ASSERT_EQ(store_->createLocal(overflowing_address, overflow_src), NIXL_SUCCESS);
    auto overflow = put(overflow_src, remote_view, 1);
    overflow.src_offset = 4;
    EXPECT_EQ(store_->prepareSubmission(overflow, prepared), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyMemViewStoreTest, AtomicAddNeedsOnlyAnActiveRemoteView) {
    nixlPreparedProxyMemView dst;
    ASSERT_EQ(store_->createRemote(remote(), {}, dst), NIXL_SUCCESS);
    nixlProxySubmission submission{};
    submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
    submission.dst_proxy_memview_id = dst.id;
    submission.dst_offset = 8;
    submission.value = 19;
    nixlBackendProxySubmission prepared{};
    ASSERT_EQ(store_->prepareSubmission(submission, prepared), NIXL_SUCCESS);
    EXPECT_EQ(prepared.size, sizeof(uint64_t));
    EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared.value, 19u);
}

TEST_F(ProxyMemViewStoreTest, RetiredSlotIsATombstoneAndIdsAreNeverReused) {
    nixlPreparedProxyMemView old_view;
    ASSERT_EQ(store_->createLocal(local(), old_view), NIXL_SUCCESS);
    ASSERT_EQ(store_->release(old_view.id, old_view.handle), NIXL_SUCCESS);
    EXPECT_EQ(store_->release(old_view.id, old_view.handle), NIXL_ERR_INVALID_PARAM);

    nixlPreparedProxyMemView new_view;
    ASSERT_EQ(store_->createLocal(local(), new_view), NIXL_SUCCESS);
    EXPECT_GT(new_view.id, old_view.id);
}

TEST_F(ProxyMemViewStoreTest, ReleaseRequiresStrongIdAndExpectedHandle) {
    nixlPreparedProxyMemView first;
    nixlPreparedProxyMemView second;
    ASSERT_EQ(store_->createLocal(local(), first), NIXL_SUCCESS);
    ASSERT_EQ(store_->createLocal(local(), second), NIXL_SUCCESS);
    EXPECT_EQ(store_->release(first.id, second.handle), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(store_->release(first.id, first.handle), NIXL_SUCCESS);
}

TEST_F(ProxyMemViewStoreTest, GrowsAcross64AndLaterDoublingBoundaries) {
    nixlPreparedProxyMemView dst;
    ASSERT_EQ(store_->createRemote(remote(), {}, dst), NIXL_SUCCESS);
    std::vector<nixlPreparedProxyMemView> locals(129);
    for (auto &view : locals) {
        ASSERT_EQ(store_->createLocal(local(), view), NIXL_SUCCESS);
    }
    EXPECT_EQ(locals[62].id, 64u);
    EXPECT_EQ(locals[63].id, 65u);
    EXPECT_EQ(locals[126].id, 128u);
    EXPECT_EQ(locals[127].id, 129u);

    nixlBackendProxySubmission prepared{};
    for (const auto &view : locals) {
        EXPECT_EQ(store_->prepareSubmission(put(view, dst), prepared), NIXL_SUCCESS);
    }
}

TEST_F(ProxyMemViewStoreTest, ConcurrentCreateRetireAndLockFreeResolution) {
    constexpr size_t count = 192;
    std::vector<nixlPreparedProxyMemView> locals(count);
    nixlPreparedProxyMemView dst;
    ASSERT_EQ(store_->createRemote(remote(), {}, dst), NIXL_SUCCESS);
    std::atomic<size_t> published{0};
    std::atomic<bool> retire{false};
    std::atomic<bool> failed{false};

    std::thread writer([&] {
        for (size_t index = 0; index < count; ++index) {
            if (store_->createLocal(local(0x1000 + index * 0x100), locals[index]) !=
                NIXL_SUCCESS) {
                failed.store(true, std::memory_order_release);
                return;
            }
            published.store(index + 1, std::memory_order_release);
        }
        while (!retire.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (size_t index = 0; index < count; index += 2) {
            if (store_->release(locals[index].id, locals[index].handle) != NIXL_SUCCESS) {
                failed.store(true, std::memory_order_release);
            }
        }
    });

    std::vector<std::thread> readers;
    for (int reader = 0; reader < 4; ++reader) {
        readers.emplace_back([&] {
            nixlBackendProxySubmission prepared{};
            size_t passes = 0;
            while (published.load(std::memory_order_acquire) < count || passes < 4) {
                const size_t limit = published.load(std::memory_order_acquire);
                for (size_t index = 0; index < limit; ++index) {
                    const nixl_status_t status =
                        store_->prepareSubmission(put(locals[index], dst), prepared);
                    if (status != NIXL_SUCCESS && status != NIXL_ERR_NOT_FOUND) {
                        failed.store(true, std::memory_order_release);
                    }
                }
                if (limit == count) {
                    ++passes;
                }
            }
        });
    }

    while (published.load(std::memory_order_acquire) < count &&
           !failed.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    retire.store(true, std::memory_order_release);
    writer.join();
    for (auto &reader : readers) {
        reader.join();
    }
    EXPECT_FALSE(failed.load(std::memory_order_acquire));
}

} // namespace gtest::proxy_memview_store
