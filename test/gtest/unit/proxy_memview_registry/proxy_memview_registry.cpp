/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "device/device_allocator.h"
#include "device/proxy/proxy_registry.h"

namespace gtest {
namespace proxy_memview_registry {

    class ProxyMemViewRegistryTest : public testing::Test {
    protected:
        class DummyBackendMD : public nixlBackendMD {
        public:
            DummyBackendMD() : nixlBackendMD(false) {}
        };

        nixlProxyMemViewRegistry registry_{nixlGetDeviceAllocator(), nullptr};
        DummyBackendMD local_md_;
        DummyBackendMD remote_md_;

        static uint32_t
        proxyMemViewId(nixlMemViewH proxy_memview) {
            if (proxy_memview == nullptr) {
                return 0;
            }
            nixlProxyDeviceMemView device_memview{};
            EXPECT_EQ(
                cudaMemcpy(
                    &device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
                cudaSuccess);
            return device_memview.proxy_memview_id;
        }

        static nixlProxyDeviceMemView
        copyDeviceMemView(nixlMemViewH proxy_memview) {
            nixlProxyDeviceMemView device_memview{};
            EXPECT_EQ(
                cudaMemcpy(
                    &device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
                cudaSuccess);
            return device_memview;
        }

        static std::vector<void *>
        copyDirectPointers(nixlMemViewH proxy_memview, size_t count) {
            std::vector<void *> direct_ptrs(count, nullptr);
            if (count != 0) {
                auto *direct_ptrs_dev =
                    nixlProxyDeviceMemViewDirectPtrs(
                        static_cast<nixlProxyDeviceMemView *>(proxy_memview));
                EXPECT_EQ(cudaMemcpy(direct_ptrs.data(),
                                     direct_ptrs_dev,
                                     sizeof(void *) * count,
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
            }
            return direct_ptrs;
        }

        nixl_meta_dlist_t
        makeLocalMetadata(uintptr_t base_addr, uint64_t dev_id = 0) {
            nixl_meta_dlist_t dlist(DRAM_SEG);
            dlist.addDesc(nixlMetaDesc(base_addr, 64, dev_id, &local_md_));
            return dlist;
        }

        nixl_remote_meta_dlist_t
        makeRemoteMetadata(uintptr_t base_addr,
                           const std::string &remote_agent = "peer",
                           uint64_t dev_id = 0,
                           nixl_mem_t mem_type = VRAM_SEG) {
            nixl_remote_meta_dlist_t dlist(mem_type);
            nixlRemoteMetaDesc desc(remote_agent);
            desc.addr = base_addr;
            desc.len = 64;
            desc.devId = dev_id;
            desc.metadataP = &remote_md_;
            dlist.addDesc(desc);
            return dlist;
        }

        /** prepLocal + prepRemote over 64-byte descriptors at the given bases. */
        void
        prepPair(uintptr_t local_addr,
                 uintptr_t remote_addr,
                 nixlMemViewH &src_proxy,
                 nixlMemViewH &dst_proxy,
                 const std::string &remote_agent = "peer") {
            ASSERT_EQ(registry_.prepLocal(makeLocalMetadata(local_addr), src_proxy), NIXL_SUCCESS);
            ASSERT_EQ(
                registry_.prepRemote(makeRemoteMetadata(remote_addr, remote_agent), {}, dst_proxy),
                NIXL_SUCCESS);
        }
    };

    TEST_F(ProxyMemViewRegistryTest, PrepLocalAssignsFirstId) {
        nixlMemViewH proxy_handle = nullptr;
        EXPECT_EQ(registry_.prepLocal(makeLocalMetadata(0x1000), proxy_handle), NIXL_SUCCESS);
        EXPECT_NE(proxy_handle, nullptr);

        const nixlProxyDeviceMemView device_memview = copyDeviceMemView(proxy_handle);
        EXPECT_EQ(device_memview.proxy_memview_id, 1u);
        EXPECT_EQ(device_memview.direct_ptr_count, 0u);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepAssignsUniqueHandlesAndMonotonicIds) {
        nixlMemViewH h1 = nullptr, h2 = nullptr, h3 = nullptr;
        ASSERT_EQ(registry_.prepLocal(makeLocalMetadata(0x1000), h1), NIXL_SUCCESS);
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x2000), {}, h2), NIXL_SUCCESS);
        ASSERT_EQ(registry_.prepLocal(makeLocalMetadata(0x3000), h3), NIXL_SUCCESS);

        EXPECT_NE(h1, h2);
        EXPECT_NE(h2, h3);
        EXPECT_NE(h1, h3);
        EXPECT_EQ(proxyMemViewId(h1), 1u);
        EXPECT_EQ(proxyMemViewId(h2), 2u);
        EXPECT_EQ(proxyMemViewId(h3), 3u);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveFindsLiveHandles) {
        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        prepPair(0x1000, 0x2000, src_proxy, dst_proxy);

        // Nothing sets a backend memview on the proxy path today, so resolve
        // reports "live" with a null backend handle.
        nixlMemViewH resolved = reinterpret_cast<nixlMemViewH>(uintptr_t{0x42});
        EXPECT_TRUE(registry_.resolve(src_proxy, resolved));
        EXPECT_EQ(resolved, nullptr);
        resolved = reinterpret_cast<nixlMemViewH>(uintptr_t{0x42});
        EXPECT_TRUE(registry_.resolve(dst_proxy, resolved));
        EXPECT_EQ(resolved, nullptr);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveMissesUnknownHandles) {
        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(registry_.resolve(reinterpret_cast<nixlMemViewH>(uintptr_t{0x99}), resolved));
    }

    TEST_F(ProxyMemViewRegistryTest, SubmissionRecordStaysPackedTo64Bytes) {
        EXPECT_EQ(sizeof(nixlProxySubmission), 64u);
        EXPECT_EQ(alignof(nixlProxySubmission), 64u);
        EXPECT_EQ(offsetof(nixlProxySubmission, op_idx), 0u);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsUnknownIds) {
        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = 7;
        submission.dst_proxy_memview_id = 9;
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_ERR_NOT_FOUND);
    }

    TEST_F(ProxyMemViewRegistryTest, ReadyEntriesProducePreparedTransportDescriptors) {
        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        prepPair(0x1000, 0x2000, src_proxy, dst_proxy, "remote-agent");

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.op_idx = 7;
        submission.channel_id = 3;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.src_offset = 5;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = 9;
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.op_idx, 7u);
        EXPECT_EQ(prepared_submission.channel_id, 3u);
        EXPECT_EQ(prepared_submission.local.mem_type, DRAM_SEG);
        EXPECT_EQ(prepared_submission.local.desc.addr, 0x1005u);
        EXPECT_EQ(prepared_submission.local.desc.len, 16u);
        EXPECT_EQ(prepared_submission.local.desc.metadataP, &local_md_);
        EXPECT_EQ(prepared_submission.remote.mem_type, VRAM_SEG);
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2009u);
        EXPECT_EQ(prepared_submission.remote.desc.len, 16u);
        EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md_);
        EXPECT_EQ(prepared_submission.remote_agent, "remote-agent");
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionAccepts64BitOffsets) {
        constexpr uint64_t kLargeOffset = (uint64_t{1} << 32) + 16;
        nixl_meta_dlist_t local_dlist(DRAM_SEG);
        local_dlist.addDesc(nixlMetaDesc(0x1000, kLargeOffset + 64, 0, &local_md_));
        nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
        nixlRemoteMetaDesc remote_desc("peer");
        remote_desc.addr = 0x2000;
        remote_desc.len = kLargeOffset + 64;
        remote_desc.devId = 0;
        remote_desc.metadataP = &remote_md_;
        remote_dlist.addDesc(remote_desc);

        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepLocal(local_dlist, src_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.prepRemote(remote_dlist, {}, dst_proxy), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.src_offset = kLargeOffset;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = kLargeOffset;
        submission.size = 32;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.local.desc.addr, uintptr_t{0x1000} + kLargeOffset);
        EXPECT_EQ(prepared_submission.remote.desc.addr, uintptr_t{0x2000} + kLargeOffset);
        EXPECT_EQ(prepared_submission.local.desc.len, 32u);
        EXPECT_EQ(prepared_submission.remote.desc.len, 32u);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionAccepts64BitSize) {
        constexpr uint64_t kLargeSize = (uint64_t{1} << 32) + 64;
        nixl_meta_dlist_t local_dlist(DRAM_SEG);
        local_dlist.addDesc(nixlMetaDesc(0x1000, kLargeSize, 0, &local_md_));
        nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
        nixlRemoteMetaDesc remote_desc("peer");
        remote_desc.addr = 0x2000;
        remote_desc.len = kLargeSize;
        remote_desc.devId = 0;
        remote_desc.metadataP = &remote_md_;
        remote_dlist.addDesc(remote_desc);

        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepLocal(local_dlist, src_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.prepRemote(remote_dlist, {}, dst_proxy), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.size = kLargeSize;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.size, kLargeSize);
        EXPECT_EQ(prepared_submission.local.desc.len, kLargeSize);
        EXPECT_EQ(prepared_submission.remote.desc.len, kLargeSize);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepRemoteRejectsNonVram) {
        nixlMemViewH dst_proxy = nullptr;
        EXPECT_EQ(registry_.prepRemote(
                      makeRemoteMetadata(0x2000, "remote-agent", 0, DRAM_SEG), {}, dst_proxy),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(dst_proxy, nullptr);

        // The rejected prep consumed no id.
        nixlMemViewH accepted = nullptr;
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x2000), {}, accepted), NIXL_SUCCESS);
        EXPECT_EQ(proxyMemViewId(accepted), 1u);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepRemoteStoresDirectPointers) {
        nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
        nixlRemoteMetaDesc first("peer0");
        first.addr = 0x2000;
        first.len = 64;
        first.devId = 0;
        first.metadataP = &remote_md_;
        remote_dlist.addDesc(first);
        nixlRemoteMetaDesc second("peer1");
        second.addr = 0x3000;
        second.len = 64;
        second.devId = 1;
        second.metadataP = &remote_md_;
        remote_dlist.addDesc(second);

        std::vector<void *> direct_ptrs{reinterpret_cast<void *>(uintptr_t{0xfeed0000}), nullptr};
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepRemote(remote_dlist, direct_ptrs, dst_proxy), NIXL_SUCCESS);

        const nixlProxyDeviceMemView device_memview = copyDeviceMemView(dst_proxy);
        EXPECT_EQ(device_memview.proxy_memview_id, proxyMemViewId(dst_proxy));
        EXPECT_EQ(device_memview.direct_ptr_count, direct_ptrs.size());
        EXPECT_EQ(copyDirectPointers(dst_proxy, direct_ptrs.size()), direct_ptrs);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionAllowsRangesEndingAtDescriptorBoundary) {
        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        prepPair(0x1000, 0x2000, src_proxy, dst_proxy);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.src_offset = 48;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = 48;
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.local.desc.addr, 0x1030u);
        EXPECT_EQ(prepared_submission.local.desc.len, 16u);
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2030u);
        EXPECT_EQ(prepared_submission.remote.desc.len, 16u);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsSourceRangeOutsideDescriptor) {
        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        prepPair(0x1000, 0x2000, src_proxy, dst_proxy);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.src_offset = 60;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.size = 8;

        nixlBackendProxySubmission prepared_submission;
        prepared_submission.op_idx = 123;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(prepared_submission.op_idx, 123u);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsDestinationRangeOutsideDescriptor) {
        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        prepPair(0x1000, 0x2000, src_proxy, dst_proxy);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = 60;
        submission.size = 8;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsOverflowingRange) {
        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        prepPair(0x1000, 0x2000, src_proxy, dst_proxy);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = std::numeric_limits<uint32_t>::max();
        submission.size = 1;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsUnsupportedOpcode) {
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x2000), {}, dst_proxy), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = static_cast<nixl_proxy_opcode_t>(99);
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);

        nixlBackendProxySubmission prepared_submission;
        prepared_submission.op_idx = 123;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_NOT_SUPPORTED);
        EXPECT_EQ(prepared_submission.op_idx, 123u);
    }

    TEST_F(ProxyMemViewRegistryTest, PreparedDescriptorsPreserveDeviceIds) {
        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepLocal(makeLocalMetadata(0x1000, 7), src_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x2000, "peer", 11), {}, dst_proxy),
                  NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.size = 8;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.local.desc.devId, 7u);
        EXPECT_EQ(prepared_submission.remote.desc.devId, 11u);
    }

    TEST_F(ProxyMemViewRegistryTest, AtomicAddUsesCounterSizeForDestinationBounds) {
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x2000), {}, dst_proxy), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = 56;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.size, sizeof(uint64_t));
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2038u);
        EXPECT_EQ(prepared_submission.remote.desc.len, sizeof(uint64_t));

        submission.dst_offset = 60;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, ReadyRemoteEntryProducesAtomicPreparedDescriptor) {
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x2000, "remote-agent"), {}, dst_proxy),
                  NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
        submission.op_idx = 7;
        submission.channel_id = 3;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = 9;
        submission.size = sizeof(uint64_t);
        submission.value = 42;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
        EXPECT_EQ(prepared_submission.op_idx, 7u);
        EXPECT_EQ(prepared_submission.channel_id, 3u);
        EXPECT_EQ(prepared_submission.remote.mem_type, VRAM_SEG);
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2009u);
        EXPECT_EQ(prepared_submission.remote.desc.len, sizeof(uint64_t));
        EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md_);
        EXPECT_EQ(prepared_submission.remote_agent, "remote-agent");
        EXPECT_EQ(prepared_submission.value, 42u);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsEmptyRemoteAgent) {
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x2000, ""), {}, dst_proxy),
                  NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsNullRemoteAgent) {
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x2000, nixl_null_agent), {}, dst_proxy),
                  NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, MetadataKindMustMatchSubmissionRole) {
        // Roles swapped: a remote memview as source, a local one as destination.
        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x1000), {}, src_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.prepLocal(makeLocalMetadata(0x2000), dst_proxy), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, RetiredEntriesStopFutureDispatchButKeepOtherEntriesUsable) {
        nixlMemViewH src_proxy = nullptr, dst_proxy = nullptr, other_proxy = nullptr;
        prepPair(0x1000, 0x2000, src_proxy, dst_proxy);
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x3000), {}, other_proxy), NIXL_SUCCESS);
        const uint32_t src_proxy_id = proxyMemViewId(src_proxy);
        const uint32_t dst_proxy_id = proxyMemViewId(dst_proxy);
        const uint32_t other_proxy_id = proxyMemViewId(other_proxy);

        ASSERT_EQ(registry_.unregister(dst_proxy), NIXL_SUCCESS);
        EXPECT_EQ(registry_.unregister(dst_proxy), NIXL_ERR_INVALID_PARAM);

        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(registry_.resolve(dst_proxy, resolved));

        nixlProxySubmission retired_submission{};
        retired_submission.opcode = nixl_proxy_opcode_t::PUT;
        retired_submission.src_proxy_memview_id = src_proxy_id;
        retired_submission.dst_proxy_memview_id = dst_proxy_id;
        retired_submission.size = 8;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(retired_submission, prepared_submission),
                  NIXL_ERR_NOT_FOUND);

        nixlProxySubmission live_submission{};
        live_submission.opcode = nixl_proxy_opcode_t::PUT;
        live_submission.src_proxy_memview_id = src_proxy_id;
        live_submission.dst_proxy_memview_id = other_proxy_id;
        live_submission.size = 8;

        EXPECT_EQ(registry_.prepareSubmission(live_submission, prepared_submission), NIXL_SUCCESS);
    }

    TEST_F(ProxyMemViewRegistryTest, IdsAreNotReusedAfterRetirement) {
        nixlMemViewH first = nullptr, second = nullptr;
        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x2000), {}, first), NIXL_SUCCESS);
        ASSERT_EQ(registry_.unregister(first), NIXL_SUCCESS);

        ASSERT_EQ(registry_.prepRemote(makeRemoteMetadata(0x3000), {}, second), NIXL_SUCCESS);
        EXPECT_EQ(proxyMemViewId(second), 2u);
    }

    // A registered memview whose descriptor list came back empty still
    // resolves, it just cannot satisfy any submission index.
    TEST_F(ProxyMemViewRegistryTest, EmptyDescriptorListRejectsEverySubmission) {
        nixl_remote_meta_dlist_t empty(VRAM_SEG);
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepRemote(empty, {}, dst_proxy), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

} // namespace proxy_memview_registry
} // namespace gtest
