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

#include <cstdint>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "device_proxy/proxy_runtime.h"

namespace gtest {
namespace proxy_memview_registry {

    class ProxyMemViewRegistryTest : public testing::Test {
    protected:
        class DummyBackendMD : public nixlBackendMD {
        public:
            DummyBackendMD() : nixlBackendMD(false) {}
        };

        nixlProxyMemViewRegistry registry_;
        DummyBackendMD local_md_;
        DummyBackendMD remote_md_;

        nixlMemViewH
        makeFakeBackendHandle(uint64_t id) {
            return reinterpret_cast<nixlMemViewH>(id);
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
                           uint64_t dev_id = 0) {
            nixl_remote_meta_dlist_t dlist(DRAM_SEG);
            nixlRemoteMetaDesc desc(remote_agent);
            desc.addr = base_addr;
            desc.len = 64;
            desc.devId = dev_id;
            desc.metadataP = &remote_md_;
            dlist.addDesc(desc);
            return dlist;
        }
    };

    TEST_F(ProxyMemViewRegistryTest, RegisterSingle) {
        nixlMemViewH proxy_handle = nullptr;
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(100), &proxy_handle),
                  NIXL_SUCCESS);
        EXPECT_NE(proxy_handle, nullptr);
    }

    TEST_F(ProxyMemViewRegistryTest, RegisterNullOutputReturnsError) {
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(100), nullptr),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, RegisterMultipleAssignsUniqueIds) {
        nixlMemViewH h1 = nullptr, h2 = nullptr, h3 = nullptr;
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &h1), NIXL_SUCCESS);
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &h2), NIXL_SUCCESS);
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(30), &h3), NIXL_SUCCESS);

        EXPECT_NE(h1, h2);
        EXPECT_NE(h2, h3);
        EXPECT_NE(h1, h3);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveByHandle) {
        auto backend = makeFakeBackendHandle(42);
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);

        nixlMemViewH resolved = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemView(proxy_handle, resolved));
        EXPECT_EQ(resolved, backend);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveById) {
        auto backend = makeFakeBackendHandle(42);
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);

        auto proxy_id = reinterpret_cast<uint64_t>(proxy_handle);
        nixlMemViewH resolved = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemViewId(proxy_id, resolved));
        EXPECT_EQ(resolved, backend);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveMultiple) {
        auto b1 = makeFakeBackendHandle(10), b2 = makeFakeBackendHandle(20);
        nixlMemViewH h1 = nullptr, h2 = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(b1, &h1), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(b2, &h2), NIXL_SUCCESS);

        nixlMemViewH r1 = nullptr, r2 = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemView(h1, r1));
        EXPECT_TRUE(registry_.resolveProxyMemView(h2, r2));
        EXPECT_EQ(r1, b1);
        EXPECT_EQ(r2, b2);
    }

    TEST_F(ProxyMemViewRegistryTest, AllocatedEntryIsResolvableBeforeMetadataPublish) {
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(42), &proxy_handle),
                  NIXL_SUCCESS);

        nixlMemViewH resolved = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemView(proxy_handle, resolved));
        EXPECT_EQ(resolved, makeFakeBackendHandle(42));
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRequiresReadyEntries) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_ERR_NOT_FOUND);
    }

    TEST_F(ProxyMemViewRegistryTest, ReadyEntriesProducePreparedTransportDescriptors) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000, "remote-agent")),
                  NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.op_idx = 7;
        submission.channel_id = 3;
        submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        submission.src_offset = 5;
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
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
        EXPECT_EQ(prepared_submission.remote.mem_type, DRAM_SEG);
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2009u);
        EXPECT_EQ(prepared_submission.remote.desc.len, 16u);
        EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md_);
        EXPECT_EQ(prepared_submission.remote_agent, "remote-agent");
    }

    TEST_F(ProxyMemViewRegistryTest, PrepMemViewProducesReadyEntries) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepMemView(makeLocalMetadata(0x1000), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.prepMemView(makeRemoteMetadata(0x2000), &dst_proxy),
                  NIXL_SUCCESS);

        nixlMemViewH resolved = makeFakeBackendHandle(42);
        EXPECT_TRUE(registry_.resolveProxyMemView(src_proxy, resolved));
        EXPECT_EQ(resolved, nullptr);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        submission.src_offset = 4;
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
        submission.dst_offset = 8;
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.local.desc.addr, 0x1004u);
        EXPECT_EQ(prepared_submission.local.desc.len, 16u);
        EXPECT_EQ(prepared_submission.local.desc.metadataP, &local_md_);
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2008u);
        EXPECT_EQ(prepared_submission.remote.desc.len, 16u);
        EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md_);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionAllowsRangesEndingAtDescriptorBoundary) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        submission.src_offset = 48;
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
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
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        submission.src_offset = 60;
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
        submission.size = 8;

        nixlBackendProxySubmission prepared_submission;
        prepared_submission.op_idx = 123;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(prepared_submission.op_idx, 123u);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsDestinationRangeOutsideDescriptor) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
        submission.dst_offset = 60;
        submission.size = 8;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsOverflowingRange) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
        submission.dst_offset = std::numeric_limits<size_t>::max();
        submission.size = 1;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsUnsupportedOpcode) {
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = static_cast<nixl_proxy_opcode_t>(99);
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);

        nixlBackendProxySubmission prepared_submission;
        prepared_submission.op_idx = 123;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_NOT_SUPPORTED);
        EXPECT_EQ(prepared_submission.op_idx, 123u);
    }

    TEST_F(ProxyMemViewRegistryTest, PreparedDescriptorsPreserveDeviceIds) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeLocalMetadata(0x1000, 7)),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000, "peer", 11)),
                  NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
        submission.size = 8;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.local.desc.devId, 7u);
        EXPECT_EQ(prepared_submission.remote.desc.devId, 11u);
    }

    TEST_F(ProxyMemViewRegistryTest, AtomicAddUsesCounterSizeForDestinationBounds) {
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
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

    TEST_F(ProxyMemViewRegistryTest, MetadataKindMustMatchSubmissionRole) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeRemoteMetadata(0x1000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeLocalMetadata(0x2000)), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, RetiredEntriesStopFutureDispatchButKeepOtherEntriesUsable) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        nixlMemViewH other_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(30), &other_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(other_proxy, makeRemoteMetadata(0x3000)), NIXL_SUCCESS);

        ASSERT_EQ(registry_.unregisterProxyMemView(dst_proxy), NIXL_SUCCESS);
        EXPECT_EQ(registry_.unregisterProxyMemView(dst_proxy), NIXL_SUCCESS);

        nixlProxySubmission retired_submission{};
        retired_submission.opcode = nixl_proxy_opcode_t::PUT;
        retired_submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        retired_submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
        retired_submission.size = 8;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(retired_submission, prepared_submission),
                  NIXL_ERR_NOT_FOUND);

        nixlProxySubmission live_submission{};
        live_submission.opcode = nixl_proxy_opcode_t::PUT;
        live_submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        live_submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(other_proxy);
        live_submission.size = 8;

        EXPECT_EQ(registry_.prepareSubmission(live_submission, prepared_submission), NIXL_SUCCESS);
    }

    TEST_F(ProxyMemViewRegistryTest, ClearRetiresExistingEntriesAndPreservesFreshIds) {
        nixlMemViewH old_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &old_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(old_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);

        registry_.clear();

        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(registry_.resolveProxyMemView(old_proxy, resolved));

        nixlMemViewH new_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &new_proxy), NIXL_SUCCESS);
        EXPECT_NE(old_proxy, new_proxy);
        EXPECT_TRUE(registry_.resolveProxyMemView(new_proxy, resolved));
        EXPECT_EQ(resolved, makeFakeBackendHandle(20));
    }

    TEST_F(ProxyMemViewRegistryTest, StoreMetadataRejectsRetiredEntries) {
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &proxy_handle), NIXL_SUCCESS);
        ASSERT_EQ(registry_.unregisterProxyMemView(proxy_handle), NIXL_SUCCESS);
        EXPECT_EQ(registry_.storeMetadata(proxy_handle, makeLocalMetadata(0x1000)),
                  NIXL_ERR_NOT_FOUND);
    }

} // namespace proxy_memview_registry
} // namespace gtest
