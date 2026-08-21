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

#include "device_proxy/proxy_runtime.h"

namespace gtest {
namespace proxy_memview_registry {
    namespace {

        class proxyMemViewRegistryTest : public testing::Test {
        protected:
            class dummyBackendMd : public nixlBackendMD {
            public:
                dummyBackendMd() : nixlBackendMD(false) {}
            };

            nixlProxyMemViewRegistry registry;
            dummyBackendMd localMd;
            dummyBackendMd remoteMd;

            nixlMemViewH
            makeFakeBackendHandle(uint64_t id) {
                return reinterpret_cast<nixlMemViewH>(id);
            }

            static uint32_t
            proxyMemViewId(nixlMemViewH proxy_memview) {
                if (proxy_memview == nullptr) {
                    return 0;
                }
                nixlProxyDeviceMemView device_memview{};
                EXPECT_EQ(cudaMemcpy(&device_memview,
                                     proxy_memview,
                                     sizeof(device_memview),
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
                return device_memview.proxyMemViewId;
            }

            static nixlProxyDeviceMemView
            copyDeviceMemView(nixlMemViewH proxy_memview) {
                nixlProxyDeviceMemView device_memview{};
                EXPECT_EQ(cudaMemcpy(&device_memview,
                                     proxy_memview,
                                     sizeof(device_memview),
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
                return device_memview;
            }

            static std::vector<void *>
            copyDirectPointers(nixlMemViewH proxy_memview, size_t count) {
                std::vector<void *> direct_ptrs(count, nullptr);
                if (count != 0) {
                    auto *direct_ptrs_dev =
                        static_cast<nixlProxyDeviceMemView *>(proxy_memview)->directPtrs;
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
                dlist.addDesc(nixlMetaDesc(base_addr, 64, dev_id, &localMd));
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
                desc.metadataP = &remoteMd;
                dlist.addDesc(desc);
                return dlist;
            }
        };

        TEST_F(proxyMemViewRegistryTest, RegisterSingle) {
            nixlMemViewH proxy_handle = nullptr;
            EXPECT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(100), &proxy_handle),
                      NIXL_SUCCESS);
            EXPECT_NE(proxy_handle, nullptr);

            const nixlProxyDeviceMemView device_memview = copyDeviceMemView(proxy_handle);
            EXPECT_EQ(device_memview.proxyMemViewId, 1u);
            EXPECT_EQ(device_memview.directPtrCount, 0u);
        }

        TEST_F(proxyMemViewRegistryTest, RegisterNullOutputReturnsError) {
            EXPECT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(100), nullptr),
                      NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyMemViewRegistryTest, RegisterMultipleAssignsUniqueIds) {
            nixlMemViewH h1 = nullptr, h2 = nullptr, h3 = nullptr;
            EXPECT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &h1), NIXL_SUCCESS);
            EXPECT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &h2), NIXL_SUCCESS);
            EXPECT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(30), &h3), NIXL_SUCCESS);

            EXPECT_NE(h1, h2);
            EXPECT_NE(h2, h3);
            EXPECT_NE(h1, h3);
        }

        TEST_F(proxyMemViewRegistryTest, ResolveByHandle) {
            auto backend = makeFakeBackendHandle(42);
            nixlMemViewH proxy_handle = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);

            nixlMemViewH resolved = nullptr;
            EXPECT_TRUE(registry.resolveProxyMemView(proxy_handle, resolved));
            EXPECT_EQ(resolved, backend);
        }

        TEST_F(proxyMemViewRegistryTest, ResolveById) {
            auto backend = makeFakeBackendHandle(42);
            nixlMemViewH proxy_handle = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);

            auto proxy_id = proxyMemViewId(proxy_handle);
            nixlMemViewH resolved = nullptr;
            EXPECT_TRUE(registry.resolveProxyMemViewId(proxy_id, resolved));
            EXPECT_EQ(resolved, backend);
        }

        TEST_F(proxyMemViewRegistryTest, ResolveMultiple) {
            auto b1 = makeFakeBackendHandle(10), b2 = makeFakeBackendHandle(20);
            nixlMemViewH h1 = nullptr, h2 = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(b1, &h1), NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(b2, &h2), NIXL_SUCCESS);

            nixlMemViewH r1 = nullptr, r2 = nullptr;
            EXPECT_TRUE(registry.resolveProxyMemView(h1, r1));
            EXPECT_TRUE(registry.resolveProxyMemView(h2, r2));
            EXPECT_EQ(r1, b1);
            EXPECT_EQ(r2, b2);
        }

        TEST_F(proxyMemViewRegistryTest, SubmissionRecordStaysPackedTo64Bytes) {
            EXPECT_EQ(sizeof(nixlProxySubmission), 64u);
            EXPECT_EQ(alignof(nixlProxySubmission), 64u);
            EXPECT_EQ(offsetof(nixlProxySubmission, opIdx), 0u);
        }

        TEST_F(proxyMemViewRegistryTest, AllocatedEntryIsResolvableBeforeMetadataPublish) {
            nixlMemViewH proxy_handle = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(42), &proxy_handle),
                      NIXL_SUCCESS);

            nixlMemViewH resolved = nullptr;
            EXPECT_TRUE(registry.resolveProxyMemView(proxy_handle, resolved));
            EXPECT_EQ(resolved, makeFakeBackendHandle(42));
        }

        TEST_F(proxyMemViewRegistryTest, PrepareSubmissionRequiresReadyEntries) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.size = 16;

            nixlBackendProxySubmission prepared_submission;
            EXPECT_EQ(registry.prepareSubmission(submission, prepared_submission),
                      NIXL_ERR_NOT_FOUND);
        }

        TEST_F(proxyMemViewRegistryTest, ReadyEntriesProducePreparedTransportDescriptors) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000, "remote-agent")),
                      NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.opIdx = 7;
            submission.channelId = 3;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.srcOffset = 5;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 9;
            submission.size = 16;

            nixlBackendProxySubmission prepared_submission;
            ASSERT_EQ(registry.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
            EXPECT_EQ(prepared_submission.opIdx, 7u);
            EXPECT_EQ(prepared_submission.channelId, 3u);
            EXPECT_EQ(prepared_submission.local.memType, DRAM_SEG);
            EXPECT_EQ(prepared_submission.local.desc.addr, 0x1005u);
            EXPECT_EQ(prepared_submission.local.desc.len, 16u);
            EXPECT_EQ(prepared_submission.local.desc.metadataP, &localMd);
            EXPECT_EQ(prepared_submission.remote.memType, VRAM_SEG);
            EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2009u);
            EXPECT_EQ(prepared_submission.remote.desc.len, 16u);
            EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remoteMd);
            EXPECT_EQ(prepared_submission.remoteAgent, "remote-agent");
        }

        TEST_F(proxyMemViewRegistryTest, PrepareSubmissionAccepts64BitOffsets) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);

            constexpr uint64_t kLargeOffset = (uint64_t{1} << 32) + 16;
            nixl_meta_dlist_t local_dlist(DRAM_SEG);
            local_dlist.addDesc(nixlMetaDesc(0x1000, kLargeOffset + 64, 0, &localMd));
            nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
            nixlRemoteMetaDesc remote_desc("peer");
            remote_desc.addr = 0x2000;
            remote_desc.len = kLargeOffset + 64;
            remote_desc.devId = 0;
            remote_desc.metadataP = &remoteMd;
            remote_dlist.addDesc(remote_desc);

            ASSERT_EQ(registry.storeMetadata(src_proxy, local_dlist), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, remote_dlist), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.srcOffset = kLargeOffset;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = kLargeOffset;
            submission.size = 32;

            nixlBackendProxySubmission prepared_submission;
            ASSERT_EQ(registry.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
            EXPECT_EQ(prepared_submission.local.desc.addr, uintptr_t{0x1000} + kLargeOffset);
            EXPECT_EQ(prepared_submission.remote.desc.addr, uintptr_t{0x2000} + kLargeOffset);
            EXPECT_EQ(prepared_submission.local.desc.len, 32u);
            EXPECT_EQ(prepared_submission.remote.desc.len, 32u);
        }

        TEST_F(proxyMemViewRegistryTest, PrepareSubmissionAccepts64BitSize) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);

            constexpr uint64_t kLargeSize = (uint64_t{1} << 32) + 64;
            nixl_meta_dlist_t local_dlist(DRAM_SEG);
            local_dlist.addDesc(nixlMetaDesc(0x1000, kLargeSize, 0, &localMd));
            nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
            nixlRemoteMetaDesc remote_desc("peer");
            remote_desc.addr = 0x2000;
            remote_desc.len = kLargeSize;
            remote_desc.devId = 0;
            remote_desc.metadataP = &remoteMd;
            remote_dlist.addDesc(remote_desc);

            ASSERT_EQ(registry.storeMetadata(src_proxy, local_dlist), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, remote_dlist), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.size = kLargeSize;

            nixlBackendProxySubmission prepared_submission;
            ASSERT_EQ(registry.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
            EXPECT_EQ(prepared_submission.size, kLargeSize);
            EXPECT_EQ(prepared_submission.local.desc.len, kLargeSize);
            EXPECT_EQ(prepared_submission.remote.desc.len, kLargeSize);
        }

        TEST_F(proxyMemViewRegistryTest, StoreRemoteMetadataRejectsNonVram) {
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);

            EXPECT_EQ(registry.storeMetadata(
                          dst_proxy, makeRemoteMetadata(0x2000, "remote-agent", 0, DRAM_SEG)),
                      NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyMemViewRegistryTest, PrepMemViewProducesReadyEntries) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.prepMemView(makeLocalMetadata(0x1000), &src_proxy), NIXL_SUCCESS);
            ASSERT_EQ(registry.prepMemView(makeRemoteMetadata(0x2000), &dst_proxy), NIXL_SUCCESS);

            nixlMemViewH resolved = makeFakeBackendHandle(42);
            EXPECT_TRUE(registry.resolveProxyMemView(src_proxy, resolved));
            EXPECT_EQ(resolved, nullptr);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.srcOffset = 4;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 8;
            submission.size = 16;

            nixlBackendProxySubmission prepared_submission;
            ASSERT_EQ(registry.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
            EXPECT_EQ(prepared_submission.local.desc.addr, 0x1004u);
            EXPECT_EQ(prepared_submission.local.desc.len, 16u);
            EXPECT_EQ(prepared_submission.local.desc.metadataP, &localMd);
            EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2008u);
            EXPECT_EQ(prepared_submission.remote.desc.len, 16u);
            EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remoteMd);
        }

        TEST_F(proxyMemViewRegistryTest, PrepRemoteMemViewStoresDirectPointers) {
            nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
            nixlRemoteMetaDesc first("peer0");
            first.addr = 0x2000;
            first.len = 64;
            first.devId = 0;
            first.metadataP = &remoteMd;
            remote_dlist.addDesc(first);
            nixlRemoteMetaDesc second("peer1");
            second.addr = 0x3000;
            second.len = 64;
            second.devId = 1;
            second.metadataP = &remoteMd;
            remote_dlist.addDesc(second);

            std::vector<void *> direct_ptrs{reinterpret_cast<void *>(uintptr_t{0xfeed0000}),
                                            nullptr};
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.prepMemView(remote_dlist, direct_ptrs, &dst_proxy), NIXL_SUCCESS);

            const nixlProxyDeviceMemView device_memview = copyDeviceMemView(dst_proxy);
            EXPECT_EQ(device_memview.proxyMemViewId, proxyMemViewId(dst_proxy));
            EXPECT_EQ(device_memview.directPtrCount, direct_ptrs.size());
            EXPECT_EQ(copyDirectPointers(dst_proxy, direct_ptrs.size()), direct_ptrs);
        }

        TEST_F(proxyMemViewRegistryTest, PrepareSubmissionAllowsRangesEndingAtDescriptorBoundary) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.srcOffset = 48;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 48;
            submission.size = 16;

            nixlBackendProxySubmission prepared_submission;
            ASSERT_EQ(registry.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
            EXPECT_EQ(prepared_submission.local.desc.addr, 0x1030u);
            EXPECT_EQ(prepared_submission.local.desc.len, 16u);
            EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2030u);
            EXPECT_EQ(prepared_submission.remote.desc.len, 16u);
        }

        TEST_F(proxyMemViewRegistryTest, PrepareSubmissionRejectsSourceRangeOutsideDescriptor) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.srcOffset = 60;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.size = 8;

            nixlBackendProxySubmission prepared_submission;
            prepared_submission.opIdx = 123;
            EXPECT_EQ(registry.prepareSubmission(submission, prepared_submission),
                      NIXL_ERR_INVALID_PARAM);
            EXPECT_EQ(prepared_submission.opIdx, 123u);
        }

        TEST_F(proxyMemViewRegistryTest,
               PrepareSubmissionRejectsDestinationRangeOutsideDescriptor) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 60;
            submission.size = 8;

            nixlBackendProxySubmission prepared_submission;
            EXPECT_EQ(registry.prepareSubmission(submission, prepared_submission),
                      NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyMemViewRegistryTest, PrepareSubmissionRejectsOverflowingRange) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = std::numeric_limits<uint32_t>::max();
            submission.size = 1;

            nixlBackendProxySubmission prepared_submission;
            EXPECT_EQ(registry.prepareSubmission(submission, prepared_submission),
                      NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyMemViewRegistryTest, PrepareSubmissionRejectsUnsupportedOpcode) {
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = static_cast<nixl_proxy_opcode_t>(99);
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);

            nixlBackendProxySubmission prepared_submission;
            prepared_submission.opIdx = 123;
            EXPECT_EQ(registry.prepareSubmission(submission, prepared_submission),
                      NIXL_ERR_NOT_SUPPORTED);
            EXPECT_EQ(prepared_submission.opIdx, 123u);
        }

        TEST_F(proxyMemViewRegistryTest, PreparedDescriptorsPreserveDeviceIds) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(src_proxy, makeLocalMetadata(0x1000, 7)),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000, "peer", 11)),
                      NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.size = 8;

            nixlBackendProxySubmission prepared_submission;
            ASSERT_EQ(registry.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
            EXPECT_EQ(prepared_submission.local.desc.devId, 7u);
            EXPECT_EQ(prepared_submission.remote.desc.devId, 11u);
        }

        TEST_F(proxyMemViewRegistryTest, AtomicAddUsesCounterSizeForDestinationBounds) {
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 56;

            nixlBackendProxySubmission prepared_submission;
            ASSERT_EQ(registry.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
            EXPECT_EQ(prepared_submission.size, sizeof(uint64_t));
            EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2038u);
            EXPECT_EQ(prepared_submission.remote.desc.len, sizeof(uint64_t));

            submission.dstOffset = 60;
            EXPECT_EQ(registry.prepareSubmission(submission, prepared_submission),
                      NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyMemViewRegistryTest, ReadyRemoteEntryProducesAtomicPreparedDescriptor) {
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000, "remote-agent")),
                      NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            submission.opIdx = 7;
            submission.channelId = 3;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 9;
            submission.size = sizeof(uint64_t);
            submission.value = 42;

            nixlBackendProxySubmission prepared_submission;
            ASSERT_EQ(registry.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
            EXPECT_EQ(prepared_submission.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
            EXPECT_EQ(prepared_submission.opIdx, 7u);
            EXPECT_EQ(prepared_submission.channelId, 3u);
            EXPECT_EQ(prepared_submission.remote.memType, VRAM_SEG);
            EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2009u);
            EXPECT_EQ(prepared_submission.remote.desc.len, sizeof(uint64_t));
            EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remoteMd);
            EXPECT_EQ(prepared_submission.remoteAgent, "remote-agent");
            EXPECT_EQ(prepared_submission.value, 42u);
        }

        TEST_F(proxyMemViewRegistryTest, PrepareSubmissionRejectsEmptyRemoteAgent) {
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000, "")),
                      NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);

            nixlBackendProxySubmission prepared_submission;
            EXPECT_EQ(registry.prepareSubmission(submission, prepared_submission),
                      NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyMemViewRegistryTest, PrepareSubmissionRejectsNullRemoteAgent) {
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(
                registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000, nixl_null_agent)),
                NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);

            nixlBackendProxySubmission prepared_submission;
            EXPECT_EQ(registry.prepareSubmission(submission, prepared_submission),
                      NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyMemViewRegistryTest, MetadataKindMustMatchSubmissionRole) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(src_proxy, makeRemoteMetadata(0x1000)), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeLocalMetadata(0x2000)), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.size = 16;

            nixlBackendProxySubmission prepared_submission;
            EXPECT_EQ(registry.prepareSubmission(submission, prepared_submission),
                      NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyMemViewRegistryTest,
               RetiredEntriesStopFutureDispatchButKeepOtherEntriesUsable) {
            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            nixlMemViewH other_proxy = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(30), &other_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);
            ASSERT_EQ(registry.storeMetadata(other_proxy, makeRemoteMetadata(0x3000)),
                      NIXL_SUCCESS);
            const uint32_t src_proxy_id = proxyMemViewId(src_proxy);
            const uint32_t dst_proxy_id = proxyMemViewId(dst_proxy);
            const uint32_t other_proxy_id = proxyMemViewId(other_proxy);

            ASSERT_EQ(registry.unregisterProxyMemView(dst_proxy), NIXL_SUCCESS);
            EXPECT_EQ(registry.unregisterProxyMemView(dst_proxy), NIXL_ERR_INVALID_PARAM);

            nixlProxySubmission retired_submission{};
            retired_submission.opcode = nixl_proxy_opcode_t::PUT;
            retired_submission.srcProxyMemViewId = src_proxy_id;
            retired_submission.dstProxyMemViewId = dst_proxy_id;
            retired_submission.size = 8;

            nixlBackendProxySubmission prepared_submission;
            EXPECT_EQ(registry.prepareSubmission(retired_submission, prepared_submission),
                      NIXL_ERR_NOT_FOUND);

            nixlProxySubmission live_submission{};
            live_submission.opcode = nixl_proxy_opcode_t::PUT;
            live_submission.srcProxyMemViewId = src_proxy_id;
            live_submission.dstProxyMemViewId = other_proxy_id;
            live_submission.size = 8;

            EXPECT_EQ(registry.prepareSubmission(live_submission, prepared_submission),
                      NIXL_SUCCESS);
        }

        TEST_F(proxyMemViewRegistryTest, StoreMetadataRejectsRetiredEntries) {
            nixlMemViewH proxy_handle = nullptr;
            ASSERT_EQ(registry.registerProxyMemView(makeFakeBackendHandle(10), &proxy_handle),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry.unregisterProxyMemView(proxy_handle), NIXL_SUCCESS);
            EXPECT_EQ(registry.storeMetadata(proxy_handle, makeLocalMetadata(0x1000)),
                      NIXL_ERR_NOT_FOUND);
        }

    } // namespace
} // namespace proxy_memview_registry
} // namespace gtest
