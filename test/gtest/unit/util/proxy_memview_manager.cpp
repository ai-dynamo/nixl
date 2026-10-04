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

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "device/proxy/proxy_memview_manager.h"
#include "device/proxy/proxy_submission.h"
#include "mocks/proxy_mocks.h"

namespace gtest {
namespace proxy_memview_manager {

    using proxy_mocks::DummyBackendMD;
    using proxy_mocks::makeLocalDlist;
    using proxy_mocks::makeRemoteDesc;
    using proxy_mocks::MockDeviceOps;

    class ProxyMemViewManagerTest : public testing::Test {
    protected:
        /** Device memory is host memory under the mock: the view is readable in place. */
        static const nixlProxyDeviceMemView &
        view(nixlMemViewH handle) {
            return *static_cast<const nixlProxyDeviceMemView *>(handle);
        }

        static uint64_t
        tokenOf(nixlMemViewH handle) {
            return handle == nullptr ? 0 : view(handle).host_view;
        }

        nixlMemViewH
        prepLocal(uintptr_t addr, size_t len = 64, uint64_t dev_id = 0) {
            nixlMemViewH handle = nullptr;
            EXPECT_EQ(manager_.prepLocal(makeLocalDlist(addr, len, dev_id, &local_md_), handle),
                      NIXL_SUCCESS);
            return handle;
        }

        nixlMemViewH
        prepRemote(const nixl_remote_meta_dlist_t &dlist,
                   const std::vector<void *> &direct_ptrs = {}) {
            nixlMemViewH handle = nullptr;
            EXPECT_EQ(manager_.prepRemote(dlist, direct_ptrs, handle), NIXL_SUCCESS);
            return handle;
        }

        nixlMemViewH
        prepRemote(const std::string &agent = "peer",
                   uintptr_t addr = 0x2000,
                   size_t len = 64,
                   uint64_t dev_id = 0) {
            nixl_remote_meta_dlist_t dlist(VRAM_SEG);
            dlist.addDesc(makeRemoteDesc(agent, addr, len, dev_id, &remote_md_));
            return prepRemote(dlist);
        }

        static nixlProxyCommand
        put(uint64_t src,
            uint64_t dst,
            uint64_t size,
            uint64_t src_offset = 0,
            uint64_t dst_offset = 0) {
            nixlProxyCommand record{};
            record.opcode = nixl_proxy_opcode_t::PUT;
            record.src_view = src;
            record.operand = src_offset;
            record.dst_view = dst;
            record.dst_offset = dst_offset;
            record.size = size;
            return record;
        }

        static nixlProxyCommand
        atomicAdd(uint64_t dst, uint64_t dst_offset = 0, uint64_t value = 42) {
            nixlProxyCommand record{};
            record.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            record.dst_view = dst;
            record.dst_offset = dst_offset;
            record.size = sizeof(uint64_t);
            record.operand = value;
            return record;
        }

        nixl_status_t
        prepare(const nixlProxyCommand &command,
                nixl::proxyBackendSubmission &prepared,
                uint32_t channel = 0) {
            return nixl::resolveSubmission(command, channel, /*peer=*/0, prepared);
        }

        MockDeviceOps allocator_;
        nixlProxyDeviceContextData context_{};
        nixl::proxyMemViewManager manager_{allocator_, &context_};
        DummyBackendMD local_md_;
        DummyBackendMD remote_md_;
    };

    TEST_F(ProxyMemViewManagerTest, PrepareSubmissionResolvesAndValidates) {
        const uint64_t src = tokenOf(prepLocal(0x1000, 64, /*dev_id=*/7));
        const uint64_t dst = tokenOf(prepRemote("remote-agent", 0x2000, 64, /*dev_id=*/11));
        const uint64_t empty = tokenOf(prepRemote(nixl_remote_meta_dlist_t(VRAM_SEG)));
        constexpr uint64_t kLarge = (uint64_t{1} << 32) + 64;
        const uint64_t big_src = tokenOf(prepLocal(0x5000, kLarge + 64));
        const uint64_t big_dst = tokenOf(prepRemote("peer", 0x6000, kLarge + 64));

        nixl::proxyBackendSubmission prepared;
        nixlProxyCommand record = put(src, dst, 16, 5, 9);
        record.op_idx = 7;
        ASSERT_EQ(prepare(record, prepared, /*channel=*/3), NIXL_SUCCESS);
        EXPECT_EQ(prepared.op_idx, 7u);
        EXPECT_EQ(prepared.channel_id, 3u);
        EXPECT_EQ(prepared.opcode, nixl_proxy_opcode_t::PUT);
        EXPECT_EQ(prepared.size, 16u);
        EXPECT_EQ(prepared.local, nixlMetaDesc(0x1005, 16, 7, &local_md_));
        EXPECT_EQ(prepared.remote, nixlMetaDesc(0x2009, 16, 11, &remote_md_));

        record = atomicAdd(dst, 9, 42);
        record.size = 3;
        ASSERT_EQ(prepare(record, prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
        EXPECT_EQ(prepared.size, sizeof(uint64_t));
        EXPECT_EQ(prepared.remote.addr, 0x2009u);
        EXPECT_EQ(prepared.remote.len, sizeof(uint64_t));
        EXPECT_EQ(prepared.value, 42u);

        ASSERT_EQ(prepare(put(big_src, big_dst, 32, kLarge, kLarge), prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.local.addr, uintptr_t{0x5000} + kLarge);
        EXPECT_EQ(prepared.remote.addr, uintptr_t{0x6000} + kLarge);
        ASSERT_EQ(prepare(put(big_src, big_dst, kLarge), prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.size, kLarge);
        EXPECT_EQ(prepared.local.len, kLarge);

        ASSERT_EQ(prepare(put(src, dst, 16, 48, 48), prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.local.addr, 0x1030u);
        EXPECT_EQ(prepared.remote.addr, 0x2030u);

        nixlProxyCommand unsupported = atomicAdd(dst);
        unsupported.opcode = static_cast<nixl_proxy_opcode_t>(99);

        struct Row {
            const char *name;
            nixlProxyCommand record;
            nixl_status_t expected;
        };

        const std::vector<Row> rows = {
            {"source past the end", put(src, dst, 8, 60, 0), NIXL_ERR_INVALID_PARAM},
            {"destination past the end", put(src, dst, 8, 0, 60), NIXL_ERR_INVALID_PARAM},
            {"offset far past the end",
             put(src, dst, 1, 0, std::numeric_limits<uint64_t>::max()),
             NIXL_ERR_INVALID_PARAM},
            {"counter past the end", atomicAdd(dst, 60), NIXL_ERR_INVALID_PARAM},
            {"roles swapped", put(dst, src, 16), NIXL_ERR_INVALID_PARAM},
            {"empty descriptor list", atomicAdd(empty), NIXL_ERR_INVALID_PARAM},
            {"null tokens", put(0, 0, 16), NIXL_ERR_NOT_FOUND},
            {"unsupported opcode", unsupported, NIXL_ERR_NOT_SUPPORTED},
        };
        for (const auto &row : rows) {
            prepared.op_idx = 123;
            EXPECT_EQ(prepare(row.record, prepared), row.expected) << row.name;
            EXPECT_EQ(prepared.op_idx, 123u) << row.name;
        }
    }

    TEST_F(ProxyMemViewManagerTest, ViewsPreserveContextPointersAndDescriptorOrder) {
        const nixlMemViewH src = prepLocal(0x1000);
        EXPECT_EQ(view(src).direct_ptr_count, 0u);
        EXPECT_EQ(view(src).context, &context_);

        DummyBackendMD peer1_md;
        nixl_remote_meta_dlist_t peers(VRAM_SEG);
        peers.addDesc(makeRemoteDesc("peer0", 0x4000, 64, 0, &remote_md_));
        peers.addDesc(makeRemoteDesc("peer1", 0x5000, 64, 1, &peer1_md));
        const std::vector<void *> direct_ptrs{reinterpret_cast<void *>(uintptr_t{0xfeed0000}),
                                              nullptr};
        const nixlMemViewH dst = prepRemote(peers, direct_ptrs);
        ASSERT_EQ(view(dst).direct_ptr_count, 2u);
        void *const *stored = nixlProxyDeviceMemViewDirectPtrs(&view(dst));
        EXPECT_EQ(std::vector<void *>(stored, stored + 2), direct_ptrs);

        nixl_remote_meta_dlist_t reversed(VRAM_SEG);
        reversed.addDesc(peers[1]);
        reversed.addDesc(peers[0]);
        for (auto handle : {dst, prepRemote(reversed)}) {
            for (uint64_t index = 0; index < 2; ++index) {
                auto record = put(tokenOf(src), tokenOf(handle), 8);
                record.dst_index = index;
                auto expected = static_cast<nixlMetaDesc>(peers[handle == dst ? index : 1 - index]);
                expected.len = record.size;
                nixl::proxyBackendSubmission prepared;
                ASSERT_EQ(prepare(record, prepared), NIXL_SUCCESS);
                EXPECT_EQ(prepared.remote, expected);
            }
        }
    }

    TEST_F(ProxyMemViewManagerTest, FailedPreparationRollsBack) {
        nixl_remote_meta_dlist_t dlist(VRAM_SEG);
        dlist.addDesc(makeRemoteDesc("peer", 0x2000, 64, 0, &remote_md_));
        nixlMemViewH handle = &context_;
        for (int fail_after : {0, 1, 2}) { // Allocation, header copy, direct-pointer copy.
            SCOPED_TRACE(fail_after);
            allocator_.fail_after = fail_after;
            EXPECT_EQ(manager_.prepRemote(dlist, {nullptr}, handle), NIXL_ERR_BACKEND);
            EXPECT_EQ(handle, &context_);
            EXPECT_EQ(allocator_.liveAllocations(), 0u);
        }
        nixl_remote_meta_dlist_t dram(DRAM_SEG);
        dram.addDesc(dlist[0]);
        EXPECT_EQ(manager_.prepRemote(dram, {}, handle), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(handle, &context_);
        EXPECT_EQ(allocator_.liveAllocations(), 0u);
    }

    TEST_F(ProxyMemViewManagerTest, RemoteHolesKeepTheirIndices) {
        nixl_remote_meta_dlist_t dlist(VRAM_SEG);
        dlist.addDesc(nixlRemoteMetaDesc(nixl_null_agent));
        dlist.addDesc(makeRemoteDesc("peer", 0x2000, 64, 7, &remote_md_));
        dlist.addDesc(nixlRemoteMetaDesc(""));
        const auto token = tokenOf(prepRemote(dlist));
        nixl::proxyBackendSubmission prepared;
        for (uint64_t index = 0; index < 3; ++index) {
            auto record = atomicAdd(token, 8);
            record.dst_index = index;
            EXPECT_EQ(prepare(record, prepared),
                      index == 1 ? NIXL_SUCCESS : NIXL_ERR_INVALID_PARAM);
        }
        EXPECT_EQ(prepared.remote, nixlMetaDesc(0x2008, 8, 7, &remote_md_));
        auto record = atomicAdd(token, 16);
        record.dst_index = 1;
        ASSERT_EQ(prepare(record, prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.remote, nixlMetaDesc(0x2010, 8, 7, &remote_md_));
    }

} // namespace proxy_memview_manager
} // namespace gtest
