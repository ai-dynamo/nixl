/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include "nixl.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"

namespace gtest {
namespace libblkio {

    // ---------------------------------------------------------------------------
    // Fixture: constructs a nixlAgent and a LIBBLKIO backend for each test.
    // No real block device is required for these unit tests — they exercise
    // parameter handling, memory registration error paths, and API contracts.
    // ---------------------------------------------------------------------------
    class LibblkioBackendTest : public ::testing::Test {
    protected:
        static constexpr const char *agent_name = "libblkio_unit_test_agent";
        static constexpr uint64_t dev_id = 1;

        nixlAgent *agent_ = nullptr;
        nixlBackendH *backend_ = nullptr;

        void
        SetUp() override {
            agent_ = new nixlAgent(agent_name, nixlAgentConfig(true));
        }

        void
        TearDown() override {
            delete agent_;
            agent_ = nullptr;
            backend_ = nullptr;
        }

        nixl_status_t
        createBackend(const nixl_b_params_t &params) {
            return agent_->createBackend("LIBBLKIO", params, backend_);
        }

        nixl_b_params_t
        defaultParams(const std::string &device_list = "") {
            nixl_b_params_t params;
            params["api_type"] = "IO_URING";
            params["device_list"] = device_list;
            params["direct_io"] = "0";
            params["io_polling"] = "0";
            return params;
        }
    };

    // ---------------------------------------------------------------------------
    // Backend construction tests (no device required)
    // ---------------------------------------------------------------------------

    TEST_F(LibblkioBackendTest, CreateBackendSucceeds) {
        EXPECT_EQ(createBackend(defaultParams()), NIXL_SUCCESS);
        EXPECT_NE(backend_, nullptr);
    }

    TEST_F(LibblkioBackendTest, CreateBackendWithInvalidApiTypeDefaultsToIoUring) {
        auto params = defaultParams();
        params["api_type"] = "INVALID_API";
        EXPECT_EQ(createBackend(params), NIXL_SUCCESS);
        EXPECT_NE(backend_, nullptr);
    }

    TEST_F(LibblkioBackendTest, CreateBackendWithDirectIoFlag) {
        auto params = defaultParams();
        params["direct_io"] = "1";
        EXPECT_EQ(createBackend(params), NIXL_SUCCESS);
    }

    TEST_F(LibblkioBackendTest, CreateBackendWithIoPollingFlag) {
        auto params = defaultParams();
        params["io_polling"] = "1";
        EXPECT_EQ(createBackend(params), NIXL_SUCCESS);
    }

    TEST_F(LibblkioBackendTest, CreateBackendWithEmptyDeviceList) {
        EXPECT_EQ(createBackend(defaultParams("")), NIXL_SUCCESS);
    }

    // ---------------------------------------------------------------------------
    // DRAM memory registration tests (no device required)
    // ---------------------------------------------------------------------------

    class LibblkioDramRegTest : public LibblkioBackendTest {
    protected:
        static constexpr size_t buf_size = 4096;

        void *buf_ = nullptr;

        void
        SetUp() override {
            LibblkioBackendTest::SetUp();
            ASSERT_EQ(createBackend(defaultParams()), NIXL_SUCCESS);
            ASSERT_EQ(posix_memalign(&buf_, 4096, buf_size), 0);
        }

        void
        TearDown() override {
            free(buf_);
            LibblkioBackendTest::TearDown();
        }
    };

    TEST_F(LibblkioDramRegTest, RegisterDramMemSucceeds) {
        nixl_reg_dlist_t dlist(DRAM_SEG);
        nixlBlobDesc desc;
        desc.addr = reinterpret_cast<uintptr_t>(buf_);
        desc.len = buf_size;
        desc.devId = dev_id;
        desc.metaInfo = "";
        dlist.addDesc(desc);

        EXPECT_EQ(agent_->registerMem(dlist), NIXL_SUCCESS);
        EXPECT_EQ(agent_->deregisterMem(dlist), NIXL_SUCCESS);
    }

    TEST_F(LibblkioDramRegTest, DeregisterWithoutRegisterDoesNotCrash) {
        nixl_reg_dlist_t dlist(DRAM_SEG);
        nixlBlobDesc desc;
        desc.addr = reinterpret_cast<uintptr_t>(buf_);
        desc.len = buf_size;
        desc.devId = dev_id;
        desc.metaInfo = "";
        dlist.addDesc(desc);

        EXPECT_EQ(agent_->registerMem(dlist), NIXL_SUCCESS);
        EXPECT_EQ(agent_->deregisterMem(dlist), NIXL_SUCCESS);
        // Second deregister should not crash (idempotent or graceful error)
        agent_->deregisterMem(dlist);
    }

    TEST_F(LibblkioDramRegTest, RegisterMultipleDramDescriptors) {
        static constexpr int num_bufs = 4;
        std::vector<void *> bufs(num_bufs, nullptr);
        for (auto &b : bufs) {
            ASSERT_EQ(posix_memalign(&b, 4096, buf_size), 0);
        }

        nixl_reg_dlist_t dlist(DRAM_SEG);
        for (int i = 0; i < num_bufs; i++) {
            nixlBlobDesc desc;
            desc.addr = reinterpret_cast<uintptr_t>(bufs[i]);
            desc.len = buf_size;
            desc.devId = dev_id;
            desc.metaInfo = "";
            dlist.addDesc(desc);
        }

        EXPECT_EQ(agent_->registerMem(dlist), NIXL_SUCCESS);
        EXPECT_EQ(agent_->deregisterMem(dlist), NIXL_SUCCESS);

        for (auto &b : bufs) {
            free(b);
        }
    }

    // ---------------------------------------------------------------------------
    // BLK_SEG registration error path: no device in device_list
    // ---------------------------------------------------------------------------

    TEST_F(LibblkioBackendTest, RegisterBlkSegWithEmptyDeviceListFails) {
        ASSERT_EQ(createBackend(defaultParams("")), NIXL_SUCCESS);

        nixl_reg_dlist_t dlist(BLK_SEG);
        nixlBlobDesc desc;
        desc.addr = 0;
        desc.len = 512 * 1024;
        desc.devId = dev_id;
        desc.metaInfo = "";
        dlist.addDesc(desc);

        EXPECT_NE(agent_->registerMem(dlist), NIXL_SUCCESS);
    }

} // namespace libblkio
} // namespace gtest
