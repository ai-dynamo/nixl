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
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <thread>
#include "nixl.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"

namespace gtest {
namespace libblkio {

    // ---------------------------------------------------------------------------
    // Integration test fixture.
    //
    // Requires a real block device specified via the NIXL_LIBBLKIO_TEST_DEVICE
    // environment variable. Tests in this fixture are automatically skipped if
    // the variable is unset or the device cannot be opened.
    // ---------------------------------------------------------------------------
    class LibblkioIntegrationTest : public ::testing::Test {
    protected:
        static constexpr size_t transfer_size = 512 * 1024; // 512 KB
        static constexpr uint64_t dev_id = 1;

        std::string device_path_;
        nixlAgent *agent_ = nullptr;
        nixlBackendH *backend_ = nullptr;
        void *write_buf_ = nullptr;
        void *read_buf_ = nullptr;
        nixl_reg_dlist_t *write_dlist_ = nullptr;
        nixl_reg_dlist_t *read_dlist_ = nullptr;
        nixl_reg_dlist_t *blk_dlist_ = nullptr;

        void
        SetUp() override {
            const char *dev = std::getenv("NIXL_LIBBLKIO_TEST_DEVICE");
            if (!dev || *dev == '\0') {
                GTEST_SKIP() << "Set NIXL_LIBBLKIO_TEST_DEVICE to a disposable block device";
            }
            device_path_ = dev;

            agent_ = new nixlAgent("libblkio_integration_test_agent", nixlAgentConfig(true));

            nixl_b_params_t params;
            params["api_type"] = "IO_URING";
            params["device_list"] = "1:B:" + device_path_;
            params["direct_io"] = "0";
            params["io_polling"] = "0";

            nixl_status_t status = agent_->createBackend("LIBBLKIO", params, backend_);
            if (status != NIXL_SUCCESS) {
                GTEST_SKIP() << "Failed to create LIBBLKIO backend (device unavailable?): "
                             << status;
            }

            const long page_size = sysconf(_SC_PAGESIZE);
            ASSERT_EQ(posix_memalign(&write_buf_, page_size, transfer_size), 0);
            ASSERT_EQ(posix_memalign(&read_buf_, page_size, transfer_size), 0);

            write_dlist_ = new nixl_reg_dlist_t(DRAM_SEG);
            read_dlist_ = new nixl_reg_dlist_t(DRAM_SEG);
            blk_dlist_ = new nixl_reg_dlist_t(BLK_SEG);

            nixlBlobDesc wd;
            wd.addr = reinterpret_cast<uintptr_t>(write_buf_);
            wd.len = transfer_size;
            wd.devId = dev_id;
            wd.metaInfo = "";
            write_dlist_->addDesc(wd);

            nixlBlobDesc rd;
            rd.addr = reinterpret_cast<uintptr_t>(read_buf_);
            rd.len = transfer_size;
            rd.devId = dev_id;
            rd.metaInfo = "";
            read_dlist_->addDesc(rd);

            nixlBlobDesc bd;
            bd.addr = 0;
            bd.len = transfer_size;
            bd.devId = dev_id;
            bd.metaInfo = device_path_;
            blk_dlist_->addDesc(bd);

            ASSERT_EQ(agent_->registerMem(*write_dlist_), NIXL_SUCCESS);
            ASSERT_EQ(agent_->registerMem(*read_dlist_), NIXL_SUCCESS);
            ASSERT_EQ(agent_->registerMem(*blk_dlist_), NIXL_SUCCESS);
        }

        void
        TearDown() override {
            if (agent_) {
                if (write_dlist_) agent_->deregisterMem(*write_dlist_);
                if (read_dlist_) agent_->deregisterMem(*read_dlist_);
                if (blk_dlist_) agent_->deregisterMem(*blk_dlist_);
            }
            delete write_dlist_;
            delete read_dlist_;
            delete blk_dlist_;
            free(write_buf_);
            free(read_buf_);
            delete agent_;
        }

        void
        fillPattern(void *buf, size_t size, char pattern) {
            memset(buf, pattern, size);
        }
    };

    // ---------------------------------------------------------------------------
    // Write then read back, verify data integrity
    // ---------------------------------------------------------------------------

    TEST_F(LibblkioIntegrationTest, WriteReadVerify) {
        fillPattern(write_buf_, transfer_size, 0xAB);
        memset(read_buf_, 0, transfer_size);

        nixl_xfer_dlist_t write_xfer(DRAM_SEG);
        nixlBasicDesc wd;
        wd.addr = reinterpret_cast<uintptr_t>(write_buf_);
        wd.len = transfer_size;
        wd.devId = dev_id;
        write_xfer.addDesc(wd);

        nixl_xfer_dlist_t blk_xfer(BLK_SEG);
        nixlBasicDesc bd;
        bd.addr = 0;
        bd.len = transfer_size;
        bd.devId = dev_id;
        blk_xfer.addDesc(bd);

        nixl_opt_args_t opt_args;
        opt_args.backends.push_back(backend_);

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_->createXferReq(NIXL_WRITE, write_xfer, blk_xfer, "", req, &opt_args),
                  NIXL_SUCCESS);
        ASSERT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);

        // Poll for completion with timeout
        constexpr int timeout_ms = 5000;
        constexpr int poll_interval_ms = 10;
        int elapsed_ms = 0;
        nixl_status_t status;
        while (elapsed_ms < timeout_ms) {
            status = agent_->getXferStatus(req);
            if (status == NIXL_SUCCESS) {
                break;
            } else if (status != NIXL_IN_PROG) {
                FAIL() << "Write transfer failed with status: " << status;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
            elapsed_ms += poll_interval_ms;
        }
        EXPECT_EQ(status, NIXL_SUCCESS) << "Write transfer timed out after " << timeout_ms << "ms";
        agent_->releaseXferReq(req);

        req = nullptr;
        nixl_xfer_dlist_t read_xfer(DRAM_SEG);
        nixlBasicDesc rdd;
        rdd.addr = reinterpret_cast<uintptr_t>(read_buf_);
        rdd.len = transfer_size;
        rdd.devId = dev_id;
        read_xfer.addDesc(rdd);

        ASSERT_EQ(agent_->createXferReq(NIXL_READ, read_xfer, blk_xfer, "", req, &opt_args),
                  NIXL_SUCCESS);
        ASSERT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);

        // Poll for completion with timeout
        elapsed_ms = 0;
        while (elapsed_ms < timeout_ms) {
            status = agent_->getXferStatus(req);
            if (status == NIXL_SUCCESS) {
                break;
            } else if (status != NIXL_IN_PROG) {
                FAIL() << "Read transfer failed with status: " << status;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
            elapsed_ms += poll_interval_ms;
        }
        EXPECT_EQ(status, NIXL_SUCCESS) << "Read transfer timed out after " << timeout_ms << "ms";
        agent_->releaseXferReq(req);

        EXPECT_EQ(memcmp(write_buf_, read_buf_, transfer_size), 0);
    }

    TEST_F(LibblkioIntegrationTest, WriteReadVerifyDistinctPatterns) {
        static constexpr char pattern1 = 0x55;
        static constexpr char pattern2 = 0xAA;

        fillPattern(write_buf_, transfer_size, pattern1);
        memset(read_buf_, 0, transfer_size);

        nixl_xfer_dlist_t write_xfer(DRAM_SEG);
        nixlBasicDesc wd;
        wd.addr = reinterpret_cast<uintptr_t>(write_buf_);
        wd.len = transfer_size;
        wd.devId = dev_id;
        write_xfer.addDesc(wd);

        nixl_xfer_dlist_t blk_xfer(BLK_SEG);
        nixlBasicDesc bd;
        bd.addr = 0;
        bd.len = transfer_size;
        bd.devId = dev_id;
        blk_xfer.addDesc(bd);

        nixl_opt_args_t opt_args;
        opt_args.backends.push_back(backend_);

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_->createXferReq(NIXL_WRITE, write_xfer, blk_xfer, "", req, &opt_args),
                  NIXL_SUCCESS);
        ASSERT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);

        // Poll for completion with timeout
        constexpr int timeout_ms2 = 5000;
        constexpr int poll_interval_ms2 = 10;
        int elapsed_ms2 = 0;
        nixl_status_t status2;
        while (elapsed_ms2 < timeout_ms2) {
            status2 = agent_->getXferStatus(req);
            if (status2 == NIXL_SUCCESS) {
                break;
            } else if (status2 != NIXL_IN_PROG) {
                FAIL() << "Write transfer failed with status: " << status2;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms2));
            elapsed_ms2 += poll_interval_ms2;
        }
        EXPECT_EQ(status2, NIXL_SUCCESS)
            << "Write transfer timed out after " << timeout_ms2 << "ms";
        agent_->releaseXferReq(req);

        // Overwrite with a second pattern to confirm read returns what was written
        fillPattern(write_buf_, transfer_size, pattern2);

        req = nullptr;
        nixl_xfer_dlist_t read_xfer(DRAM_SEG);
        nixlBasicDesc rdd;
        rdd.addr = reinterpret_cast<uintptr_t>(read_buf_);
        rdd.len = transfer_size;
        rdd.devId = dev_id;
        read_xfer.addDesc(rdd);

        ASSERT_EQ(agent_->createXferReq(NIXL_READ, read_xfer, blk_xfer, "", req, &opt_args),
                  NIXL_SUCCESS);
        ASSERT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);

        // Poll for completion with timeout
        elapsed_ms2 = 0;
        while (elapsed_ms2 < timeout_ms2) {
            status2 = agent_->getXferStatus(req);
            if (status2 == NIXL_SUCCESS) {
                break;
            } else if (status2 != NIXL_IN_PROG) {
                FAIL() << "Read transfer failed with status: " << status2;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms2));
            elapsed_ms2 += poll_interval_ms2;
        }
        EXPECT_EQ(status2, NIXL_SUCCESS) << "Read transfer timed out after " << timeout_ms2 << "ms";
        agent_->releaseXferReq(req);

        // read_buf_ should contain pattern1, not pattern2
        const char *rbuf = static_cast<const char *>(read_buf_);
        for (size_t i = 0; i < transfer_size; i++) {
            ASSERT_EQ(rbuf[i], pattern1) << "Mismatch at byte " << i;
        }
    }

} // namespace libblkio
} // namespace gtest
