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
    // Helper: poll a transfer request to completion with timeout.
    // ---------------------------------------------------------------------------
    static nixl_status_t
    pollToCompletion(nixlAgent *agent, nixlXferReqH *req, int timeout_ms = 5000) {
        constexpr int poll_interval_ms = 10;
        int elapsed_ms = 0;
        nixl_status_t status;
        while (elapsed_ms < timeout_ms) {
            status = agent->getXferStatus(req);
            if (status != NIXL_IN_PROG) {
                return status;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
            elapsed_ms += poll_interval_ms;
        }
        return NIXL_IN_PROG; // timed out
    }

    // ---------------------------------------------------------------------------
    // Integration test fixture.
    //
    // Requires a real block device specified via the NIXL_LIBBLKIO_TEST_DEVICE
    // environment variable. Tests in this fixture are automatically skipped if
    // the variable is unset or the device cannot be opened.
    // ---------------------------------------------------------------------------
    class LibblkioIntegrationTest : public ::testing::Test {
    protected:
        static constexpr const char *agent_name = "libblkio_integration_test_agent";
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

            const char *api_env = std::getenv("NIXL_LIBBLKIO_API_TYPE");
            std::string api_type = (api_env && *api_env != '\0') ? api_env : "IO_URING";

            agent_ = new nixlAgent(agent_name, nixlAgentConfig(true));

            nixl_b_params_t params;
            params["api_type"] = api_type;
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
                if (write_dlist_) {
                    agent_->deregisterMem(*write_dlist_);
                }
                if (read_dlist_) {
                    agent_->deregisterMem(*read_dlist_);
                }
                if (blk_dlist_) {
                    agent_->deregisterMem(*blk_dlist_);
                }
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

        /** Post a transfer and poll to completion. */
        nixl_status_t
        doTransfer(nixl_xfer_op_t op, void *dram_buf, size_t len, uint64_t blk_offset) {
            nixl_xfer_dlist_t dram_xfer(DRAM_SEG);
            nixlBasicDesc dd;
            dd.addr = reinterpret_cast<uintptr_t>(dram_buf);
            dd.len = len;
            dd.devId = dev_id;
            dram_xfer.addDesc(dd);

            nixl_xfer_dlist_t blk_xfer(BLK_SEG);
            nixlBasicDesc bd;
            bd.addr = blk_offset;
            bd.len = len;
            bd.devId = dev_id;
            blk_xfer.addDesc(bd);

            nixl_opt_args_t opt_args;
            opt_args.backends.push_back(backend_);

            nixlXferReqH *req = nullptr;
            nixl_status_t status =
                agent_->createXferReq(op, dram_xfer, blk_xfer, agent_name, req, &opt_args);
            if (status != NIXL_SUCCESS) {
                return status;
            }

            status = agent_->postXferReq(req);
            // postXfer may return NIXL_IN_PROG (async) or NIXL_SUCCESS (sync)
            if (status == NIXL_IN_PROG) {
                status = pollToCompletion(agent_, req);
            }
            agent_->releaseXferReq(req);
            return status;
        }
    };

    // ---------------------------------------------------------------------------
    // Write then read back, verify data integrity
    // ---------------------------------------------------------------------------

    TEST_F(LibblkioIntegrationTest, WriteReadVerify) {
        fillPattern(write_buf_, transfer_size, 0xAB);
        memset(read_buf_, 0, transfer_size);

        ASSERT_EQ(doTransfer(NIXL_WRITE, write_buf_, transfer_size, 0), NIXL_SUCCESS);
        ASSERT_EQ(doTransfer(NIXL_READ, read_buf_, transfer_size, 0), NIXL_SUCCESS);

        EXPECT_EQ(memcmp(write_buf_, read_buf_, transfer_size), 0);
    }

    TEST_F(LibblkioIntegrationTest, WriteReadVerifyDistinctPatterns) {
        static constexpr char pattern1 = 0x55;
        static constexpr char pattern2 = 0xAA;

        // Write pattern1 to the device
        fillPattern(write_buf_, transfer_size, pattern1);
        ASSERT_EQ(doTransfer(NIXL_WRITE, write_buf_, transfer_size, 0), NIXL_SUCCESS);

        // Overwrite write_buf_ with pattern2 to confirm read returns pattern1
        fillPattern(write_buf_, transfer_size, pattern2);

        // Read back
        memset(read_buf_, 0, transfer_size);
        ASSERT_EQ(doTransfer(NIXL_READ, read_buf_, transfer_size, 0), NIXL_SUCCESS);

        // read_buf_ should contain pattern1, not pattern2
        const char *rbuf = static_cast<const char *>(read_buf_);
        for (size_t i = 0; i < transfer_size; i++) {
            ASSERT_EQ(rbuf[i], pattern1) << "Mismatch at byte " << i;
        }
    }

} // namespace libblkio
} // namespace gtest
