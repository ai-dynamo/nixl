/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef __PLUGIN_TEST_H
#define __PLUGIN_TEST_H

#include <gtest/gtest.h>
#include <functional>

#include "nixl.h"

#include "memory_handler.h"
#include "plugin_manager.h"
#include "backend_engine.h"

namespace plugins_common {

/*
 * Base class for all plugin tests.
 * Provides common functionality for all plugin tests.
 */
class SetupBackendTestFixture : public testing::TestWithParam<nixlBackendInitParams> {
protected:
    std::unique_ptr<nixlBackendEngine> remote_backend_engine_;
    std::unique_ptr<nixlBackendEngine> backend_engine_;

    void
    SetUp() override;

    void
    resetLocalBuf();

    bool
    checkLocalBuf();

    bool
    setupLocalXfer(nixl_mem_t local_mem_type,
                   nixl_mem_t xfer_mem_type,
                   bool split_buf = true,
                   int num_bufs = 1);

    bool
    setupRemoteXfer(nixl_mem_t local_mem_type,
                    nixl_mem_t xfer_mem_type,
                    bool split_buf = true,
                    int num_bufs = 1);

    bool
    testXfer(nixl_xfer_op_t op);

    bool
    verifyXfer();

    bool
    teardownXfer();

    bool
    testLocalXfer(nixl_xfer_op_t op);

    bool
    testRemoteXfer(nixl_xfer_op_t op);

    bool
    testGenNotif(std::string msg);

    bool
    isLoaded() {
        return isSetup_;
    }

private:
    static const char LOCAL_BUF_BYTE = 0x11;
    static const char XFER_BUF_BYTE = 0x22;
    static const size_t NUM_ENTRIES = 4;
    static const size_t ENTRY_SIZE = 16;
    static const size_t BUF_SIZE = NUM_ENTRIES * ENTRY_SIZE;
    static const size_t MAX_NUM_BUFS = 3;

    std::unique_ptr<MemoryHandler> localMemHandler_[MAX_NUM_BUFS];
    std::unique_ptr<MemoryHandler> xferMemHandler_[MAX_NUM_BUFS];
    std::unique_ptr<nixl_meta_dlist_t> reqSrcDescs_;
    std::unique_ptr<nixl_meta_dlist_t> reqDstDescs_;
    nixlBackendEngine *xferBackendEngine_;
    nixl_opt_b_args_t optionalXferArgs_;
    nixlBackendMD *xferLoadedMD_;
    std::string remoteAgent_;
    nixlBackendReqH *handle_;
    std::string localAgent_;
    std::string xferAgent_;
    bool isSetup_ = false;
    bool split_buf_;
    int localDevId_;
    int xferDevId_;
    int num_bufs_;

    nixl_status_t
    backendAllocReg(nixlBackendEngine *engine,
                    nixl_mem_t mem_type,
                    size_t len,
                    std::unique_ptr<MemoryHandler> &mem_handler,
                    int buf_index,
                    int dev_id);

    nixl_status_t
    backendDeregDealloc(nixlBackendEngine *engine, std::unique_ptr<MemoryHandler> &mem_handler);

    void
    populateDescList(nixl_meta_dlist_t &descs, std::unique_ptr<MemoryHandler> mem_handler[]);

    bool
    verifyConnInfo();

    void
    setupNotifs(std::string msg);

    bool
    prepXferMem(nixl_mem_t local_mem_type, nixl_mem_t xfer_mem_type, bool is_remote);

    bool
    verifyNotifs(std::string &msg);
};


} // namespace plugins_common
#endif // __PLUGIN_TEST_H
