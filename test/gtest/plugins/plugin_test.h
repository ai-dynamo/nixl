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

#include "nixl.h"

#include "memory_handler.h"
#include "plugin_manager.h"
#include "backend_engine.h"

namespace plugin_test {

/*
 * Base class for all plugin tests.
 * Provides common functionality for all plugin tests.
 */
class SetupBackendTestFixture : public testing::TestWithParam<nixlBackendInitParams> {
private:
    static const size_t NUM_BUF_ENTRIES = 64;
    static const size_t BUF_ENTRY_SIZE = 1024 * 1024;
    static const size_t BUF_SIZE = BUF_ENTRY_SIZE * NUM_BUF_ENTRIES;

    std::unique_ptr<nixl_meta_dlist_t> reqSrcDescs_;
    std::unique_ptr<nixl_meta_dlist_t> reqDstDescs_;
    nixlBackendEngine *xferBackendEngine_;
    nixl_opt_b_args_t optionalXferArgs_;
    nixlBackendMD *xferLoadedMem_;
    std::string remoteAgent_;
    nixlBackendReqH *handle_;
    nixlBackendMD *localMem_;
    nixlBackendMD *xferMem_;
    std::string localAgent_;
    std::string xferAgent_;
    bool isSetup_ = false;
    void *localMemBuf_;
    void *xferMemBuf_;
    int xferDevId_;

    nixl_status_t
    backendAllocReg(nixlBackendEngine *engine,
                    nixl_mem_t mem_type,
                    size_t len,
                    void *&mem_buf,
                    nixlBackendMD *&md,
                    int dev_id);
    void *
    getMemBufPtr(nixl_mem_t mem_type, void *addr, size_t len);
    void
    releaseMemBufPtr(nixl_mem_t mem_type, void *addr);
    nixl_status_t
    backendDeregDealloc(nixlBackendEngine *engine,
                        nixl_mem_t mem_type,
                        void *mem_buf,
                        nixlBackendMD *&md,
                        int dev_id);
    void
    setBuf(nixl_mem_t mem_type, void *addr, char byte, int dev_id);
    void
    populateDescList(nixl_meta_dlist_t &descs, void *buf, nixlBackendMD *&md, int dev_id);

    bool
    VerifyConnInfo(bool is_remote);
    void
    SetupNotifs(std::string msg);
    bool
    PrepXferMem(nixl_mem_t mem_type, bool is_remote);
    bool
    VerifyNotifs(std::string &msg);
    bool
    VerifyXferData(void *src_mem_buf, void *dst_mem_buf);

protected:
    std::unique_ptr<nixlBackendEngine> remote_backend_engine_;
    std::unique_ptr<nixlBackendEngine> backend_engine_;

    void
    SetUp() override;
    void
    TearDown() override;

    bool
    SetupLocalXfer(nixl_mem_t mem_type);
    bool
    SetupRemoteXfer(nixl_mem_t mem_type);
    bool
    TestXfer(nixl_xfer_op_t op);
    bool
    VerifyXfer();
    bool
    TeardownXfer(nixl_mem_t mem_type);
    bool
    TestLocalXfer(nixl_mem_t mem_type, nixl_xfer_op_t op);
    bool
    TestRemoteXfer(nixl_mem_t mem_type, nixl_xfer_op_t op);
    bool
    TestGenNotif(std::string msg);
    bool
    IsLoaded() {
        return isSetup_;
    }
};


} // namespace plugin_test
#endif // __PLUGIN_TEST_H
