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
private:
    static const char LOCAL_BUF_BYTE = 0x11;
    static const char XFER_BUF_BYTE = 0x22;
    static const size_t BUF_SIZE = 4;

    std::unique_ptr<nixl_meta_dlist_t> reqSrcDescs_;
    std::unique_ptr<nixl_meta_dlist_t> reqDstDescs_;
    std::function<void()> resetLocalBufCallback_;
    std::function<bool()> teardownCallback_;
    nixlBackendEngine *xferBackendEngine_;
    nixl_opt_b_args_t optionalXferArgs_;
    nixlBackendMD *xferLoadedMem_;
    std::string remoteAgent_;
    nixlBackendReqH *handle_;
    nixlBackendMD *localMD_;
    nixlBackendMD *xferMD_;
    std::string localAgent_;
    std::string xferAgent_;
    bool isSetup_ = false;
    void *localMemBuf_;
    void *xferMemBuf_;
    int localDevId_;
    int xferDevId_;

    template<nixl_mem_t MemType>
    nixl_status_t
    backendAllocReg(nixlBackendEngine *engine,
                    size_t len,
                    void *&mem_buf,
                    nixlBackendMD *&md,
                    int dev_id);
   
    template<nixl_mem_t MemType>
    nixl_status_t
    backendDeregDealloc(nixlBackendEngine *engine,
                        void *mem_buf,
                        nixlBackendMD *&md,
                        int dev_id);
    
    template<nixl_mem_t MemType>
    void
    populateDescList(nixl_meta_dlist_t &descs, void *buf, nixlBackendMD *&md, int dev_id);

    bool
    VerifyConnInfo();
    
    void
    SetupNotifs(std::string msg);
    
    template<nixl_mem_t LocalMemType, nixl_mem_t XferMemType>
    bool
    PrepXferMem(bool is_remote);
    
    bool
    VerifyNotifs(std::string &msg);

    template<nixl_mem_t LocalMemType, nixl_mem_t XferMemType>
    bool
    DeregDeallocCallback();
    
protected:
    std::unique_ptr<nixlBackendEngine> remote_backend_engine_;
    std::unique_ptr<nixlBackendEngine> backend_engine_;

    void
    SetUp() override;

    void
    ResetLocalBuf();

    bool
    CheckLocalBuf();

    template<nixl_mem_t LocalMemType, nixl_mem_t XferMemType>
    bool
    SetupLocalXfer();

    template<nixl_mem_t LocalMemType, nixl_mem_t XferMemType>
    bool
    SetupRemoteXfer();

    bool
    TestXfer(nixl_xfer_op_t op);

    bool
    VerifyXfer();

    bool
    TeardownXfer();

    bool
    TestLocalXfer(nixl_xfer_op_t op);

    bool
    TestRemoteXfer(nixl_xfer_op_t op);

    bool
    TestGenNotif(std::string msg);

    bool
    IsLoaded() {
        return isSetup_;
    }
};


} // namespace plugins_common
#endif // __PLUGIN_TEST_H
