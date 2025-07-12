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

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <absl/strings/str_format.h>

#include "nixl.h"
#include "common/nixl_log.h"
#include "plugin_manager.h"
#include "backend_engine.h"
#include "plugins_common.h"

namespace plugins_common {

template<nixl_mem_t MemType>
nixl_status_t
SetupBackendTestFixture::backendAllocReg(nixlBackendEngine *engine,
                                        size_t len,
                                        void *&mem_buf,
                                        nixlBackendMD *&md,
                                        int dev_id) {
    nixlBlobDesc desc;
    nixl_status_t ret;

    try {
        mem_buf = MemoryHandler<MemType>::allocate(len);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to allocate memory: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    MemoryHandler<MemType>::populateBlobDesc(&desc, mem_buf, len, dev_id);

    NIXL_INFO << "Registering memory type " << MemType << " at address " << mem_buf
            << " with length " << len << " and device ID " << dev_id;

    ret = engine->registerMem(desc, MemType, md);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to register memory: " << ret;
        return ret;
    }

    return NIXL_SUCCESS;
}

template<nixl_mem_t MemType>
nixl_status_t
SetupBackendTestFixture::backendDeregDealloc(nixlBackendEngine *engine,
                                             void *mem_buf,
                                             nixlBackendMD *&md,
                                             int dev_id) {
    nixl_status_t ret;

    NIXL_INFO << "Deregistering memory type " << MemType << " at address " << mem_buf
              << " with device ID " << dev_id;

    ret = engine->deregisterMem(md);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deregister memory: " << ret;
        return ret;
    }

    MemoryHandler<MemType>::deallocate(mem_buf);

    return NIXL_SUCCESS;
}

void
SetupBackendTestFixture::ResetLocalBuf() {
    if (resetLocalBufCallback_) {
        resetLocalBufCallback_();
    }
}

bool
SetupBackendTestFixture::CheckLocalBuf() {
    for (size_t i = 0; i < BUF_SIZE; i++) {
        uint8_t expected_byte = (uint8_t)LOCAL_BUF_BYTE + i;
        if (((uint8_t *)localMemBuf_)[i] != expected_byte) {
            NIXL_ERROR << absl::StrFormat("Verification failed at index %d! local: %x, expected: %x",
                i, ((uint8_t *)localMemBuf_)[i], expected_byte);
            return false;
        }
    }
    NIXL_INFO << "OK";
    return true;
}

template<nixl_mem_t MemType>
void
SetupBackendTestFixture::populateDescList(nixl_meta_dlist_t &descs,
                                          void *buf,
                                          nixlBackendMD *&md,
                                          int dev_id) {
    nixlMetaDesc req;
    MemoryHandler<MemType>::populateMetaDesc(&req, buf, BUF_SIZE, dev_id, md);
    descs.addDesc(req);
}

void
SetupBackendTestFixture::SetUp() {
    if (backend_engine_->getInitErr()) {
        CHECK(false) << "Failed to initialize backend engine";
    }

    localAgent_ = "Agent1";
    remoteAgent_ = "Agent2";

    isSetup_ = true;
}

bool
SetupBackendTestFixture::VerifyConnInfo() {
    std::string conn_info;
    nixl_status_t ret;

    ret = backend_engine_->getConnInfo(conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get connection info";
        return false;
    }

    ret = xferBackendEngine_->getConnInfo(conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get remote connection info";
        return false;
    }

    ret = backend_engine_->loadRemoteConnInfo(xferAgent_, conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to load remote connection info";
        return false;
    }

    return true;
}

void
SetupBackendTestFixture::SetupNotifs(std::string msg) {
    optionalXferArgs_.notifMsg = msg;
    optionalXferArgs_.hasNotif = true;
}

template<nixl_mem_t LocalMemType, nixl_mem_t XferMemType>
bool
SetupBackendTestFixture::PrepXferMem(bool is_remote) {
    nixl_status_t ret;

    ret = backendAllocReg<LocalMemType>
            (backend_engine_.get(), BUF_SIZE, localMemBuf_, localMD_, localDevId_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to register local memory";
        return false;
    }

    ret = backendAllocReg<XferMemType>(
            xferBackendEngine_, BUF_SIZE, xferMemBuf_, xferMD_, xferDevId_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to register xfer memory";
        return false;
    }

    if (is_remote) {
        nixlBlobDesc info;
        info.addr = (uintptr_t)xferMemBuf_;
        info.len = BUF_SIZE;
        info.devId = xferDevId_;
        ret = backend_engine_->getPublicData(xferMD_, info.metaInfo);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get meta info";
            return false;
        }
        if (info.metaInfo.size() == 0) {
            NIXL_ERROR << "Failed to get meta info";
            return false;
        }

        ret = backend_engine_->loadRemoteMD(info, XferMemType, xferAgent_, xferLoadedMem_);
    } else {
        ret = backend_engine_->loadLocalMD(xferMD_, xferLoadedMem_);
    }
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to load MD from " << xferAgent_;
        return false;
    }

    reqSrcDescs_ = std::make_unique<nixl_meta_dlist_t>(LocalMemType);
    reqDstDescs_ = std::make_unique<nixl_meta_dlist_t>(XferMemType);
    populateDescList<LocalMemType>(*reqSrcDescs_, localMemBuf_, localMD_, localDevId_);
    populateDescList<XferMemType>(*reqDstDescs_, xferMemBuf_, xferLoadedMem_, xferDevId_);

    MemoryHandler<LocalMemType>::set(localMemBuf_, LOCAL_BUF_BYTE, BUF_SIZE);
    MemoryHandler<XferMemType>::set(xferMemBuf_, XFER_BUF_BYTE, BUF_SIZE);

    return true;
}

bool
SetupBackendTestFixture::TestXfer(nixl_xfer_op_t op) {
    nixlBackendReqH *handle_;
    nixl_status_t ret;

    ret = backend_engine_->prepXfer(
            op, *reqSrcDescs_, *reqDstDescs_, xferAgent_, handle_, &optionalXferArgs_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to prepare transfer";
        return false;
    }

    ret = backend_engine_->postXfer(
            op, *reqSrcDescs_, *reqDstDescs_, xferAgent_, handle_, &optionalXferArgs_);
    if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
        NIXL_ERROR << "Failed to post transfer";
        return false;
    }

    auto end_time = absl::Now() + absl::Seconds(3);

    NIXL_INFO << "\t\tWaiting for transfer to complete...";

    while (ret == NIXL_IN_PROG && absl::Now() < end_time) {
        ret = backend_engine_->checkXfer(handle_);
        if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
            NIXL_ERROR << "Transfer check failed";
            return false;
        }

        if (xferBackendEngine_->supportsProgTh()) {
            xferBackendEngine_->progress();
        }
    }

    NIXL_INFO << "\nTransfer complete";

    ret = backend_engine_->releaseReqH(handle_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to release transfer handle";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::VerifyNotifs(std::string &msg) {
    notif_list_t target_notifs;
    int num_notifs = 0;
    nixl_status_t ret;

    NIXL_INFO << "\t\tChecking notification flow: ";

    auto end_time = absl::Now() + absl::Seconds(3);

    while (num_notifs == 0 && absl::Now() < end_time) {
        ret = xferBackendEngine_->getNotifs(target_notifs);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get notifications";
            return false;
        }
        num_notifs = target_notifs.size();
        if (backend_engine_->supportsProgTh()) {
            backend_engine_->progress();
        }
    }

    NIXL_INFO << "\nNotification transfer complete";

    if (num_notifs != 1) {
        NIXL_ERROR << "Expected 1 notification, got " << num_notifs;
        return false;
    }

    if (target_notifs.front().first != localAgent_) {
        NIXL_ERROR << "Expected notification from " << localAgent_ << ", got "
                  << target_notifs.front().first;
        return false;
    }
    if (target_notifs.front().second != msg) {
        NIXL_ERROR << "Expected notification message " << msg << ", got "
                  << target_notifs.front().second;
        return false;
    }

    NIXL_INFO << "OK\n"
              << "message: " << target_notifs.front().second << " from "
              << target_notifs.front().first;

    return true;
}

bool
SetupBackendTestFixture::VerifyXfer() {
    if (backend_engine_->supportsNotif()) {
        if (!VerifyNotifs(optionalXferArgs_.notifMsg)) {
            NIXL_ERROR << "Failed in notifications verification";
            return false;
        }

        optionalXferArgs_.notifMsg = "";
        optionalXferArgs_.hasNotif = false;
    }

    return true;
}

bool
SetupBackendTestFixture::TeardownXfer() {
    nixl_status_t ret;

    ret = backend_engine_->unloadMD(xferLoadedMem_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to unload MD";
        return false;
    }

    ret = backend_engine_->disconnect(xferAgent_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to disconnect";
        return false;
    }

    if (teardownCallback_) {
        if (!teardownCallback_()) {
            NIXL_ERROR << "Teardown callback failed";
            return false;
        }
    }

    return true;
}

template<nixl_mem_t LocalMemType, nixl_mem_t XferMemType>
bool
SetupBackendTestFixture::DeregDeallocCallback() {
    nixl_status_t ret;

    ret = backendDeregDealloc<XferMemType>(xferBackendEngine_, xferMemBuf_, xferMD_, xferDevId_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deallocate xfer memory";
        return false;
    }

    ret = backendDeregDealloc<LocalMemType>(backend_engine_.get(), localMemBuf_, localMD_, localDevId_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deallocate local memory";
        return false;
    }

    return true;
}

template<nixl_mem_t LocalMemType, nixl_mem_t XferMemType>
bool
SetupBackendTestFixture::SetupLocalXfer() {
    CHECK(backend_engine_->supportsLocal()) << "Backend engine does not support local transfers";

    xferBackendEngine_ = backend_engine_.get();
    xferAgent_ = localAgent_;
    localDevId_ = 0;
    xferDevId_ = 0;

    if (xferBackendEngine_->supportsNotif()) {
        SetupNotifs("Test");
    }

    if (!PrepXferMem<LocalMemType, XferMemType>(false /* local xfer */)) {
        NIXL_ERROR << "Failed to prepare transfer";
        return false;
    }

    teardownCallback_ = [this]() {
        return DeregDeallocCallback<LocalMemType, XferMemType>();
    };

    resetLocalBufCallback_ = [this]() {
        MemoryHandler<LocalMemType>::set(localMemBuf_, 0x00, BUF_SIZE);
    };

    return true;
}

bool
SetupBackendTestFixture::TestLocalXfer(nixl_xfer_op_t op) {
    if (!TestXfer(op)) {
        NIXL_ERROR << "Failed to test transfer";
        return false;
    }

    if (!VerifyXfer()) {
        NIXL_ERROR << "Failed in transfer verification";
        return false;
    }

    return true;
}

template<nixl_mem_t LocalMemType, nixl_mem_t XferMemType>
bool
SetupBackendTestFixture::SetupRemoteXfer() {
    CHECK(backend_engine_->supportsRemote()) << "Backend engine does not support remote transfers";

    xferBackendEngine_ = remote_backend_engine_.get();
    xferAgent_ = remoteAgent_;
    localDevId_ = 0;
    xferDevId_ = 1;

    if (!VerifyConnInfo()) {
        NIXL_ERROR << "Failed to verify connection info";
        return false;
    }

    if (backend_engine_->supportsNotif()) {
        SetupNotifs("Test");
    }

    if (!PrepXferMem<LocalMemType, XferMemType>(true /* remote xfer */)) {
        NIXL_ERROR << "Failed to prepare transfer";
        return false;
    }

    teardownCallback_ = [this]() {
        return DeregDeallocCallback<LocalMemType, XferMemType>();
    };

    resetLocalBufCallback_ = [this]() {
        MemoryHandler<LocalMemType>::set(localMemBuf_, 0x00, BUF_SIZE);
    };

    return true;
}

bool
SetupBackendTestFixture::TestRemoteXfer(nixl_xfer_op_t op) {
    if (!TestXfer(op)) {
        NIXL_ERROR << "Failed to test transfer";
        return false;
    }

    if (!VerifyXfer()) {
        NIXL_ERROR << "Failed in transfer verification";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::TestGenNotif(std::string msg) {
    nixl_status_t ret;

    ret = backend_engine_->genNotif(remoteAgent_, msg);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to generate notification";
        return false;
    }

    if (!VerifyNotifs(msg)) {
        NIXL_ERROR << "Failed in notification verification";
        return false;
    }

    return true;
}

// Explicit template instantiations
template nixl_status_t plugins_common::SetupBackendTestFixture::backendAllocReg<DRAM_SEG>(nixlBackendEngine *engine, size_t len, void *&mem_buf, nixlBackendMD *&md, int dev_id);
template nixl_status_t plugins_common::SetupBackendTestFixture::backendAllocReg<OBJ_SEG>(nixlBackendEngine *engine, size_t len, void *&mem_buf, nixlBackendMD *&md, int dev_id);
template bool plugins_common::SetupBackendTestFixture::DeregDeallocCallback<DRAM_SEG, OBJ_SEG>();
template bool plugins_common::SetupBackendTestFixture::SetupLocalXfer<DRAM_SEG, OBJ_SEG>();
template bool plugins_common::SetupBackendTestFixture::PrepXferMem<DRAM_SEG, OBJ_SEG>(bool is_remote);

} // namespace plugins_common
