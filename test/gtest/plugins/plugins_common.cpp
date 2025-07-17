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

nixl_status_t
SetupBackendTestFixture::backendAllocReg(nixlBackendEngine *engine,
                                         nixl_mem_t mem_type,
                                         size_t len,
                                         std::unique_ptr<MemoryHandler> &mem_handler,
                                         int buf_index,
                                         int dev_id) {
    mem_handler = std::make_unique<MemoryHandler>(mem_type, dev_id + buf_index);

    try {
        mem_handler->allocate(len);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to allocate memory: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    nixlBlobDesc desc;
    mem_handler->populateBlobDesc(&desc, buf_index);

    NIXL_INFO << "Registering memory type " << mem_handler->getMemType() << " with length " << len
              << " and device ID " << mem_handler->getDevId();

    nixlBackendMD *md;
    nixl_status_t ret;
    ret = engine->registerMem(desc, mem_handler->getMemType(), md);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to register memory: " << ret;
        return ret;
    }
    mem_handler->setMD(md);

    return NIXL_SUCCESS;
}

nixl_status_t
SetupBackendTestFixture::backendDeregDealloc(nixlBackendEngine *engine,
                                             std::unique_ptr<MemoryHandler> &mem_handler) {
    nixl_status_t ret;

    NIXL_INFO << "Deregistering memory type " << mem_handler->getMemType() << " with device ID "
              << mem_handler->getDevId();

    ret = engine->deregisterMem(mem_handler->getMD());
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deregister memory: " << ret;
        return ret;
    }

    mem_handler->deallocate();

    return NIXL_SUCCESS;
}

void
SetupBackendTestFixture::resetLocalBuf() {
    for (int i = 0; i < num_bufs_; i++)
        localMemHandler_[i]->reset();
}

bool
SetupBackendTestFixture::checkLocalBuf() {
    for (int i = 0; i < num_bufs_; i++) {
        if (!localMemHandler_[i]->check(LOCAL_BUF_BYTE + i)) return false;
    }
    return true;
}

void
SetupBackendTestFixture::populateDescList(nixl_meta_dlist_t &descs,
                                          std::unique_ptr<MemoryHandler> mem_handler[]) {
    int num_entries = split_buf_ ? NUM_ENTRIES : 1;
    int entry_size = split_buf_ ? ENTRY_SIZE : BUF_SIZE;

    for (int buf_i = 0; buf_i < num_bufs_; buf_i++) {
        for (int entry_i = 0; entry_i < num_entries; entry_i++) {
            nixlMetaDesc desc;
            mem_handler[buf_i]->populateMetaDesc(&desc, entry_i, entry_size);
            descs.addDesc(desc);
        }
    }
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
SetupBackendTestFixture::verifyConnInfo() {
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
SetupBackendTestFixture::setupNotifs(std::string msg) {
    optionalXferArgs_.notifMsg = msg;
    optionalXferArgs_.hasNotif = true;
}

bool
SetupBackendTestFixture::prepXferMem(nixl_mem_t local_mem_type,
                                     nixl_mem_t xfer_mem_type,
                                     bool is_remote) {
    nixl_status_t ret;

    for (int i = 0; i < num_bufs_; i++) {
        ret = backendAllocReg(
            backend_engine_.get(), local_mem_type, BUF_SIZE, localMemHandler_[i], i, localDevId_);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to register local memory";
            return false;
        }

        ret = backendAllocReg(
            xferBackendEngine_, xfer_mem_type, BUF_SIZE, xferMemHandler_[i], i, xferDevId_);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to register xfer memory";
            return false;
        }

        localMemHandler_[i]->set(LOCAL_BUF_BYTE + i);
        xferMemHandler_[i]->set(XFER_BUF_BYTE + i);
    }

    if (is_remote) {
        nixlBlobDesc info;
        xferMemHandler_[0]->populateBlobDesc(&info);
        ret = backend_engine_->getPublicData(xferMemHandler_[0]->getMD(), info.metaInfo);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get meta info";
            return false;
        }
        if (info.metaInfo.size() == 0) {
            NIXL_ERROR << "Failed to get meta info";
            return false;
        }

        ret = backend_engine_->loadRemoteMD(info, xfer_mem_type, xferAgent_, xferLoadedMD_);
    } else {
        ret = backend_engine_->loadLocalMD(xferMemHandler_[0]->getMD(), xferLoadedMD_);
    }
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to load MD from " << xferAgent_;
        return false;
    }

    reqSrcDescs_ = std::make_unique<nixl_meta_dlist_t>(local_mem_type);
    reqDstDescs_ = std::make_unique<nixl_meta_dlist_t>(xfer_mem_type);
    populateDescList(*reqSrcDescs_, localMemHandler_);
    populateDescList(*reqDstDescs_, xferMemHandler_);

    return true;
}

bool
SetupBackendTestFixture::testXfer(nixl_xfer_op_t op) {
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
SetupBackendTestFixture::verifyNotifs(std::string &msg) {
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
SetupBackendTestFixture::verifyXfer() {
    if (backend_engine_->supportsNotif()) {
        if (!verifyNotifs(optionalXferArgs_.notifMsg)) {
            NIXL_ERROR << "Failed in notifications verification";
            return false;
        }

        optionalXferArgs_.notifMsg = "";
        optionalXferArgs_.hasNotif = false;
    }

    return true;
}

bool
SetupBackendTestFixture::teardownXfer() {
    nixl_status_t ret;

    ret = backend_engine_->unloadMD(xferLoadedMD_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to unload MD";
        return false;
    }

    ret = backend_engine_->disconnect(xferAgent_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to disconnect";
        return false;
    }

    for (int i = 0; i < num_bufs_; i++) {
        ret = backendDeregDealloc(xferBackendEngine_, xferMemHandler_[i]);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to deallocate xfer memory";
            return false;
        }

        ret = backendDeregDealloc(backend_engine_.get(), localMemHandler_[i]);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to deallocate local memory";
            return false;
        }
    }

    return true;
}

bool
SetupBackendTestFixture::setupLocalXfer(nixl_mem_t local_mem_type,
                                        nixl_mem_t xfer_mem_type,
                                        bool split_buf,
                                        int num_bufs) {
    CHECK(backend_engine_->supportsLocal()) << "Backend engine does not support local transfers";
    CHECK(num_bufs <= (int)MAX_NUM_BUFS) << "Number of buffers exceeds maximum number of buffers";

    xferBackendEngine_ = backend_engine_.get();
    xferAgent_ = localAgent_;
    split_buf_ = split_buf;
    num_bufs_ = num_bufs;
    localDevId_ = 0;
    xferDevId_ = 0;

    if (xferBackendEngine_->supportsNotif()) {
        setupNotifs("Test");
    }

    if (!prepXferMem(local_mem_type, xfer_mem_type, false /* local xfer */)) {
        NIXL_ERROR << "Failed to prepare transfer";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::testLocalXfer(nixl_xfer_op_t op) {
    if (!testXfer(op)) {
        NIXL_ERROR << "Failed to test transfer";
        return false;
    }

    if (!verifyXfer()) {
        NIXL_ERROR << "Failed in transfer verification";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::setupRemoteXfer(nixl_mem_t local_mem_type,
                                         nixl_mem_t xfer_mem_type,
                                         bool split_buf,
                                         int num_bufs) {
    CHECK(backend_engine_->supportsRemote()) << "Backend engine does not support remote transfers";
    CHECK(num_bufs <= (int)MAX_NUM_BUFS) << "Number of buffers exceeds maximum number of buffers";

    xferBackendEngine_ = remote_backend_engine_.get();
    xferAgent_ = remoteAgent_;
    split_buf_ = split_buf;
    num_bufs_ = num_bufs;
    localDevId_ = 0;
    xferDevId_ = 1;

    if (!verifyConnInfo()) {
        NIXL_ERROR << "Failed to verify connection info";
        return false;
    }

    if (backend_engine_->supportsNotif()) {
        setupNotifs("Test");
    }

    if (!prepXferMem(local_mem_type, xfer_mem_type, true /* remote xfer */)) {
        NIXL_ERROR << "Failed to prepare transfer";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::testRemoteXfer(nixl_xfer_op_t op) {
    if (!testXfer(op)) {
        NIXL_ERROR << "Failed to test transfer";
        return false;
    }

    if (!verifyXfer()) {
        NIXL_ERROR << "Failed in transfer verification";
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::testGenNotif(std::string msg) {
    nixl_status_t ret;

    ret = backend_engine_->genNotif(remoteAgent_, msg);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to generate notification";
        return false;
    }

    if (!verifyNotifs(msg)) {
        NIXL_ERROR << "Failed in notification verification";
        return false;
    }

    return true;
}

} // namespace plugins_common
