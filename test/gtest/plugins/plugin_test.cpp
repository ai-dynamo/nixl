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

#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "nixl.h"
#include "plugin_manager.h"
#include "backend_engine.h"

#include "plugin_test.h"

namespace plugin_test {

nixl_status_t
SetupBackendTestFixture::backendAllocReg(nixlBackendEngine *engine,
                                         nixl_mem_t mem_type,
                                         size_t len,
                                         void *&mem_buf,
                                         nixlBackendMD *&md,
                                         int dev_id = 0) {
    nixlBlobDesc desc;
    nixl_status_t ret;

    try {
        switch (mem_type) {
        case DRAM_SEG:
            mem_buf = MemoryHandler<DRAM_SEG>::allocate(len);
            break;
        case VRAM_SEG:
            mem_buf = MemoryHandler<VRAM_SEG>::allocate(len, dev_id);
            break;
        default:
            std::cerr << "Unsupported memory type: " << mem_type << std::endl;
            assert(0);
        }
    }
    catch (const std::exception &e) {
        std::cerr << "Failed to allocate memory: " << e.what() << std::endl;
        return NIXL_ERR_BACKEND;
    }

    desc.addr = (uintptr_t)mem_buf;
    desc.len = len;
    desc.devId = dev_id;

    std::cout << "Registering memory type " << mem_type << " at address " << mem_buf
              << " with length " << len << " and device ID " << dev_id << std::endl;

    ret = engine->registerMem(desc, mem_type, md);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register memory: " << ret << std::endl;
        return ret;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
SetupBackendTestFixture::backendDeregDealloc(nixlBackendEngine *engine,
                                             nixl_mem_t mem_type,
                                             void *mem_buf,
                                             nixlBackendMD *&md,
                                             int dev_id = 0) {
    nixl_status_t ret;

    std::cout << "Deregistering memory type " << mem_type << " at address " << mem_buf
              << " with device ID " << dev_id << std::endl;

    ret = engine->deregisterMem(md);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to deregister memory: " << ret << std::endl;
        return ret;
    }

    switch (mem_type) {
    case DRAM_SEG:
        MemoryHandler<DRAM_SEG>::deallocate(mem_buf);
        break;
    case VRAM_SEG:
        MemoryHandler<VRAM_SEG>::deallocate(mem_buf, dev_id);
        break;
    default:
        std::cerr << "Unsupported memory type!" << std::endl;
        assert(0);
    }

    return NIXL_SUCCESS;
}

void
SetupBackendTestFixture::setBuf(nixl_mem_t mem_type, void *addr, char byte, int dev_id = 0) {
    switch (mem_type) {
    case DRAM_SEG:
        MemoryHandler<DRAM_SEG>::set(addr, byte, BUF_SIZE);
        break;
    case VRAM_SEG:
        MemoryHandler<VRAM_SEG>::set(addr, byte, BUF_SIZE, dev_id);
        break;
    default:
        std::cerr << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void *
SetupBackendTestFixture::getMemBufPtr(nixl_mem_t mem_type, void *addr, size_t len) {
    switch (mem_type) {
    case DRAM_SEG:
        return MemoryHandler<DRAM_SEG>::getPtr(addr, len);
    case VRAM_SEG:
        return MemoryHandler<VRAM_SEG>::getPtr(addr, len);
    default:
        std::cerr << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void
SetupBackendTestFixture::releaseMemBufPtr(nixl_mem_t mem_type, void *addr) {
    switch (mem_type) {
    case DRAM_SEG:
        MemoryHandler<DRAM_SEG>::releasePtr(addr);
        break;
    case VRAM_SEG:
        MemoryHandler<VRAM_SEG>::releasePtr(addr);
        break;
    default:
        std::cerr << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void
SetupBackendTestFixture::populateDescList(nixl_meta_dlist_t &descs,
                                          void *buf,
                                          nixlBackendMD *&md,
                                          int dev_id = 0) {
    for (size_t i = 0; i < NUM_BUF_ENTRIES; i++) {
        nixlMetaDesc req;
        req.addr = (uintptr_t)(((char *)buf) + i * BUF_ENTRY_SIZE); // random offset
        req.len = BUF_ENTRY_SIZE;
        req.devId = dev_id;
        req.metadataP = md;
        descs.addDesc(req);
    }
}

void
SetupBackendTestFixture::SetUp() {
    nixl_status_t ret;

    if (backend_engine_->getInitErr()) {
        std::cerr << "Failed to initialize backend engine" << std::endl;
        assert(0);
    }

    localAgent_ = "Agent1";
    remoteAgent_ = "Agent2";

    ret = backendAllocReg(backend_engine_.get(), DRAM_SEG, BUF_SIZE, localMemBuf_, localMem_);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register memory" << std::endl;
        assert(0);
    }

    isSetup_ = true;
}

void
SetupBackendTestFixture::TearDown() {
    nixl_status_t ret;

    ret = backendDeregDealloc(backend_engine_.get(), DRAM_SEG, localMemBuf_, localMem_);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to deregister memory" << std::endl;
        assert(0);
    }
}

bool
SetupBackendTestFixture::VerifyConnInfo(bool is_remote) {
    std::string conn_info;
    nixl_status_t ret;

    ret = backend_engine_->getConnInfo(conn_info);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to get connection info" << std::endl;
        return false;
    }

    if (is_remote) {
        ret = xferBackendEngine_->getConnInfo(conn_info);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to get remote connection info" << std::endl;
            return false;
        }
    }

    ret = backend_engine_->loadRemoteConnInfo(xferAgent_, conn_info);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to load remote connection info" << std::endl;
        return false;
    }

    return true;
}

void
SetupBackendTestFixture::SetupNotifs(std::string msg) {
    optionalXferArgs_.notifMsg = msg;
    optionalXferArgs_.hasNotif = true;
}

bool
SetupBackendTestFixture::PrepXferMem(nixl_mem_t mem_type, bool is_remote) {
    nixl_status_t ret;

    ret = backendAllocReg(
            xferBackendEngine_, mem_type, BUF_SIZE, xferMemBuf_, xferMem_, xferDevId_);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register memory" << std::endl;
        return false;
    }

    if (is_remote) {
        nixlBlobDesc info;
        info.addr = (uintptr_t)xferMemBuf_;
        info.len = BUF_SIZE;
        info.devId = xferDevId_;
        ret = backend_engine_->getPublicData(xferMem_, info.metaInfo);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to get meta info" << std::endl;
            return false;
        }
        if (info.metaInfo.size() == 0) {
            std::cerr << "Failed to get meta info" << std::endl;
            return false;
        }

        ret = backend_engine_->loadRemoteMD(info, mem_type, xferAgent_, xferLoadedMem_);
    } else {
        ret = backend_engine_->loadLocalMD(xferMem_, xferLoadedMem_);
    }
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to load MD from " << xferAgent_ << std::endl;
        return false;
    }

    reqSrcDescs_ = std::make_unique<nixl_meta_dlist_t>(DRAM_SEG);
    reqDstDescs_ = std::make_unique<nixl_meta_dlist_t>(mem_type);
    populateDescList(*reqSrcDescs_, localMemBuf_, localMem_);
    populateDescList(*reqDstDescs_, xferMemBuf_, xferLoadedMem_);

    setBuf(mem_type, localMemBuf_, 0xbb);
    setBuf(mem_type, xferMemBuf_, 0x44);

    return true;
}

bool
SetupBackendTestFixture::TestXfer(nixl_xfer_op_t op) {
    nixlBackendReqH *handle_;
    nixl_status_t ret;

    ret = backend_engine_->prepXfer(
            op, *reqSrcDescs_, *reqDstDescs_, xferAgent_, handle_, &optionalXferArgs_);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to prepare transfer" << std::endl;
        return false;
    }

    ret = backend_engine_->postXfer(
            op, *reqSrcDescs_, *reqDstDescs_, xferAgent_, handle_, &optionalXferArgs_);
    if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
        std::cerr << "Failed to post transfer" << std::endl;
        return false;
    }

    auto end_time = absl::Now() + absl::Seconds(3);

    std::cout << "\t\tWaiting for transfer to complete..." << std::endl;

    while (ret == NIXL_IN_PROG && absl::Now() < end_time) {
        ret = backend_engine_->checkXfer(handle_);
        if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
            std::cerr << "Transfer check failed" << std::endl;
            return false;
        }

        if (xferBackendEngine_->supportsProgTh()) {
            xferBackendEngine_->progress();
        }
    }

    std::cout << "\nTransfer complete" << std::endl;

    ret = backend_engine_->releaseReqH(handle_);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to release transfer handle" << std::endl;
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::VerifyNotifs(std::string &msg) {
    notif_list_t target_notifs;
    int num_notifs = 0;
    nixl_status_t ret;

    std::cout << "\t\tChecking notification flow: " << std::endl;

    auto end_time = absl::Now() + absl::Seconds(3);

    while (num_notifs == 0 && absl::Now() < end_time) {
        ret = xferBackendEngine_->getNotifs(target_notifs);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to get notifications" << std::endl;
            return false;
        }
        num_notifs = target_notifs.size();
        if (backend_engine_->supportsProgTh()) {
            backend_engine_->progress();
        }
    }

    std::cout << "\nNotification transfer complete" << std::endl;

    if (num_notifs != 1) {
        std::cerr << "Expected 1 notification, got " << num_notifs << std::endl;
        return false;
    }

    if (target_notifs.front().first != localAgent_) {
        std::cerr << "Expected notification from " << localAgent_ << ", got "
                  << target_notifs.front().first << std::endl;
        return false;
    }
    if (target_notifs.front().second != msg) {
        std::cerr << "Expected notification message " << msg << ", got "
                  << target_notifs.front().second << std::endl;
        return false;
    }

    std::cout << "OK\n"
              << "message: " << target_notifs.front().second << " from "
              << target_notifs.front().first << std::endl;

    return true;
}

bool
SetupBackendTestFixture::VerifyXferData(void *src_mem_buf, void *dst_mem_buf) {
    void *src_val_ptr;
    void *dst_val_ptr;

    absl::SleepFor(absl::Seconds(1));

    std::cout << "\t\tData verification: " << std::flush;

    src_val_ptr = getMemBufPtr(reqSrcDescs_->getType(), src_mem_buf, BUF_SIZE);
    dst_val_ptr = getMemBufPtr(reqDstDescs_->getType(), dst_mem_buf, BUF_SIZE);

    // Perform correctness check.
    for (size_t i = 0; i < BUF_SIZE; i++) {
        if (((uint8_t *)src_val_ptr)[i] != ((uint8_t *)dst_val_ptr)[i]) {
            std::cerr << "Verification failed at index " << i << std::endl;
            return false;
        }
    }

    std::cout << "OK" << std::endl;

    releaseMemBufPtr(reqSrcDescs_->getType(), src_val_ptr);
    releaseMemBufPtr(reqDstDescs_->getType(), dst_val_ptr);

    return true;
}

bool
SetupBackendTestFixture::VerifyXfer() {
    if (backend_engine_->supportsNotif()) {
        if (!VerifyNotifs(optionalXferArgs_.notifMsg)) {
            std::cerr << "Failed in notifications verification" << std::endl;
            return false;
        }

        optionalXferArgs_.notifMsg = "";
        optionalXferArgs_.hasNotif = false;
    }

    if (!VerifyXferData(localMemBuf_, xferMemBuf_)) {
        std::cerr << "Failed in transfer verification" << std::endl;
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::TeardownXfer(nixl_mem_t mem_type) {
    nixl_status_t ret;

    ret = backend_engine_->unloadMD(xferLoadedMem_);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to unload MD" << std::endl;
        return false;
    }

    ret = backendDeregDealloc(xferBackendEngine_, mem_type, xferMemBuf_, xferMem_);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to deallocate memory" << std::endl;
        return false;
    }

    ret = backend_engine_->disconnect(xferAgent_);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to disconnect" << std::endl;
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::SetupLocalXfer(nixl_mem_t mem_type) {
    assert(backend_engine_->supportsLocal());

    xferBackendEngine_ = backend_engine_.get();
    xferAgent_ = localAgent_;
    xferDevId_ = 0;

    if (!VerifyConnInfo(false /* local xfer */)) {
        std::cerr << "Failed to verify connection info" << std::endl;
        return false;
    }

    if (xferBackendEngine_->supportsNotif()) {
        SetupNotifs("Test");
    }

    if (!PrepXferMem(mem_type, false /* local xfer */)) {
        std::cerr << "Failed to prepare transfer" << std::endl;
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::TestLocalXfer(nixl_mem_t mem_type, nixl_xfer_op_t op) {
    if (!SetupLocalXfer(mem_type)) {
        std::cerr << "Failed to setup local xfer" << std::endl;
        return false;
    }

    if (!TestXfer(op)) {
        std::cerr << "Failed to test transfer" << std::endl;
        return false;
    }

    if (!VerifyXfer()) {
        std::cerr << "Failed in transfer verification" << std::endl;
        return false;
    }

    if (!TeardownXfer(mem_type)) {
        std::cerr << "Failed in xfer memory cleanup" << std::endl;
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::SetupRemoteXfer(nixl_mem_t mem_type) {
    assert(backend_engine_->supportsRemote());

    xferBackendEngine_ = remote_backend_engine_.get();
    xferAgent_ = remoteAgent_;
    xferDevId_ = 1;

    if (!VerifyConnInfo(true /* remote xfer */)) {
        std::cerr << "Failed to verify connection info" << std::endl;
        return false;
    }

    if (backend_engine_->supportsNotif()) {
        SetupNotifs("Test");
    }

    if (!PrepXferMem(mem_type, true /* remote xfer */)) {
        std::cerr << "Failed to prepare transfer" << std::endl;
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::TestRemoteXfer(nixl_mem_t mem_type, nixl_xfer_op_t op) {
    if (!SetupRemoteXfer(mem_type)) {
        std::cerr << "Failed to setup remote xfer" << std::endl;
        return false;
    }

    if (!TestXfer(op)) {
        std::cerr << "Failed to test transfer" << std::endl;
        return false;
    }

    if (!VerifyXfer()) {
        std::cerr << "Failed in transfer verification" << std::endl;
        return false;
    }

    if (!TeardownXfer(mem_type)) {
        std::cerr << "Failed in xfer memory cleanup" << std::endl;
        return false;
    }

    return true;
}

bool
SetupBackendTestFixture::TestGenNotif(std::string msg) {
    nixl_status_t ret;

    ret = backend_engine_->genNotif(remoteAgent_, msg);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to generate notification" << std::endl;
        return false;
    }

    if (!VerifyNotifs(msg)) {
        std::cerr << "Failed in notification verification" << std::endl;
        return false;
    }

    return true;
}

} // namespace plugin_test
