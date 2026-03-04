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
#ifndef __TRANSFER_HANDLER_H
#define __TRANSFER_HANDLER_H

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include "backend_engine.h"
#include "common/nixl_log.h"
#include "gtest/gtest.h"
#include "memory_handler.h"

namespace gtest::plugins {

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType> class transferHandler {
public:
    transferHandler(std::shared_ptr<nixlBackendEngine> src_engine,
                    std::shared_ptr<nixlBackendEngine> dst_engine,
                    std::string src_agent_name,
                    std::string dst_agent_name,
                    bool split_buf,
                    int num_bufs,
                    size_t buffer_size = 0)
        : srcBackendEngine_(src_engine),
          dstBackendEngine_(dst_engine),
          srcAgentName_(src_agent_name),
          dstAgentName_(dst_agent_name),
          srcDevId_(0),
          bufferSize_(buffer_size > 0 ? buffer_size : BUF_SIZE),
          numEntries_(buffer_size > 0 ? 1 : NUM_ENTRIES),
          entrySize_(buffer_size > 0 ? buffer_size : ENTRY_SIZE) {

        bool remote_xfer = srcAgentName_ != dstAgentName_;
        if (remote_xfer) {
            CHECK(src_engine->supportsRemote()) << "Local engine does not support remote transfers";
            dstDevId_ = 1;
            verifyConnInfo();
        } else {
            CHECK(src_engine->supportsLocal()) << "Local engine does not support local transfers";
            dstDevId_ = srcDevId_;
        }

        for (int i = 0; i < num_bufs; i++) {
            srcMem_.emplace_back(
                std::make_unique<memoryHandler<srcMemType>>(bufferSize_, srcDevId_ + i));
            dstMem_.emplace_back(
                std::make_unique<memoryHandler<dstMemType>>(bufferSize_, dstDevId_ + i));
        }

        if (dstBackendEngine_->supportsNotif()) setupNotifs("Test");

        registerMems();
        prepMems(split_buf, remote_xfer);
    }

    ~transferHandler() {
        EXPECT_EQ(srcBackendEngine_->unloadMD(xferLoadedMd_), NIXL_SUCCESS);
        EXPECT_EQ(srcBackendEngine_->disconnect(dstAgentName_), NIXL_SUCCESS);
        deregisterMems();
    }

    void
    testTransfer(nixl_xfer_op_t op) {
        performTransfer(op);
        verifyTransfer(op);
    }

    void
    setLocalMem() {
        for (size_t i = 0; i < srcMem_.size(); i++)
            srcMem_[i]->setIncreasing(LOCAL_BUF_BYTE + i);
    }

    void
    resetLocalMem() {
        for (const auto &mem : srcMem_)
            mem->reset();
    }

    void
    checkLocalMem() {
        for (size_t i = 0; i < srcMem_.size(); i++)
            EXPECT_TRUE(srcMem_[i]->checkIncreasing(LOCAL_BUF_BYTE + i));
    }

private:
    static constexpr uint8_t LOCAL_BUF_BYTE = 0x11;
    static constexpr uint8_t XFER_BUF_BYTE = 0x22;
    static constexpr size_t NUM_ENTRIES = 4;
    static constexpr size_t ENTRY_SIZE = 16;
    static constexpr size_t BUF_SIZE = NUM_ENTRIES * ENTRY_SIZE;

    std::vector<std::unique_ptr<memoryHandler<srcMemType>>> srcMem_;
    std::vector<std::unique_ptr<memoryHandler<dstMemType>>> dstMem_;
    std::shared_ptr<nixlBackendEngine> srcBackendEngine_;
    std::shared_ptr<nixlBackendEngine> dstBackendEngine_;
    std::unique_ptr<nixl_meta_dlist_t> srcDescs_;
    std::unique_ptr<nixl_meta_dlist_t> dstDescs_;
    nixl_opt_b_args_t xferOptArgs_;
    nixlBackendMD *xferLoadedMd_;
    std::string srcAgentName_;
    std::string dstAgentName_;
    int srcDevId_;
    int dstDevId_;
    size_t bufferSize_;
    size_t numEntries_;
    size_t entrySize_;

    void
    registerMems() {
        nixlBlobDesc src_desc;
        nixlBlobDesc dst_desc;
        nixlBackendMD *md;

        for (size_t i = 0; i < srcMem_.size(); i++) {
            srcMem_[i]->populateBlobDesc(&src_desc, i);
            ASSERT_EQ(srcBackendEngine_->registerMem(src_desc, srcMemType, md), NIXL_SUCCESS);
            srcMem_[i]->setMD(md);

            dstMem_[i]->populateBlobDesc(&dst_desc, i);
            ASSERT_EQ(dstBackendEngine_->registerMem(dst_desc, dstMemType, md), NIXL_SUCCESS);
            dstMem_[i]->setMD(md);
        }
    }

    void
    deregisterMems() {
        for (size_t i = 0; i < srcMem_.size(); i++) {
            ASSERT_EQ(srcBackendEngine_->deregisterMem(srcMem_[i]->getMD()), NIXL_SUCCESS);
            ASSERT_EQ(dstBackendEngine_->deregisterMem(dstMem_[i]->getMD()), NIXL_SUCCESS);
        }
    }

    void
    prepMems(bool split_buf, bool remote_xfer) {
        if (remote_xfer) {
            nixlBlobDesc info;
            dstMem_[0]->populateBlobDesc(&info);
            ASSERT_EQ(srcBackendEngine_->getPublicData(dstMem_[0]->getMD(), info.metaInfo),
                      NIXL_SUCCESS);
            ASSERT_GT(info.metaInfo.size(), 0);
            ASSERT_EQ(
                srcBackendEngine_->loadRemoteMD(info, dstMemType, dstAgentName_, xferLoadedMd_),
                NIXL_SUCCESS);
        } else {
            ASSERT_EQ(srcBackendEngine_->loadLocalMD(dstMem_[0]->getMD(), xferLoadedMd_),
                      NIXL_SUCCESS);
        }

        srcDescs_ = std::make_unique<nixl_meta_dlist_t>(srcMemType);
        dstDescs_ = std::make_unique<nixl_meta_dlist_t>(dstMemType);

        int num_entries = split_buf ? numEntries_ : 1;
        int entry_size = split_buf ? entrySize_ : bufferSize_;
        for (size_t i = 0; i < srcMem_.size(); i++) {
            for (int entry_i = 0; entry_i < num_entries; entry_i++) {
                nixlMetaDesc desc;
                srcMem_[i]->populateMetaDesc(&desc, entry_i, entry_size);
                srcDescs_->addDesc(desc);
                dstMem_[i]->populateMetaDesc(&desc, entry_i, entry_size);
                dstDescs_->addDesc(desc);
            }
        }
    }

    void
    performTransfer(nixl_xfer_op_t op) {
        nixlBackendReqH *handle;
        nixl_status_t ret;

        ASSERT_EQ(srcBackendEngine_->prepXfer(
                      op, *srcDescs_, *dstDescs_, dstAgentName_, handle, &xferOptArgs_),
                  NIXL_SUCCESS);

        ret = srcBackendEngine_->postXfer(
            op, *srcDescs_, *dstDescs_, dstAgentName_, handle, &xferOptArgs_);
        ASSERT_TRUE(ret == NIXL_SUCCESS || ret == NIXL_IN_PROG);

        NIXL_INFO << "\t\tWaiting for transfer to complete...";

        auto end_time = absl::Now() + absl::Seconds(3);

        while (ret == NIXL_IN_PROG && absl::Now() < end_time) {
            ret = srcBackendEngine_->checkXfer(handle);
            ASSERT_TRUE(ret == NIXL_SUCCESS || ret == NIXL_IN_PROG);
        }

        NIXL_INFO << "\nTransfer complete";

        ASSERT_EQ(srcBackendEngine_->releaseReqH(handle), NIXL_SUCCESS);
    }

    void
    verifyTransfer(nixl_xfer_op_t op) {
        if (srcBackendEngine_->supportsNotif()) {
            verifyNotifs(xferOptArgs_.notifMsg);

            xferOptArgs_.notifMsg = "";
            xferOptArgs_.hasNotif = false;
        }
    }

    void
    verifyNotifs(std::string &msg) {
        notif_list_t target_notifs;
        int num_notifs = 0;

        NIXL_INFO << "\t\tChecking notification flow: ";

        auto end_time = absl::Now() + absl::Seconds(3);
        while (num_notifs == 0 && absl::Now() < end_time) {
            ASSERT_EQ(dstBackendEngine_->getNotifs(target_notifs), NIXL_SUCCESS);
            num_notifs = target_notifs.size();
        }

        NIXL_INFO << "\nNotification transfer complete";

        ASSERT_EQ(num_notifs, 1) << "Expected 1 notification, got " << num_notifs;
        ASSERT_EQ(target_notifs.front().first, srcAgentName_)
            << "Expected notification from " << srcAgentName_ << ", got "
            << target_notifs.front().first;
        ASSERT_EQ(target_notifs.front().second, msg)
            << "Expected notification message " << msg << ", got " << target_notifs.front().second;

        NIXL_INFO << "OK\n"
                  << "message: " << target_notifs.front().second << " from "
                  << target_notifs.front().first;
    }

    void
    setupNotifs(std::string msg) {
        xferOptArgs_.notifMsg = msg;
        xferOptArgs_.hasNotif = true;
    }

    void
    verifyConnInfo() {
        std::string conn_info;

        ASSERT_EQ(srcBackendEngine_->getConnInfo(conn_info), NIXL_SUCCESS);
        ASSERT_EQ(dstBackendEngine_->getConnInfo(conn_info), NIXL_SUCCESS);
        ASSERT_EQ(srcBackendEngine_->loadRemoteConnInfo(dstAgentName_, conn_info), NIXL_SUCCESS);
    }
};

} // namespace gtest::plugins
#endif // __TRANSFER_HANDLER_H
