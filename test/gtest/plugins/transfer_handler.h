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
#ifndef __TRANSFER_HANDLER_H
#define __TRANSFER_HANDLER_H

#include <random>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include "backend_engine.h"
#include "common/nixl_log.h"
#include "gtest/gtest.h"
#include "memory_handler.h"

namespace gtest::plugins {

int
getRandomInt(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(min, max);
    return dist(gen);
}

struct transferMemConfig {
    const size_t numEntries_ = 1;
    const size_t entrySize_ = 64;
    const size_t numBufs_ = 1;
    const uint8_t srcBufByte_ = getRandomInt(0, 255);
    const uint8_t dstBufByte_ = getRandomInt(0, 255);

    size_t
    bufSize() const {
        return numEntries_ * entrySize_;
    }
};

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType> class transferHandler {
public:
    transferHandler(std::shared_ptr<nixlBackendEngine> src_engine,
                    std::shared_ptr<nixlBackendEngine> dst_engine,
                    std::string src_agent_name,
                    std::string dst_agent_name,
                    transferMemConfig mem_cfg = transferMemConfig())
        : srcBackendEngine_(src_engine),
          dstBackendEngine_(dst_engine),
          srcDescs_(std::make_unique<nixl_meta_dlist_t>(srcMemType)),
          dstDescs_(std::make_unique<nixl_meta_dlist_t>(dstMemType)),
          memConfig_(std::move(mem_cfg)),
          srcAgentName_(src_agent_name),
          dstAgentName_(dst_agent_name),
          isRemoteXfer_(srcAgentName_ != dstAgentName_),
          srcDevId_(0),
          dstDevId_(isRemoteXfer_ ? 1 : 0) {
        if (dstBackendEngine_->supportsNotif()) setupNotifs("Test");
    }

    void
    setupMems() {
        for (size_t i = 0; i < memConfig_.numBufs_; i++) {
            srcMem_.emplace_back(
                std::make_unique<memoryHandler<srcMemType>>(memConfig_.bufSize(), srcDevId_ + i));
            dstMem_.emplace_back(
                std::make_unique<memoryHandler<dstMemType>>(memConfig_.bufSize(), dstDevId_ + i));
        }

        registerMems();
        prepareMems();
    }

    ~transferHandler() {
        EXPECT_EQ(srcBackendEngine_->unloadMD(xferLoadedMd_), NIXL_SUCCESS);
        EXPECT_EQ(srcBackendEngine_->disconnect(dstAgentName_), NIXL_SUCCESS);
        deregisterMems();
    }

    void
    testTransfer(nixl_xfer_op_t op) {
        verifyConnInfo();
        ASSERT_EQ(prepareTransfer(op), NIXL_SUCCESS);
        ASSERT_EQ(postTransfer(op), NIXL_SUCCESS);
        ASSERT_EQ(waitForTransfer(), NIXL_SUCCESS);
        ASSERT_EQ(srcBackendEngine_->releaseReqH(xferHandle_), NIXL_SUCCESS);
        verifyTransfer(op);
    }

    nixl_status_t
    prepareTransfer(nixl_xfer_op_t op) {
        return srcBackendEngine_->prepXfer(
            op, *srcDescs_, *dstDescs_, dstAgentName_, xferHandle_, &xferOptArgs_);
    }

    nixl_status_t
    postTransfer(nixl_xfer_op_t op) {
        nixl_status_t ret;
        ret = srcBackendEngine_->postXfer(
            op, *srcDescs_, *dstDescs_, dstAgentName_, xferHandle_, &xferOptArgs_);
        return (ret == NIXL_SUCCESS || ret == NIXL_IN_PROG) ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
    }

    nixl_status_t
    waitForTransfer() {
        nixl_status_t ret = NIXL_IN_PROG;
        auto end_time = absl::Now() + absl::Seconds(3);

        NIXL_INFO << "\t\tWaiting for transfer to complete...";
        while (ret == NIXL_IN_PROG && absl::Now() < end_time) {
            ret = srcBackendEngine_->checkXfer(xferHandle_);
            if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) return ret;

            if (dstBackendEngine_->supportsProgTh()) dstBackendEngine_->progress();
        }
        NIXL_INFO << "\nTransfer complete";

        return NIXL_SUCCESS;
    }

    void
    addSrcDesc(nixlMetaDesc &meta_desc) {
        srcDescs_->addDesc(meta_desc);
    }

    void
    addDstDesc(nixlMetaDesc &meta_desc) {
        dstDescs_->addDesc(meta_desc);
    }

    void
    setSrcMem() {
        for (size_t i = 0; i < srcMem_.size(); i++)
            srcMem_[i]->setIncreasing(memConfig_.srcBufByte_ + i);
    }

    void
    resetSrcMem() {
        for (const auto &mem : srcMem_)
            mem->reset();
    }

    void
    checkSrcMem() {
        for (size_t i = 0; i < srcMem_.size(); i++)
            EXPECT_TRUE(srcMem_[i]->checkIncreasing(memConfig_.srcBufByte_ + i));
    }

private:
    std::vector<std::unique_ptr<memoryHandler<srcMemType>>> srcMem_;
    std::vector<std::unique_ptr<memoryHandler<dstMemType>>> dstMem_;
    const std::shared_ptr<nixlBackendEngine> srcBackendEngine_;
    const std::shared_ptr<nixlBackendEngine> dstBackendEngine_;
    const std::unique_ptr<nixl_meta_dlist_t> srcDescs_;
    const std::unique_ptr<nixl_meta_dlist_t> dstDescs_;
    const transferMemConfig memConfig_;
    const std::string srcAgentName_;
    const std::string dstAgentName_;
    nixl_opt_b_args_t xferOptArgs_;
    nixlBackendMD *xferLoadedMd_;
    nixlBackendReqH *xferHandle_;
    const bool isRemoteXfer_;
    const int srcDevId_;
    const int dstDevId_;

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
    prepareMems() {
        if (isRemoteXfer_) {
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

        for (size_t i = 0; i < srcMem_.size(); i++) {
            for (size_t entry_i = 0; entry_i < memConfig_.numEntries_; entry_i++) {
                nixlMetaDesc desc;
                srcMem_[i]->populateMetaDesc(&desc, entry_i, memConfig_.entrySize_);
                srcDescs_->addDesc(desc);
                dstMem_[i]->populateMetaDesc(&desc, entry_i, memConfig_.entrySize_);
                dstDescs_->addDesc(desc);
            }
        }
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
            if (srcBackendEngine_->supportsProgTh()) srcBackendEngine_->progress();
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
        if (!isRemoteXfer_) return;

        std::string conn_info;
        ASSERT_EQ(srcBackendEngine_->getConnInfo(conn_info), NIXL_SUCCESS);
        ASSERT_EQ(dstBackendEngine_->getConnInfo(conn_info), NIXL_SUCCESS);
        ASSERT_EQ(srcBackendEngine_->loadRemoteConnInfo(dstAgentName_, conn_info), NIXL_SUCCESS);
    }
};

} // namespace gtest::plugins
#endif // __TRANSFER_HANDLER_H
