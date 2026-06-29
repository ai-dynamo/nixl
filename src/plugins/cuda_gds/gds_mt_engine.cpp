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
#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>

#include "common/backend.h"
#include "common/nixl_log.h"
#include "gds_mt_engine.h"

namespace {
const size_t default_thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);

[[nodiscard]] size_t
getThreadCount(const nixlBackendInitParams *init_params) {
    nixl_b_params_t *params = init_params->customParams;
    const size_t count =
        nixl::getBackendParamDefaulted(params, "thread_count", default_thread_count);
    return (count > 0) ? count : default_thread_count;
}

void
runCuFileOp(const GdsXferReq *req, std::atomic<nixl_status_t> *overall_status) {
    ssize_t nbytes = 0;
    if (req->op == CUFILE_READ) {
        nbytes = cuFileRead(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            NIXL_ERROR << "GDS_MT: cuFileRead failed: " << strerror(errno);
            overall_status->store(NIXL_ERR_BACKEND);
            return;
        }
    } else if (req->op == CUFILE_WRITE) {
        nbytes = cuFileWrite(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            NIXL_ERROR << "GDS_MT: cuFileWrite failed: " << strerror(errno);
            overall_status->store(NIXL_ERR_BACKEND);
            return;
        }
    } else {
        overall_status->store(NIXL_ERR_INVALID_PARAM);
        return;
    }

    if ((size_t)nbytes != req->size) {
        NIXL_ERROR << "GDS_MT: error: short " << ((req->op == CUFILE_READ) ? "read: " : "write: ")
                   << nbytes << " out of " << req->size << " bytes - address=" << req->addr;
        overall_status->store(NIXL_ERR_BACKEND);
        return;
    }
}
} // namespace

nixlGdsMtReqH::~nixlGdsMtReqH() {
    if (running_transfer.valid()) {
        running_transfer.wait();
    }
}

nixlGdsMtEngine::nixlGdsMtEngine(const nixlBackendInitParams *init_params)
    : nixlGdsEngine(init_params),
      thread_count_(getThreadCount(init_params)) {
    // Base ctor opened the cuFile driver; bail if that failed.
    if (this->initErr) {
        return;
    }

    try {
        executor_ = std::make_unique<tf::Executor>(thread_count_);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "GDS_MT: failed to create executor: " << e.what();
        this->initErr = true;
        return;
    }
    NIXL_DEBUG << "GDS_MT: thread count=" << thread_count_;
}

nixl_status_t
nixlGdsMtEngine::finalizePrep(std::vector<GdsXferReq> &&reqs, nixlBackendReqH *&handle) const {
    if (reqs.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto gds_handle = std::make_unique<nixlGdsMtReqH>();
    gds_handle->request_list = std::move(reqs);

    for (GdsXferReq &req : gds_handle->request_list) {
        GdsXferReq *captured_req = &req;
        gds_handle->taskflow.emplace(
            [captured_req, overall_status = &gds_handle->overall_status]() {
                runCuFileOp(captured_req, overall_status);
            });
    }

    handle = gds_handle.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsMtEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    auto *gds_handle = static_cast<nixlGdsMtReqH *>(handle);

    gds_handle->overall_status.store(NIXL_SUCCESS);
    gds_handle->running_transfer = executor_->run(gds_handle->taskflow);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlGdsMtEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *gds_handle = static_cast<nixlGdsMtReqH *>(handle);
    if (gds_handle->running_transfer.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready) {
        return NIXL_IN_PROG;
    }
    gds_handle->running_transfer.get();
    return gds_handle->overall_status.load();
}

nixl_status_t
nixlGdsMtEngine::releaseReqH(nixlBackendReqH *handle) const {
    delete static_cast<nixlGdsMtReqH *>(handle);
    return NIXL_SUCCESS;
}
