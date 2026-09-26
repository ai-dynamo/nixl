/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <exception>
#include <string>
#include <thread>
#include <utility>

#include <hip/hip_runtime.h>

#include "ais_mt_engine.h"
#include "common/backend.h"
#include "common/nixl_log.h"

size_t
defaultAisMtThreadCount() noexcept {
    static const size_t thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);
    return thread_count;
}

namespace {
[[nodiscard]] size_t
getThreadCount(const nixlBackendInitParams *init_params) {
    nixl_b_params_t *params = init_params->customParams;
    const size_t default_thread_count = defaultAisMtThreadCount();
    const size_t count =
        nixl::getBackendParamDefaulted(params, "thread_count", default_thread_count);
    return (count > 0) ? count : default_thread_count;
}

// hipFileRead/hipFileWrite report failure two different ways (see hipfile.h):
// -1 means a system error in errno, any other negative value is the negated
// hipFileOpError_t. Reading errno for the second case prints whatever was left
// there -- typically "Success", which is worse than no message at all.
std::string
describeHipFileIoError(ssize_t rc) {
    if (rc == -1) {
        return "system error: " + nixl_strerror(errno);
    }

    const long err = -rc;
    std::string desc = "hipFile error " + std::to_string(err);
    // A driver error carries the real cause in the HIP runtime's error state.
    if (err == hipFileHipDriverError) {
        const hipError_t hip_err = hipPeekAtLastError();
        desc += " (hipFileHipDriverError; hip error " + std::to_string(hip_err) + ": " +
            hipGetErrorString(hip_err) + ")";
    }
    return desc;
}

void
runHipFileOp(const aisXferReq *req, std::atomic<nixl_status_t> *overall_status) {
    // hipFile I/O targets the buffer's owning GPU, and device selection is
    // per-thread, so each worker selects it before issuing the transfer.
    if (req->dev_id >= 0) {
        const hipError_t dev_err = hipSetDevice(req->dev_id);
        if (dev_err != hipSuccess) {
            NIXL_ERROR << "AIS_MT: hipSetDevice failed: " << hipGetErrorString(dev_err);
            overall_status->store(NIXL_ERR_BACKEND);
            return;
        }
    }

    ssize_t nbytes = 0;
    if (req->op == hipFileBatchRead) {
        nbytes = hipFileRead(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            NIXL_ERROR << "AIS_MT: hipFileRead failed: " << describeHipFileIoError(nbytes);
            overall_status->store(NIXL_ERR_BACKEND);
            return;
        }
    } else if (req->op == hipFileBatchWrite) {
        nbytes = hipFileWrite(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            NIXL_ERROR << "AIS_MT: hipFileWrite failed: " << describeHipFileIoError(nbytes);
            overall_status->store(NIXL_ERR_BACKEND);
            return;
        }
    } else {
        overall_status->store(NIXL_ERR_INVALID_PARAM);
        return;
    }

    if ((size_t)nbytes != req->size) {
        NIXL_ERROR << "AIS_MT: error: short "
                   << ((req->op == hipFileBatchRead) ? "read: " : "write: ") << nbytes << " out of "
                   << req->size << " bytes - address=" << req->addr;
        overall_status->store(NIXL_ERR_BACKEND);
        return;
    }
}
} // namespace

nixlAisMtReqH::~nixlAisMtReqH() {
    if (running_transfer.valid()) {
        // TODO: A follow-up should make active release nonblocking. This wait
        // preserves the pre-consolidation AIS_MT request-lifetime behavior.
        running_transfer.wait();
    }
}

nixlAisMtEngine::nixlAisMtEngine(const nixlBackendInitParams *init_params)
    : nixlAisEngine(init_params),
      thread_count_(getThreadCount(init_params)) {
    // Base ctor opened the hipFile driver; bail if that failed.
    if (this->initErr) {
        return;
    }

    try {
        executor_ = std::make_unique<tf::Executor>(thread_count_);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "AIS_MT: failed to create executor: " << e.what();
        this->initErr = true;
        return;
    }
    NIXL_DEBUG << "AIS_MT: thread count=" << thread_count_;
}

nixl_status_t
nixlAisMtEngine::finalizePrep(std::vector<aisXferReq> &&reqs, nixlBackendReqH *&handle) const {
    if (reqs.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto ais_handle = std::make_unique<nixlAisMtReqH>();
    ais_handle->request_list = std::move(reqs);

    // One task per request, matching GDS_MT. The early-exit below has no GDS_MT
    // counterpart; it is kept from the original AIS engine so a bad transfer
    // does not keep the pool busy. It is best-effort rather than a barrier:
    // tasks already running still finish.
    for (aisXferReq &req : ais_handle->request_list) {
        aisXferReq *captured_req = &req;
        ais_handle->taskflow.emplace(
            [captured_req, overall_status = &ais_handle->overall_status]() {
                if (overall_status->load() != NIXL_SUCCESS) {
                    return;
                }
                runHipFileOp(captured_req, overall_status);
            });
    }

    handle = ais_handle.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAisMtEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    auto *ais_handle = static_cast<nixlAisMtReqH *>(handle);

    ais_handle->overall_status.store(NIXL_SUCCESS);
    ais_handle->running_transfer = executor_->run(ais_handle->taskflow);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlAisMtEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *ais_handle = static_cast<nixlAisMtReqH *>(handle);
    if (ais_handle->running_transfer.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready) {
        return NIXL_IN_PROG;
    }
    ais_handle->running_transfer.get();

    return ais_handle->overall_status.load();
}

nixl_status_t
nixlAisMtEngine::releaseReqH(nixlBackendReqH *handle) const {
    delete static_cast<nixlAisMtReqH *>(handle);
    return NIXL_SUCCESS;
}
