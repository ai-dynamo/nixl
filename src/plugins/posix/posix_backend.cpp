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

#include <iostream>
#include <cmath>
#include <errno.h>
#include <stdexcept>
#include "posix_backend.h"
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include "common/nixl_log.h"
#include "nixl_types.h"
#include "file/file_utils.h"

namespace {
bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (remote_agent != local_agent) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Remote agent must match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);
        return false;
    }

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    if (remote.getType() != FILE_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be FILE_SEG, got %d",
                                      remote.getType());
        return false;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Mismatch in descriptor counts - local: %d, remote: %d",
            local.descCount(),
            remote.descCount());
        return false;
    }

    return true;
}

nixlPosixBackendReqH &
castPosixHandle(nixlBackendReqH *handle) {
    if (!handle) {
        throw nixlPosixBackendReqH::exception("received null handle", NIXL_ERR_INVALID_PARAM);
    }
    return dynamic_cast<nixlPosixBackendReqH &>(*handle);
}

static std::string_view
getIoQueueType(const nixl_b_params_t *custom_params) {
    // Check for explicit backend request
    if (custom_params) {
        // First check if AIO is explicitly requested
        if (custom_params->count("use_aio") > 0) {
            const auto &value = custom_params->at("use_aio");
            if (value == "true" || value == "1") {
                return "AIO";
            }
        }

        // Then check if io_uring is explicitly requested
        if (custom_params->count("use_uring") > 0) {
            const auto &value = custom_params->at("use_uring");
            if (value == "true" || value == "1") {
                return "URING";
            }
        }
        // Then check if linux_aio is explicitly requested
        if (custom_params->count("use_posix_aio") > 0) {
            const auto &value = custom_params->at("use_posix_aio");
            if (value == "true" || value == "1") {
                return "POSIXAIO";
            }
        }
    }

    return nixlPosixIOQueue::getDefaultIoQueueType();
}

static uint32_t
getIOSPoolSize(const nixl_b_params_t *custom_params) {
    uint32_t ios_pool_size = 0;
    if (custom_params) {
        if (custom_params->count("ios_pool_size") > 0) {
            const auto &value = custom_params->at("ios_pool_size");
            ios_pool_size = std::stoi(value);
        }
    }
    return ios_pool_size;
}

static uint32_t
getKernelQueueSize(const nixl_b_params_t *custom_params) {
    int kernel_queue_size = 0;
    if (custom_params) {
        if (custom_params->count("kernel_queue_size") > 0) {
            const auto &value = custom_params->at("kernel_queue_size");
            kernel_queue_size = std::stoi(value);
        }
    }

    return kernel_queue_size;
}

// Log completion percentage at regular intervals (every log_percent_step percent)
void
logOnPercentStep(unsigned int completed, unsigned int total) {
    constexpr unsigned int default_log_percent_step = 10;
    static_assert(default_log_percent_step >= 1 && default_log_percent_step <= 100,
                  "log_percent_step must be in [1, 100]");
    unsigned int log_percent_step = total < 10 ? 1 : default_log_percent_step;

    if (total == 0) {
        NIXL_ERROR << "Tried to log completion percentage with total == 0";
        return;
    }
    // Only log at each percentage step
    if (completed % (total / log_percent_step) == 0) {
        NIXL_DEBUG << absl::StrFormat("Queue progress: %.1f%% complete",
                                      (completed * 100.0 / total));
    }
}
} // namespace

// -----------------------------------------------------------------------------
// POSIX Backend Request Handle Implementation
// -----------------------------------------------------------------------------

// NOTE: we initialize num_confirmed_ios_ to the number of descriptors, so if checkXfer is called
// before postXfer, it will return NIXL_SUCCESS immediately.
nixlPosixBackendReqH::nixlPosixBackendReqH(const nixl_xfer_op_t &op,
                                           const nixl_meta_dlist_t &loc,
                                           const nixl_meta_dlist_t &rem,
                                           const nixl_opt_b_args_t *args,
                                           std::unique_ptr<nixlPosixIOQueue> &io_queue)
    : operation(op),
      local(loc),
      remote(rem),
      opt_args(args),
      queue_depth_(loc.descCount()),
      num_confirmed_ios_(queue_depth_),
      io_queue_(io_queue) {
    NIXL_ASSERT(local.descCount());
    NIXL_ASSERT(remote.descCount());
}

void
nixlPosixBackendReqH::ioDone(uint32_t data_size, int error) {
    num_confirmed_ios_++;
    logOnPercentStep(num_confirmed_ios_, queue_depth_);
}

void
nixlPosixBackendReqH::ioDoneClb(void *ctx, uint32_t data_size, int error) {
    nixlPosixBackendReqH *self = static_cast<nixlPosixBackendReqH *>(ctx);
    self->ioDone(data_size, error);
}

nixl_status_t
nixlPosixBackendReqH::prepXfer() {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixBackendReqH::checkXfer() {
    if (num_confirmed_ios_ == queue_depth_) {
        return NIXL_SUCCESS;
    }

    nixl_status_t status = io_queue_->poll();
    if (status < 0) {
        return status;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlPosixBackendReqH::postXfer() {
    num_confirmed_ios_ = 0;

    for (auto [local_it, remote_it] = std::make_pair(local.begin(), remote.begin());
         local_it != local.end() && remote_it != remote.end();
         ++local_it, ++remote_it) {
        nixl_status_t status = io_queue_->enqueue(remote_it->devId,
                                                  reinterpret_cast<void *>(local_it->addr),
                                                  remote_it->len,
                                                  remote_it->addr,
                                                  operation == NIXL_READ,
                                                  ioDoneClb,
                                                  this);

        if (status != NIXL_SUCCESS) {
            // Currently we do not support partial submissions, so it's all or nothing
            NIXL_ERROR << absl::StrFormat("Error preparing I/O operation: %d", status);
            return status;
        }
    }

    return io_queue_->post();
}

// -----------------------------------------------------------------------------
// POSIX Engine Implementation
// -----------------------------------------------------------------------------

nixlPosixEngine::nixlPosixEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      io_queue_type_(getIoQueueType(init_params->customParams)),
      io_queue_(nixlPosixIOQueue::instantiate(io_queue_type_,
                                              getIOSPoolSize(init_params->customParams),
                                              getKernelQueueSize(init_params->customParams))),
      io_queue_lock_(init_params->syncMode) {
    if (io_queue_type_.empty()) {
        initErr = true;
        NIXL_ERROR << "Failed to initialize POSIX backend - no supported io queue type found";
        return;
    }
    NIXL_INFO << absl::StrFormat("POSIX backend initialized using io queue type: %s",
                                 io_queue_type_);
}

nixl_status_t
nixlPosixEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) != supported_mems.end())
        return NIXL_SUCCESS;

    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlPosixEngine::deregisterMem(nixlBackendMD *) {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixEngine::prepXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    try {
        auto posix_handle =
            std::make_unique<nixlPosixBackendReqH>(operation, local, remote, opt_args, io_queue_);
        NIXL_LOCK_GUARD(io_queue_lock_);
        nixl_status_t status = posix_handle->prepXfer();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        handle = posix_handle.release();
        return NIXL_SUCCESS;
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << absl::StrFormat("Error: %s", e.what());
        return e.code();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << absl::StrFormat("Unexpected error: %s", e.what());
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlPosixEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    try {
        auto &posix_handle = castPosixHandle(handle);
        NIXL_LOCK_GUARD(io_queue_lock_);
        nixl_status_t status = posix_handle.postXfer();
        if (status != NIXL_IN_PROG) {
            NIXL_ERROR << "Error in submitting queue";
        }
        return status;
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlPosixEngine::checkXfer(nixlBackendReqH *handle) const {
    try {
        auto &posix_handle = castPosixHandle(handle);
        NIXL_LOCK_GUARD(io_queue_lock_);
        return posix_handle.checkXfer();
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlPosixEngine::releaseReqH(nixlBackendReqH *handle) const {
    try {
        auto &posix_handle = castPosixHandle(handle);
        posix_handle.~nixlPosixBackendReqH();
        return NIXL_SUCCESS;
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlPosixEngine::queryMem(const nixl_reg_dlist_t &descs,
                          std::vector<nixl_query_resp_t> &resp) const {
    // Extract metadata from descriptors which are file names
    // Different plugins might customize parsing of metaInfo to get the file names
    std::vector<nixl_blob_t> metadata(descs.descCount());
    for (int i = 0; i < descs.descCount(); ++i)
        metadata[i] = descs[i].metaInfo;

    return nixl::queryFileInfoList(metadata, resp);
}
