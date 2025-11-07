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

#include <iostream>
#include <cmath>
#include <errno.h>
#include <stdexcept>
#include "posix_backend.h"
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include "common/nixl_log.h"
#include "queue_factory_impl.h"
#include "nixl_types.h"
#include "file/file_utils.h"

namespace {
    bool isValidPrepXferParams(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               const std::string &local_agent) {
        if (remote_agent != local_agent) {
            NIXL_ERROR << absl::StrFormat("Error: Remote agent must match the requesting agent (%s). Got %s",
                                        local_agent, remote_agent);
            return false;
        }

        if (local.getType() != DRAM_SEG) {
            NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d", local.getType());
            return false;
        }

        if (remote.getType() != FILE_SEG) {
            NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be FILE_SEG, got %d", remote.getType());
            return false;
        }

        if (local.descCount() != remote.descCount()) {
            NIXL_ERROR << absl::StrFormat("Error: Mismatch in descriptor counts - local: %d, remote: %d",
                                        local.descCount(), remote.descCount());
            return false;
        }

        return true;
    }

    nixlPosixBackendReqH& castPosixHandle(nixlBackendReqH* handle) {
        if (!handle) {
            throw nixlPosixBackendReqH::exception("received null handle", NIXL_ERR_INVALID_PARAM);
        }
        return dynamic_cast<nixlPosixBackendReqH&>(*handle);
    }

    // Stringify function for queue_t
    inline const char* to_string(nixlPosixQueue::queue_t type) {
        using queue_t = nixlPosixQueue::queue_t;
        switch (type) {
            case queue_t::AIO: return "AIO";
            case queue_t::URING: return "URING";
            case queue_t::POSIXAIO:
                return "POSIXAIO";
            case queue_t::UNSUPPORTED: return "UNSUPPORTED";
            default: return "UNKNOWN";
        }
    }

    static nixlPosixQueue::queue_t getQueueType(const nixl_b_params_t* custom_params) {
        using queue_t = nixlPosixQueue::queue_t;

        // Check for explicit backend request
        if (custom_params) {
            // First check if AIO is explicitly requested
            if (custom_params->count("use_aio") > 0) {
                const auto& value = custom_params->at("use_aio");
                if (value == "true" || value == "1") {
                    if (!QueueFactory::isLinuxAioAvailable()) {
                        NIXL_ERROR << "linux_aio backend requested but not available at runtime";
                        return queue_t::UNSUPPORTED;
                    }
                    return queue_t::AIO;
                }
            }

            // Then check if io_uring is explicitly requested
            if (custom_params->count("use_uring") > 0) {
                const auto& value = custom_params->at("use_uring");
                if (value == "true" || value == "1") {
                    if (!QueueFactory::isUringAvailable()) {
                        NIXL_ERROR << "io_uring backend requested but not available at runtime";
                        return queue_t::UNSUPPORTED;
                    }
                    return queue_t::URING;
                }
            }

            // Then check if linux_aio is explicitly requested
            if (custom_params->count("use_posix_aio") > 0) {
                const auto &value = custom_params->at("use_posix_aio");
                if (value == "true" || value == "1") {
                    return queue_t::POSIXAIO;
                }
            }
        }

        if (QueueFactory::isLinuxAioAvailable()) {
            return queue_t::AIO;
        }
        if (QueueFactory::isUringAvailable()) {
            return queue_t::URING;
        }
        return queue_t::POSIXAIO;
    }
}

// -----------------------------------------------------------------------------
// POSIX Backend Request Handle Implementation
// -----------------------------------------------------------------------------

nixlPosixBackendReqH::nixlPosixBackendReqH(const nixl_xfer_op_t &op,
                                           const nixl_meta_dlist_t &loc,
                                           const nixl_meta_dlist_t &rem,
                                           const nixl_opt_b_args_t* args,
                                           const nixl_b_params_t* params)
    : operation(op)
    , local(loc)
    , remote(rem)
    , opt_args(args)
    , custom_params_(params)
    , queue_depth_(loc.descCount())
    , queue_type_(getQueueType(params)) {
    if (queue_type_ == nixlPosixQueue::queue_t::UNSUPPORTED) {
        throw exception(absl::StrFormat("Unsupported queue type"), NIXL_ERR_NOT_SUPPORTED);
    }

    if (local.descCount() == 0 || remote.descCount() == 0) {
        throw exception(
            absl::StrFormat("Invalid descriptor count - local: %zu, remote: %zu", local.descCount(), remote.descCount()),
            NIXL_ERR_INVALID_PARAM);
    }

    nixl_status_t status = initQueues();
    if (status != NIXL_SUCCESS) {
        throw exception(absl::StrFormat("Failed to initialize queues: %s", to_string(queue_type_)),
                        status);
    }
}


nixl_status_t nixlPosixBackendReqH::initQueues() {
    try {
        switch (queue_type_) {
            case nixlPosixQueue::queue_t::AIO:
                queue = QueueFactory::createLinuxAioQueue(queue_depth_, operation);
                break;
            case nixlPosixQueue::queue_t::URING:
                queue = QueueFactory::createUringQueue(queue_depth_, operation);
                break;
            case nixlPosixQueue::queue_t::POSIXAIO:
                queue = QueueFactory::createPosixAioQueue(queue_depth_, operation);
                break;
            default:
                NIXL_ERROR << absl::StrFormat("Invalid queue type: %s", to_string(queue_type_));
                return NIXL_ERR_INVALID_PARAM;
        }
        return NIXL_SUCCESS;
    } catch (const nixlPosixBackendReqH::exception& e) {
        NIXL_ERROR << absl::StrFormat("Failed to initialize queues: %s", e.what());
        return e.code();
    } catch (const std::exception& e) {
        NIXL_ERROR << absl::StrFormat("Failed to initialize queues: %s", e.what());
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t nixlPosixBackendReqH::prepXfer() {
    for (auto [local_it, remote_it] = std::make_pair(local.begin(), remote.begin());
         local_it != local.end() && remote_it != remote.end();
         ++local_it, ++remote_it) {
        nixl_status_t status = queue->prepIO(
            remote_it->devId,
            reinterpret_cast<void*>(local_it->addr),
            remote_it->len,
            remote_it->addr
        );

        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Error preparing I/O operation";
            return status;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t nixlPosixBackendReqH::checkXfer() {
    return queue->checkCompleted();
}

nixl_status_t nixlPosixBackendReqH::postXfer() {
    return queue->submit (local, remote);
}

// -----------------------------------------------------------------------------
// POSIX Engine Implementation
// -----------------------------------------------------------------------------

nixlPosixEngine::nixlPosixEngine(const nixlBackendInitParams* init_params)
    : nixlBackendEngine(init_params)
    , queue_type_(getQueueType(init_params->customParams)) {
    if (queue_type_ == nixlPosixQueue::queue_t::UNSUPPORTED) {
        initErr = true;
        NIXL_ERROR << absl::StrFormat(
            "Failed to initialize POSIX backend - requested queue type not available: %s",
            to_string(queue_type_));
        return;
    }
    NIXL_INFO << absl::StrFormat("POSIX backend initialized using queue type: %s",
                                 to_string(queue_type_));
}

nixl_status_t nixlPosixEngine::registerMem(const nixlBlobDesc &mem,
                                           const nixl_mem_t &nixl_mem,
                                           nixlBackendMD* &out) {
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) != supported_mems.end())
        return NIXL_SUCCESS;

    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t nixlPosixEngine::deregisterMem(nixlBackendMD *) {
    return NIXL_SUCCESS;
}

nixl_status_t nixlPosixEngine::prepXfer(const nixl_xfer_op_t &operation,
                                        const nixl_meta_dlist_t &local,
                                        const nixl_meta_dlist_t &remote,
                                        const std::string &remote_agent,
                                        nixlBackendReqH* &handle,
                                        const nixl_opt_b_args_t* opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    try {
        // Create a params map with our backend selection
        nixl_b_params_t params;
        switch (queue_type_) {
            case nixlPosixQueue::queue_t::AIO:
                params["use_aio"] = "true";
                break;
            case nixlPosixQueue::queue_t::URING:
                params["use_uring"] = "true";
                break;
            case nixlPosixQueue::queue_t::POSIXAIO:
                params["use_posix_aio"] = "true";
                break;
            default:
                NIXL_ERROR << absl::StrFormat("Invalid queue type: %s", to_string(queue_type_));
                return NIXL_ERR_INVALID_PARAM;
        }

        auto posix_handle = std::make_unique<nixlPosixBackendReqH>(operation, local, remote, opt_args, &params);
        nixl_status_t status = posix_handle->prepXfer();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        handle = posix_handle.release();
        return NIXL_SUCCESS;
    } catch (const nixlPosixBackendReqH::exception& e) {
        NIXL_ERROR << absl::StrFormat("Error: %s", e.what());
        return e.code();
    } catch (const std::exception& e) {
        NIXL_ERROR << absl::StrFormat("Unexpected error: %s", e.what());
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t nixlPosixEngine::postXfer(const nixl_xfer_op_t &operation,
                                        const nixl_meta_dlist_t &local,
                                        const nixl_meta_dlist_t &remote,
                                        const std::string &remote_agent,
                                        nixlBackendReqH* &handle,
                                        const nixl_opt_b_args_t* opt_args) const {
    try {
        auto& posix_handle = castPosixHandle(handle);
        nixl_status_t status = posix_handle.postXfer();
        if (status != NIXL_IN_PROG) {
            NIXL_ERROR << "Error in submitting queue";
        }
        return status;
    } catch (const nixlPosixBackendReqH::exception& e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    return NIXL_ERR_BACKEND;
}

nixl_status_t nixlPosixEngine::checkXfer(nixlBackendReqH* handle) const {
    try {
        auto& posix_handle = castPosixHandle(handle);
        return posix_handle.checkXfer();
    }
    catch (const nixlPosixBackendReqH::exception& e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    return NIXL_ERR_BACKEND;
}

nixl_status_t nixlPosixEngine::releaseReqH(nixlBackendReqH* handle) const {
    try {
        auto& posix_handle = castPosixHandle(handle);
        posix_handle.~nixlPosixBackendReqH();
        return NIXL_SUCCESS;
    } catch (const nixlPosixBackendReqH::exception& e) {
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
