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
#include "common/status.h"
#include "common/nixl_log.h"
#include "queue_factory_impl.h"


// Helper class to manage backend availability and selection
class BackendManager {
    public:
        static bool isAioAvailable() {
            return QueueFactory::isAioAvailable();
        }

        static bool isUringAvailable() {
            return QueueFactory::isUringAvailable();
        }

        static std::tuple<nixl_status_t, bool> shouldUseAio(const nixl_b_params_t* custom_params) {
            // Check for explicit backend request
            if (custom_params) {
                // First check if AIO is explicitly requested
                if (custom_params->count("use_aio") > 0) {
                    const auto& value = custom_params->at("use_aio");
                    if (value == "true" || value == "1") {
                        if (!BackendManager::isAioAvailable()) {
                            NIXL_ERROR << "AIO backend requested but not available";
                            return {NIXL_ERR_NOT_SUPPORTED, false};
                        }
                        return {NIXL_SUCCESS, true};
                    }
                }

                // Then check if io_uring is explicitly requested
                if (custom_params->count("use_uring") > 0) {
                    const auto& value = custom_params->at("use_uring");
                    if (value == "true" || value == "1") {
#ifndef HAVE_LIBURING
                        NIXL_ERROR << "io_uring backend requested but not available - not built with liburing support";
                        return {NIXL_ERR_NOT_SUPPORTED, false};
#endif
                        if (!BackendManager::isUringAvailable()) {
                            NIXL_ERROR << "io_uring backend requested but not available at runtime";
                            return {NIXL_ERR_NOT_SUPPORTED, false};
                        }
                        return {NIXL_SUCCESS, false};
                    }
                }
            }

            // If no explicit choice is made or both are false, default to AIO if available
            if (!BackendManager::isAioAvailable()) {
                NIXL_ERROR << "No backend available - AIO not available";
                return {NIXL_ERR_NOT_SUPPORTED, false};
            }

            NIXL_INFO << "Using default AIO backend";
            return {NIXL_SUCCESS, true};
        }
};

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
    , is_prepped(false)
    , status(NIXL_SUCCESS) {

    auto [init_status, should_use_aio] = BackendManager::shouldUseAio(params);
    if (init_status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to determine backend type");
    }
    use_aio_ = should_use_aio;

    try {
        status = initQueues(use_aio_);
        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to initialize queues: " +
                                   std::to_string(status));
        }
    } catch (const std::exception& e) {
        NIXL_ERROR << "Failed to initialize queues: " << e.what();
        status = NIXL_ERR_BACKEND;
    }
}

nixl_status_t nixlPosixBackendReqH::initQueues(bool use_aio) {
    try {
        if (use_aio) {
            queue = QueueFactory::createAioQueue(queue_depth_, operation == NIXL_READ);
            if (!queue) {
                throw std::runtime_error("Failed to create AIO queue");
            }
            NIXL_INFO << "Using AIO backend";
        } else {
#ifdef HAVE_LIBURING
            // Initialize io_uring parameters with basic configuration
            struct io_uring_params params = {};
            // Start with basic parameters, no special flags
            // We can add optimizations like SQPOLL later once basic functionality works

            queue = QueueFactory::createUringQueue(queue_depth_, operation == NIXL_READ, &params);
            if (!queue) {
                throw std::runtime_error("Failed to create io_uring queue");
            }
            NIXL_INFO << "Using io_uring backend";
#else
            NIXL_ERROR << "io_uring support not compiled in";
            return NIXL_ERR_NOT_SUPPORTED;
#endif
        }
        return NIXL_SUCCESS;
    } catch (const std::exception& e) {
        NIXL_ERROR << "Failed to initialize queues: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t nixlPosixBackendReqH::prepXfer() {
    if (!queue) {
        return NIXL_ERR_BACKEND;
    }

    if (is_prepped) {
        return status;
    }

    const int desc_count = static_cast<int>(local.descCount());
    for (int i = 0; i < desc_count; ++i) {
        status = queue->prepareIO(
            remote[i].devId,
            reinterpret_cast<void*>(local[i].addr),
            remote[i].len,
            remote[i].addr
        );

        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Error preparing I/O operation";
            return status;
        }
    }

    is_prepped = true;
    return NIXL_SUCCESS;
}

nixl_status_t nixlPosixBackendReqH::checkXfer() {
    if (!queue) {
        return NIXL_ERR_BACKEND;
    }

    return queue->checkCompleted();
}

nixl_status_t nixlPosixBackendReqH::postXfer() {
    if (!queue) {
        return NIXL_ERR_BACKEND;
    }

    return queue->submit();
}

nixlPosixBackendReqH::~nixlPosixBackendReqH() {
    // Queue will be automatically cleaned up by unique_ptr
}

// -----------------------------------------------------------------------------
// POSIX Engine Implementation
// -----------------------------------------------------------------------------

nixlPosixEngine::nixlPosixEngine(const nixlBackendInitParams* init_params)
    : nixlBackendEngine(init_params) {
    use_aio = true;
    auto [init_status, should_use_aio] = BackendManager::shouldUseAio(init_params->customParams);
    if (init_status != NIXL_SUCCESS) {
        initErr = true;
        NIXL_ERROR << "Failed to initialize POSIX backend - requested backend not available";
        return;
    }
    use_aio = should_use_aio;
    NIXL_INFO << "POSIX backend initialized using " << (use_aio ? "AIO" : "io_uring") << " backend";
}

nixl_status_t nixlPosixEngine::init() {
    if (initErr) {
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}


bool nixlPosixEngine::validatePrepXferParams(const nixl_xfer_op_t &operation,
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

nixl_status_t nixlPosixEngine::registerMem(const nixlBlobDesc &mem,
                                           const nixl_mem_t &nixl_mem,
                                           nixlBackendMD* &out) {
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
                                        const nixl_opt_b_args_t* opt_args) {
    if (!validatePrepXferParams(operation, local, remote, remote_agent, localAgent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    try {
        // Create a params map with our backend selection
        nixl_b_params_t params;
        params["use_uring"] = use_aio ? "false" : "true";
        params["use_aio"] = use_aio ? "true" : "false";

        auto posix_handle = std::make_unique<nixlPosixBackendReqH>(operation, local, remote, opt_args, &params);
        nixl_status_t status = posix_handle->prepXfer();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        handle = posix_handle.release();
        return NIXL_SUCCESS;
    } catch (const std::exception& e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t nixlPosixEngine::postXfer(const nixl_xfer_op_t &operation,
                                        const nixl_meta_dlist_t &local,
                                        const nixl_meta_dlist_t &remote,
                                        const std::string &remote_agent,
                                        nixlBackendReqH* &handle,
                                        const nixl_opt_b_args_t* opt_args) {
    nixl_status_t status = NIXL_SUCCESS;

    status = static_cast<nixlPosixBackendReqH*>(handle)->postXfer();
    NIXL_LOG_AND_RETURN_IF_ERROR(status, "Error in submitting queue");

    return status;
}

nixl_status_t nixlPosixEngine::checkXfer(nixlBackendReqH* handle) {
    return static_cast<nixlPosixBackendReqH*>(handle)->checkXfer();
}

nixl_status_t nixlPosixEngine::releaseReqH(nixlBackendReqH* handle) {
    delete static_cast<nixlPosixBackendReqH*>(handle);
    return NIXL_SUCCESS;
}

nixl_mem_list_t nixlPosixEngine::getSupportedMems() const {
    return supported_mems;
}
