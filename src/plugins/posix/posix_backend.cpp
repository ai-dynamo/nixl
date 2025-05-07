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
#include <liburing.h>
#include <cmath>
#include "posix_backend.h"
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include "common/status.h"
#include "common/nixl_log.h"

namespace {
    static constexpr unsigned int max_posix_ring_size_log = 10;
    static constexpr unsigned int max_posix_ring_size = 1 << max_posix_ring_size_log;
    const nixl_mem_list_t supported_mems = {
        FILE_SEG,
        DRAM_SEG
    };

    bool validatePrepXferParams(const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                const std::string &local_agent) {
        if (remote_agent != local_agent) {
            NIXL_ERROR << absl::StrFormat("Error: Remote agent must match the requesting agent (%s). Got %s", local_agent, remote_agent);
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
}

uringQueue::uringQueue(int num_entries, io_uring_params params)
    : num_entries(num_entries), num_completed(0) {
    memset(&uring, 0, sizeof(uring));

    int uring_init_status = io_uring_queue_init_params(num_entries, &uring, &params);
    if (uring_init_status != 0) {
        throw std::runtime_error(absl::StrFormat("Failed to init io_uring - errno: %d", errno));
    }
}

uringQueue::~uringQueue() {
    io_uring_queue_exit(&uring);
}

nixl_status_t uringQueue::submit() {
    int ret = io_uring_submit(&uring);
    if (ret != num_entries) {
        return NIXL_ERR_BACKEND;
    }
    return NIXL_IN_PROG;
}

nixl_status_t uringQueue::checkCompleted() {
    if (num_completed == num_entries)
        return NIXL_SUCCESS;

    std::array<struct io_uring_cqe*, max_posix_ring_size> cqes;
    const int num_ret_cqes = io_uring_peek_batch_cqe(&uring, cqes.data(), num_entries);

    if (num_ret_cqes < 0)
        return NIXL_ERR_BACKEND;

    for (int i = 0; i < num_ret_cqes; ++i) {
        int res = cqes[i]->res;
        if (res < 0)
            return NIXL_ERR_BACKEND;
        io_uring_cqe_seen(&uring, cqes[i]);
    }

    num_completed += num_ret_cqes;
    if (num_completed == num_entries)
        return NIXL_SUCCESS;
    return NIXL_IN_PROG;
}

struct io_uring_sqe *uringQueue::getSqe() {
    return io_uring_get_sqe(&uring);
}

void nixlPosixBackendReqH::fillUringParams() {
    // We only need 2 params, as only the last one will be the remainder, and others will be max_posix_ring_size
    const unsigned int remainder = (local_desc_count - 1) % max_posix_ring_size + 1;

    switch (num_uring_params) {
        case 2:
            uring_params[1] = {
                .sq_entries = max_posix_ring_size,
                .cq_entries = max_posix_ring_size,
            };
            [[fallthrough]];
        case 1:
            uring_params[0] = {
                .sq_entries = remainder,
                .cq_entries = remainder,
            };
            break;
        default:
            ;
    }
}

void nixlPosixBackendReqH::initUrings() {
    // Initialize all but last uring
    for (int i = 0; i < num_urings - 1; ++i) {
        uring.push_back(std::make_unique<uringQueue>(max_posix_ring_size, uring_params[1]));
    }

    // Initialize last uring with remainder
    uring.push_back(std::make_unique<uringQueue>((local_desc_count - 1) % max_posix_ring_size + 1, uring_params[0]));
}

nixlPosixBackendReqH::nixlPosixBackendReqH(const nixl_xfer_op_t &operation,
                                           const nixl_meta_dlist_t &local,
                                           const nixl_meta_dlist_t &remote,
                                           const nixl_opt_b_args_t* opt_args)
    : operation(operation), local(local), local_desc_count(local.descCount()),
      remote(remote), opt_args(opt_args),
      num_urings(static_cast<int>(std::ceil(static_cast<double>(local_desc_count) / max_posix_ring_size))),
      num_uring_params(std::min(num_urings, 2)),
      uring_params(num_uring_params),
      io_uring_prep_func(operation == NIXL_READ ?
                         reinterpret_cast<io_uring_prep_func_t>(io_uring_prep_read) :
                         reinterpret_cast<io_uring_prep_func_t>(io_uring_prep_write)),
      is_prepped(false), status(NIXL_IN_PROG) {
    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        throw std::invalid_argument(absl::StrFormat("Invalid operation type: %d", operation));
    }

    fillUringParams();
    initUrings();
}

nixlPosixBackendReqH::~nixlPosixBackendReqH() {
    uring.clear();
}

nixl_status_t nixlPosixBackendReqH::prepXfer() {
    NIXL_RETURN_IF_NOT_IN_PROG(status);
    if (is_prepped)
        return status;

    for (int global_entry_index = 0; global_entry_index < local_desc_count; ++global_entry_index) {
        int uring_index = global_entry_index / max_posix_ring_size;
        struct io_uring_sqe *entry = uring[uring_index]->getSqe();
        if (!entry) {
            status = NIXL_ERR_BACKEND;
            NIXL_LOG_AND_RETURN_IF_ERROR(status, "Error in getting sqe");
        }
        io_uring_prep_func(entry, remote[global_entry_index].devId,
                           reinterpret_cast<void *>(local[global_entry_index].addr),
                           remote[global_entry_index].len,
                           remote[global_entry_index].addr);
    }
    is_prepped = true;
    status = NIXL_IN_PROG;
    return status;
}

nixl_status_t nixlPosixBackendReqH::checkXfer() {
    NIXL_RETURN_IF_NOT_IN_PROG(status);

    status = NIXL_SUCCESS;
    for (int i = 0; i < num_urings; ++i) {
        nixl_status_t uring_status = uring[i]->checkCompleted();
        if (uring_status == NIXL_ERR_BACKEND) {
            status = NIXL_ERR_BACKEND;
            NIXL_LOG_AND_RETURN_IF_ERROR(status, "Error in CQE processing");
        }
        if (uring_status == NIXL_IN_PROG)
            status = NIXL_IN_PROG;
    }
    return status;
}

nixl_status_t nixlPosixBackendReqH::postXfer() {
    NIXL_RETURN_IF_NOT_IN_PROG(status);

    for (int i = 0; i < num_urings; ++i) {
        nixl_status_t uring_status = uring[i]->submit();
        if (uring_status == NIXL_ERR_BACKEND) {
            status = NIXL_ERR_BACKEND;
            NIXL_LOG_AND_RETURN_IF_ERROR(status, "Error in submitting io_uring");
        }
    }
    return status;
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
    if (!validatePrepXferParams(operation, local, remote, remote_agent, localAgent))
        return NIXL_ERR_INVALID_PARAM;

    try {
        std::unique_ptr<nixlPosixBackendReqH> posix_handle = std::make_unique<nixlPosixBackendReqH>(operation, local, remote, opt_args);

        nixl_status_t status = posix_handle->prepXfer();
        NIXL_RETURN_IF_NOT_IN_PROG(status);

        handle = posix_handle.release();
    } catch (const std::invalid_argument& e) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, e.what());
    } catch (const std::runtime_error& e) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, e.what());
    } catch (const std::exception& e) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, absl::StrFormat("Unexpected error: %s", e.what()));
    }

    return NIXL_SUCCESS;
}

nixl_status_t nixlPosixEngine::postXfer(const nixl_xfer_op_t &operation,
                                        const nixl_meta_dlist_t &local,
                                        const nixl_meta_dlist_t &remote,
                                        const std::string &remote_agent,
                                        nixlBackendReqH* &handle,
                                        const nixl_opt_b_args_t* opt_args) {
    nixl_status_t status = NIXL_SUCCESS;

    status = static_cast<nixlPosixBackendReqH *>(handle)->postXfer();
    NIXL_LOG_AND_RETURN_IF_ERROR(status, "Error in submitting io_uring");

    return status;
}

nixl_status_t nixlPosixEngine::checkXfer(nixlBackendReqH* handle) {
    return static_cast<nixlPosixBackendReqH *>(handle)->checkXfer();
}

nixl_status_t nixlPosixEngine::releaseReqH(nixlBackendReqH* handle) {
    delete static_cast<nixlPosixBackendReqH *>(handle);
    return NIXL_SUCCESS;
}

nixl_mem_list_t nixlPosixEngine::getSupportedMems() const {
    return supported_mems;
}
