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

#include "null_backend.h"
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include "common/nixl_log.h"

nixlNullEngine::nixlNullEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    NIXL_INFO << "NULL backend initialized - all operations complete instantly";
}

nixl_status_t
nixlNullEngine::registerMem(const nixlBlobDesc &mem,
                            const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlNullEngine::deregisterMem(nixlBackendMD *) {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlNullEngine::prepXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    // Reuse member handle for minimal overhead
    handle = &null_request_;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlNullEngine::postXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    // Operation completes instantly - return SUCCESS instead of IN_PROG
    return NIXL_SUCCESS;
}

nixl_status_t
nixlNullEngine::checkXfer(nixlBackendReqH *handle) const {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlNullEngine::releaseReqH(nixlBackendReqH *handle) const {
    return NIXL_SUCCESS;
}
