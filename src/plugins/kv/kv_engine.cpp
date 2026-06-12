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

/**
 * @file kv_engine.cpp
 * @brief nixlKVEngine delegation layer; forwards nixlBackendEngine calls to impl_.
 */

#include "kv_engine.h"
#include <cassert>

nixlKVEngine::nixlKVEngine(const nixlBackendInitParams *init_params,
                           std::unique_ptr<nixlKVEngineImpl> impl)
    : nixlBackendEngine(init_params),
      impl_(std::move(impl)) {
    assert(impl_ && "nixlKVEngine requires a non-null nixlKVEngineImpl");
}

nixlKVEngine::~nixlKVEngine() = default;

nixl_mem_list_t
nixlKVEngine::getSupportedMems() const {
    return impl_->getSupportedMems();
}

nixl_status_t
nixlKVEngine::registerMem(const nixlBlobDesc &mem,
                          const nixl_mem_t &nixl_mem,
                          nixlBackendMD *&out) {
    return impl_->registerMem(mem, nixl_mem, out);
}

nixl_status_t
nixlKVEngine::deregisterMem(nixlBackendMD *meta) {
    return impl_->deregisterMem(meta);
}

nixl_status_t
nixlKVEngine::queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const {
    return impl_->queryMem(descs, resp);
}

nixl_status_t
nixlKVEngine::prepXfer(const nixl_xfer_op_t &operation,
                       const nixl_meta_dlist_t &local,
                       const nixl_meta_dlist_t &remote,
                       const std::string &remote_agent,
                       nixlBackendReqH *&handle,
                       const nixl_opt_b_args_t *opt_args) const {
    return impl_->prepXfer(operation, local, remote, remote_agent, localAgent, handle, opt_args);
}

nixl_status_t
nixlKVEngine::postXfer(const nixl_xfer_op_t &operation,
                       const nixl_meta_dlist_t &local,
                       const nixl_meta_dlist_t &remote,
                       const std::string &remote_agent,
                       nixlBackendReqH *&handle,
                       const nixl_opt_b_args_t *opt_args) const {
    return impl_->postXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixl_status_t
nixlKVEngine::checkXfer(nixlBackendReqH *handle) const {
    return impl_->checkXfer(handle);
}

nixl_status_t
nixlKVEngine::releaseReqH(nixlBackendReqH *handle) const {
    return impl_->releaseReqH(handle);
}
