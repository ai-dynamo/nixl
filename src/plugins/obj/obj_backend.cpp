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

#include "obj_backend.h"
#include "engine_utils.h"
#include "s3/engine_impl.h"
#include "s3_crt/engine_impl.h"
#if defined HAVE_CUOBJ_CLIENT
#include "s3_accel/engine_impl.h"
#endif
#include <memory>

// -----------------------------------------------------------------------------
// Obj Engine Implementation
// -----------------------------------------------------------------------------

// TODO: Consider a registration pattern as more vendor engines are added.
std::unique_ptr<nixlObjEngineImpl>
createObjEngineImpl(const nixlBackendInitParams *init_params) {
    if (isAcceleratedRequested(init_params->customParams)) {
#if defined HAVE_CUOBJ_CLIENT
        return std::make_unique<S3AccelObjEngineImpl>(init_params);
#else
        throw std::runtime_error("Accelerated Engine support not available!");
#endif
    }

    if (getCrtMinLimit(init_params->customParams) > 0) {
        return std::make_unique<S3CrtObjEngineImpl>(init_params);
    }

    return std::make_unique<DefaultObjEngineImpl>(init_params);
}

std::unique_ptr<nixlObjEngineImpl>
createObjEngineImpl(const nixlBackendInitParams *init_params,
                    std::shared_ptr<iS3Client> s3_client,
                    std::shared_ptr<iS3Client> s3_client_crt) {
    if (isAcceleratedRequested(init_params->customParams)) {
#if defined HAVE_CUOBJ_CLIENT
        return std::make_unique<S3AccelObjEngineImpl>(init_params);
#else
        throw std::runtime_error("Accelerated Engine support not available!");
#endif
    }

    if (getCrtMinLimit(init_params->customParams) > 0) {
        return std::make_unique<S3CrtObjEngineImpl>(init_params, s3_client, s3_client_crt);
    }

    return std::make_unique<DefaultObjEngineImpl>(init_params, s3_client, s3_client_crt);
}

nixlObjEngine::nixlObjEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      impl_(createObjEngineImpl(init_params)) {}

nixlObjEngine::nixlObjEngine(const nixlBackendInitParams *init_params,
                             std::shared_ptr<iS3Client> s3_client,
                             std::shared_ptr<iS3Client> s3_client_crt)
    : nixlBackendEngine(init_params),
      impl_(createObjEngineImpl(init_params, s3_client, s3_client_crt)) {}

nixlObjEngine::~nixlObjEngine() = default;

nixl_mem_list_t
nixlObjEngine::getSupportedMems() const {
    return impl_->getSupportedMems();
}

nixl_status_t
nixlObjEngine::registerMem(const nixlBlobDesc &mem,
                           const nixl_mem_t &nixl_mem,
                           nixlBackendMD *&out) {
    return impl_->registerMem(mem, nixl_mem, out);
}

nixl_status_t
nixlObjEngine::deregisterMem(nixlBackendMD *meta) {
    return impl_->deregisterMem(meta);
}

nixl_status_t
nixlObjEngine::queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const {
    return impl_->queryMem(descs, resp);
}

nixl_status_t
nixlObjEngine::prepXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    return impl_->prepXfer(operation, local, remote, remote_agent, localAgent, handle, opt_args);
}

nixl_status_t
nixlObjEngine::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    return impl_->postXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixl_status_t
nixlObjEngine::checkXfer(nixlBackendReqH *handle) const {
    return impl_->checkXfer(handle);
}

nixl_status_t
nixlObjEngine::releaseReqH(nixlBackendReqH *handle) const {
    return impl_->releaseReqH(handle);
}
