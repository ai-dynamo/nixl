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
#include "obj_engine_registry.h"
#include "engine_utils.h"
#include "common/nixl_log.h"
#include "s3/engine_impl.h"
#include "s3_crt/engine_impl.h"
#include <memory>

// -----------------------------------------------------------------------------
// Obj Engine Implementation
// -----------------------------------------------------------------------------
//
// Engine selection:
//   * Default (no params)                         -> DefaultObjEngineImpl
//   * crtMinLimit > 0                             -> S3CrtObjEngineImpl
//   * accelerated=true (no type / type=s3)        -> GenericObjEngineImpl
//   * accelerated=true, type=dell                 -> vendor-specific engine
//
// The preferred GPU-direct path is the standard, protocol-compliant
// S3-over-RDMA engine selected by `accelerated=true` with no `type` (or
// `type=s3`). It is built on the S3-over-RDMA capability of the standard S3
// client (see s3/rdma.h): it complies with the published `x-amz-rdma-*` protocol
// and requires NO per-vendor code — any S3 endpoint that implements the protocol
// (e.g. MinIO AIStor) works, and so would AWS S3 if it adopted it. RDMA is an
// explicit assertion (not auto-probed, and no silent HTTP fallback — see
// s3/client.cpp for the rationale and the future auto-fallback path).
//
// `accelerated=true, type=dell` selects a vendor-specific engine for servers
// that use vendor-specific RDMA headers; Dell uses an `x-rdma-info` header. Both
// the standard and vendor engines resolve through `objAccelEngineRegistry`: the
// standard engine registers under "s3" (a missing `type` is normalized to "s3"
// below), the Dell engine under "dell".

namespace {

template<typename... Args>
std::unique_ptr<nixlObjEngineImpl>
createAccelEngine(const nixl_b_params_t *custom_params, Args &&...args) {
    // A missing `type` selects the standard protocol-compliant engine. Normalize
    // to "s3" rather than looking up "" so resolution is deterministic: the
    // cuobj-gated s3_accel base also registers under "", and the registry
    // keeps whichever registrar runs first (unspecified static-init order).
    std::string type = getAccelType(custom_params);
    if (type.empty()) {
        type = "s3";
    }
    try {
        return objAccelEngineRegistry::instance().create(type, std::forward<Args>(args)...);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create accelerated engine: " << e.what();
        throw;
    }
}

} // namespace

std::unique_ptr<nixlObjEngineImpl>
createObjEngineImpl(const nixlBackendInitParams *init_params) {
    if (isAcceleratedRequested(init_params->customParams)) {
        return createAccelEngine(init_params->customParams, init_params);
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
        return createAccelEngine(
            init_params->customParams, init_params, std::move(s3_client), std::move(s3_client_crt));
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
