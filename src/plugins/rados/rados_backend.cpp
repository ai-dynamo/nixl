/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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
#include "rados_backend.h"

#include <memory>
#include <string>
#include <utility>

#include "common/nixl_log.h"

namespace {

bool
checkRadosResult(const char *operation, int result) {
    if (result == 0) {
        return true;
    }
    NIXL_ERROR << "RADOS " << operation << " failed with error " << result;
    return false;
}

bool
applyCephOverrides(nixl::rados::cApi &capi,
                   rados_t cluster,
                   const nixl::rados::ceph_options_t &overrides) {
    for (const auto &[option, value] : overrides) {
        const std::string operation = std::string("rados_conf_set(") + option + ')';
        if (!checkRadosResult(operation.c_str(),
                              capi.conf_set(cluster, option.c_str(), value.c_str()))) {
            return false;
        }
    }
    return true;
}

// Initialization-only, exception-safe owner of the cluster/ioctx handles:
// take_*() hands ownership to the engine on success, while any failed path
// unwinds here as ioctx -> cluster.
class bootstrap_resources {
public:
    explicit bootstrap_resources(nixl::rados::cApi &capi) : capi_(capi) {}

    bootstrap_resources(const bootstrap_resources &) = delete;
    bootstrap_resources &
    operator=(const bootstrap_resources &) = delete;

    ~bootstrap_resources() {
        if (ioctx_ != nullptr) {
            capi_.ioctx_destroy(ioctx_);
        }
        if (cluster_ != nullptr) {
            capi_.shutdown(cluster_);
        }
    }

    int
    create_cluster(const char *name) {
        return capi_.create2(&cluster_, nullptr, name, 0);
    }

    int
    conf_read_file(const char *path) {
        return capi_.conf_read_file(cluster_, path);
    }

    int
    connect() {
        return capi_.connect(cluster_);
    }

    int
    create_ioctx(const char *pool) {
        return capi_.ioctx_create(cluster_, pool, &ioctx_);
    }

    rados_t
    cluster() const {
        return cluster_;
    }

    rados_ioctx_t
    take_ioctx() {
        return std::exchange(ioctx_, nullptr);
    }

    rados_t
    take_cluster() {
        return std::exchange(cluster_, nullptr);
    }

private:
    nixl::rados::cApi &capi_;
    rados_t cluster_ = nullptr;
    rados_ioctx_t ioctx_ = nullptr;
};

// Bring up the rados client (create2 -> conf -> overrides -> connect -> pool
// ioctx) from the parsed config. On success the raw handles move into the
// out-params for engine ownership; on any failure everything acquired so far
// unwinds here and nothing is taken.
bool
bootstrapClient(nixl::rados::cApi &capi,
                const nixl::rados::backendConfig &config,
                rados_t &out_cluster,
                rados_ioctx_t &out_ioctx) {
    bootstrap_resources resources(capi);
    if (!checkRadosResult("rados_create2", resources.create_cluster(config.name.c_str()))) {
        return false;
    }
    if (!checkRadosResult("rados_conf_read_file", resources.conf_read_file(config.conf.c_str())) ||
        !applyCephOverrides(capi, resources.cluster(), config.ceph_options) ||
        !checkRadosResult("rados_connect", resources.connect())) {
        return false;
    }
    if (!checkRadosResult("rados_ioctx_create", resources.create_ioctx(config.pool.c_str()))) {
        return false;
    }
    out_ioctx = resources.take_ioctx();
    out_cluster = resources.take_cluster();
    return true;
}

} // namespace

nixlRadosEngine::nixlRadosEngine(const nixlBackendInitParams *init_params)
    : nixlRadosEngine(init_params, std::make_unique<nixl::rados::libRados>()) {}

nixlRadosEngine::nixlRadosEngine(const nixlBackendInitParams *init_params,
                                 std::unique_ptr<nixl::rados::cApi> capi)
    : nixlBackendEngine(init_params),
      capi_(std::move(capi)) {
    if (capi_ == nullptr) {
        NIXL_ERROR << "RADOS engine requires a non-null librados seam";
        initErr = true;
        return;
    }
    const auto config = nixl::rados::parseCustomParams(getCustomParams());
    if (!config) {
        NIXL_ERROR << "RADOS custom parameter validation failed: 'conf', 'pool', and "
                      "'name' must be non-empty, and every other parameter must be "
                      "'ceph.<option>'";
        initErr = true;
        return;
    }
    initialize(*config);
}

void
nixlRadosEngine::initialize(const nixl::rados::backendConfig &config) {
    if (!bootstrapClient(*capi_, config, cluster_, ioctx_)) {
        initErr = true;
        return;
    }
    clusterCreated_ = true;
    ioctxOpen_ = true;
}

nixlRadosEngine::~nixlRadosEngine() {
    if (ioctxOpen_) {
        capi_->ioctx_destroy(ioctx_);
        ioctxOpen_ = false;
    }
    if (clusterCreated_) {
        capi_->shutdown(cluster_);
        clusterCreated_ = false;
    }
}

nixl_status_t
nixlRadosEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    out = nullptr;
    if (nixl_mem != DRAM_SEG && nixl_mem != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (initErr) {
        return NIXL_ERR_BACKEND;
    }
    if (nixl_mem == DRAM_SEG) {
        // Metadata-free by design: out stays null deliberately because the
        // backend never pins, tracks, or validates caller DRAM.
        return NIXL_SUCCESS;
    }
    if (mem.metaInfo.find('\0') != std::string::npos) {
        NIXL_ERROR << "RADOS OBJ_SEG metaInfo contains an embedded NUL byte";
        return NIXL_ERR_INVALID_PARAM;
    }

    std::string object_key;
    std::unique_ptr<nixlRadosObjectMD> md;
    try {
        object_key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
        md = std::make_unique<nixlRadosObjectMD>(mem.devId, object_key);
        std::lock_guard lock(objectsMutex_);
        if (objects_.count(mem.devId) != 0) {
            NIXL_ERROR << "RADOS OBJ_SEG devId " << mem.devId << " is already registered";
            return NIXL_ERR_INVALID_PARAM;
        }
        objects_.emplace(mem.devId, object_key);
    }
    catch (const std::bad_alloc &) {
        NIXL_ERROR << "RADOS backend failed to allocate registration metadata";
        return NIXL_ERR_BACKEND;
    }
    out = md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosEngine::deregisterMem(nixlBackendMD *meta) {
    if (meta == nullptr) {
        return NIXL_SUCCESS;
    }
    const std::unique_ptr<nixlRadosObjectMD> md(static_cast<nixlRadosObjectMD *>(meta));
    std::lock_guard lock(objectsMutex_);
    const auto it = objects_.find(md->devId);
    if (it != objects_.end() && it->second == md->objectKey) {
        objects_.erase(it);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosEngine::prepXfer(const nixl_xfer_op_t &,
                          const nixl_meta_dlist_t &,
                          const nixl_meta_dlist_t &,
                          const std::string &,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *) const {
    handle = nullptr;
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlRadosEngine::postXfer(const nixl_xfer_op_t &,
                          const nixl_meta_dlist_t &,
                          const nixl_meta_dlist_t &,
                          const std::string &,
                          nixlBackendReqH *&,
                          const nixl_opt_b_args_t *) const {
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlRadosEngine::checkXfer(nixlBackendReqH *) const {
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlRadosEngine::releaseReqH(nixlBackendReqH *) const {
    return NIXL_ERR_NOT_SUPPORTED;
}
