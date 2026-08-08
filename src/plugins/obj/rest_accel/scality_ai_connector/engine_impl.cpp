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

#include "engine_impl.h"
#include "rest_client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include "cuobj_rdma_token_client.h"
#include "device_select.h"
#include <cassert>
#include <cstdlib>
#include <memory>
#include <future>
#include <vector>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <optional>

#include <cuda_runtime.h>

#include "obj_engine_registry.h"

namespace {

/**
 * Scoped CUDA device switch. cuFile's buffer registration validates that a
 * device pointer belongs to the current CUDA context; when buffers live on
 * different GPUs (multi-GPU initiator), the current device must match the
 * buffer's device or cuFileBufRegister fails the device-memory check (-5011).
 * Restores the previous device on scope exit.
 */
class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int dev) {
        if (cudaGetDevice(&prevDev_) != cudaSuccess) {
            return;
        }
        if (dev >= 0 && dev != prevDev_ && cudaSetDevice(dev) == cudaSuccess) {
            restore_ = true;
        }
    }

    ~CudaDeviceGuard() {
        if (restore_) {
            cudaSetDevice(prevDev_);
        }
    }

    CudaDeviceGuard(const CudaDeviceGuard &) = delete;
    CudaDeviceGuard &
    operator=(const CudaDeviceGuard &) = delete;

private:
    int prevDev_ = 0;
    bool restore_ = false;
};

objAccelEngineRegistrar reg_scality(
    "scality_ai_connector",
    [](const nixlBackendInitParams *p) { return std::make_unique<ScalityObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p, std::shared_ptr<iS3Client>, std::shared_ptr<iS3Client>) {
        return std::make_unique<ScalityObjEngineImpl>(p);
    });

/**
 * RDMA context structure for cuObject operations.
 */
struct rdma_ctx_t {
    /// RDMA descriptor string
    std::string rdma_desc;
};

std::string
objKeyFor(const std::string &metaInfo, uint64_t devId) {
    return metaInfo.empty() ? std::to_string(devId) : metaInfo;
}

/**
 * Validate parameters for prepXfer operation.
 */
bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (remote_agent != local_agent) {
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);
    }

    if ((local.getType() != DRAM_SEG) && (local.getType() != VRAM_SEG)) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local memory type must be VRAM_SEG or DRAM_SEG, got %d", local.getType());
        return false;
    }

    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be OBJ_SEG, got %d",
                                      remote.getType());
        return false;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local and remote descriptor counts must match. Got %d local, %d remote",
            local.descCount(),
            remote.descCount());
        return false;
    }

    return true;
}

/**
 * Transfer request handle for Scality AI Connector RDMA operations.
 */
class scalityObjTransferRequestH {
public:
    uintptr_t addr;
    size_t size;
    size_t offset;
    std::string rdma_desc;
    std::string obj_key;
    rdma_ctx_t ctx;

    scalityObjTransferRequestH() : addr(0), size(0), offset(0) {}

    scalityObjTransferRequestH(uintptr_t a, size_t s, size_t off) : addr(a), size(s), offset(off) {}

    ~scalityObjTransferRequestH() = default;
};

/**
 * Backend request handle for Scality AI Connector RDMA operations.
 * Manages multiple transfer requests and their completion futures.
 */
class nixlScalityObjBackendReqH : public nixlBackendReqH {
public:
    std::vector<scalityObjTransferRequestH> reqs_;
    std::vector<std::future<nixl_status_t>> statusFutures_;

    nixlScalityObjBackendReqH() = default;
    ~nixlScalityObjBackendReqH() = default;

    nixl_status_t
    getOverallStatus() {
        bool has_pending = false;
        auto it = statusFutures_.begin();
        while (it != statusFutures_.end()) {
            if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                auto current_status = it->get();
                if (current_status != NIXL_SUCCESS) {
                    statusFutures_.clear();
                    return current_status;
                }
                it = statusFutures_.erase(it);
            } else {
                ++it;
                has_pending = true;
            }
        }
        return has_pending ? NIXL_IN_PROG : NIXL_SUCCESS;
    }
};

/**
 * Metadata for Scality AI Connector RDMA operations.
 */
class nixlScalityObjMetadata : public nixlBackendMD {
public:
    nixlScalityObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(obj_key),
          localAddr(0) {}

    nixlScalityObjMetadata(nixl_mem_t nixl_mem, uintptr_t addr, uint64_t dev_id = 0)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(""),
          localAddr(addr) {}

    ~nixlScalityObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string objKey;
    uintptr_t localAddr;
};

/**
 * cuObject get callback: extracts RDMA descriptor from cuFile RDMA info.
 */
static ssize_t
objectGet(const void *handle,
          char *buf,
          size_t size,
          loff_t offset,
          const cufileRDMAInfo_t *infop) {
    if (infop == nullptr || infop->desc_str == nullptr) {
        NIXL_ERROR << "objectGet: infop or infop->desc_str is null";
        return -EINVAL;
    }

    void *ctx = iRdmaTokenClient::getCtx(handle);
    if (ctx == nullptr) {
        NIXL_ERROR << "objectGet: context is null";
        return -EINVAL;
    }
    NIXL_DEBUG << "objectGet: handle=" << handle << ", buf=" << static_cast<const void *>(buf)
               << ", size=" << size << ", offset=" << offset << ", infop=" << infop;
    rdma_ctx_t *rctx = static_cast<rdma_ctx_t *>(ctx);
    rctx->rdma_desc = infop->desc_str;
    return 0;
}

/**
 * cuObject put callback: extracts RDMA descriptor from cuFile RDMA info.
 */
static ssize_t
objectPut(const void *handle,
          const char *buf,
          size_t size,
          loff_t offset,
          const cufileRDMAInfo_t *infop) {
    if (infop == nullptr || infop->desc_str == nullptr) {
        NIXL_ERROR << "objectPut: infop or infop->desc_str is null";
        return -EINVAL;
    }

    void *ctx = iRdmaTokenClient::getCtx(handle);
    if (ctx == nullptr) {
        NIXL_ERROR << "objectPut: context is null";
        return -EINVAL;
    }
    NIXL_DEBUG << "objectPut: handle=" << handle << ", buf=" << static_cast<const void *>(buf)
               << ", size=" << size << ", offset=" << offset << ", infop=" << infop;
    rdma_ctx_t *rctx = static_cast<rdma_ctx_t *>(ctx);
    rctx->rdma_desc = infop->desc_str;
    return 0;
}

/// cuObject I/O operations for Scality AI Connector
CUObjIOOps scality_ops = {.get = objectGet, .put = objectPut};

// Return the injected client when provided (the test seam), otherwise build the
// production RestClient. Lets connectorClient_ be initialized as const in the
// member initializer list.
std::shared_ptr<iRestClient>
makeConnectorClient(const nixlBackendInitParams *init_params,
                    const std::shared_ptr<iRestClient> &injected) {
    if (injected) {
        return injected;
    }
    // init_params is only optional when a client is injected (the test seam);
    // the production constructor always supplies a non-null init_params.
    assert(init_params != nullptr && "init_params must be non-null when no RestClient is injected");
    return std::make_shared<RestClient>(init_params->customParams);
}

} // namespace

ScalityObjEngineImpl::ScalityObjEngineImpl(const nixlBackendInitParams *init_params)
    : ScalityObjEngineImpl(init_params, nullptr) {}

ScalityObjEngineImpl::ScalityObjEngineImpl(const nixlBackendInitParams *init_params,
                                           const std::shared_ptr<iRestClient> &connector_client)
    // DC transport via NVIDIA cuObject (CUOBJ_PROTO_RDMA_DC_V1).
    : cuClient_(std::make_shared<CuObjRdmaTokenClient>(scality_ops)),
      connectorClient_(makeConnectorClient(init_params, connector_client)) {
    NIXL_INFO << "Object storage backend initialized with Scality AI Connector RDMA client";

    // Number of GPUs, used to spread DRAM MRs across NICs (see targetCudaDevice).
    // On error, leave gpuCount_ at 0 so DRAM registration keeps the current device.
    if (cudaGetDeviceCount(&gpuCount_) != cudaSuccess || gpuCount_ < 0) {
        gpuCount_ = 0;
    }

    if (!cuClient_ || !cuClient_->isConnected()) {
        NIXL_ERROR << "RDMA token client failed to connect.";
        return;
    }
}

nixl_status_t
ScalityObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                  const nixl_mem_t &nixl_mem,
                                  nixlBackendMD *&out) {
    if (!cuClient_ || !cuClient_->isConnected()) {
        NIXL_ERROR << "RDMA token client is not connected.";
        return NIXL_ERR_BACKEND;
    }

    auto supported_mems = {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (nixl_mem == OBJ_SEG) {
        std::unique_ptr<nixlScalityObjMetadata> obj_md = std::make_unique<nixlScalityObjMetadata>(
            nixl_mem, mem.devId, objKeyFor(mem.metaInfo, mem.devId));
        devIdToObjKey_[mem.devId] = obj_md->objKey;
        out = obj_md.release();
    } else if ((nixl_mem == DRAM_SEG) || (nixl_mem == VRAM_SEG)) {
        if (mem.len > CUOBJ_MAX_MEMORY_REG_SIZE) {
            NIXL_ERROR << "Memory size too large for cuObject registration: " << mem.len;
            return NIXL_ERR_NOT_SUPPORTED;
        }

        NIXL_DEBUG << absl::StrFormat("registerMem: addr=0x%016x, len=%zu, nixl_mem=%d, devId=%d",
                                      mem.addr,
                                      mem.len,
                                      nixl_mem,
                                      mem.devId);
        std::unique_ptr<nixlScalityObjMetadata> mem_md =
            std::make_unique<nixlScalityObjMetadata>(nixl_mem, mem.addr, mem.devId);

        // Bind this MR (and its primary GID/NIC) to a CUDA device: VRAM to its own
        // GPU, DRAM round-robin across GPUs so host buffers fan across NICs.
        std::optional<CudaDeviceGuard> dev_guard;
        int target = targetCudaDevice(nixl_mem, mem.devId, gpuCount_);
        if (target >= 0) {
            dev_guard.emplace(target);
        }

        cuObjErr_t cuda_status = cuClient_->cuMemObjGetDescriptor((void *)(mem.addr), mem.len);
        if (cuda_status != CU_OBJ_SUCCESS) {
            NIXL_ERROR << "cuMemObjGetDescriptor failed with status: " << cuda_status;
            const char *cfg = std::getenv("CUFILE_ENV_PATH_JSON");
            NIXL_ERROR << "Hint: check rdma_dev_addr_list in " << (cfg ? cfg : "/etc/cufile.json");
            return NIXL_ERR_BACKEND;
        }
        out = mem_md.release();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    nixlScalityObjMetadata *md = static_cast<nixlScalityObjMetadata *>(meta);
    if (md) {
        if (md->nixlMem == OBJ_SEG) {
            std::unique_ptr<nixlScalityObjMetadata> obj_md_ptr =
                std::unique_ptr<nixlScalityObjMetadata>(md);
            devIdToObjKey_.erase(obj_md_ptr->devId);
        } else if ((md->nixlMem == DRAM_SEG) || (md->nixlMem == VRAM_SEG)) {
            std::unique_ptr<nixlScalityObjMetadata> mem_md_ptr =
                std::unique_ptr<nixlScalityObjMetadata>(md);
            // Deregister under the same device the MR was registered on so
            // cuObject releases it against the matching context (see registerMem).
            std::optional<CudaDeviceGuard> dev_guard;
            int target = targetCudaDevice(mem_md_ptr->nixlMem, mem_md_ptr->devId, gpuCount_);
            if (target >= 0) {
                dev_guard.emplace(target);
            }
            cuObjErr_t cuda_status =
                cuClient_->cuMemObjPutDescriptor((void *)(mem_md_ptr->localAddr));
            if (cuda_status != CU_OBJ_SUCCESS) {
                NIXL_ERROR << "cuMemObjPutDescriptor failed with status: " << cuda_status;
                // mem_md_ptr destructor frees the metadata as deregisterMem consumes it
                // regardless of outcome.
                return NIXL_ERR_BACKEND;
            }
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               const std::string &local_agent,
                               nixlBackendReqH *&handle,
                               const nixl_opt_b_args_t *opt_args) const {
    if (!cuClient_ || !cuClient_->isConnected()) {
        NIXL_ERROR << "RDMA token client is not connected.";
        return NIXL_ERR_BACKEND;
    }

    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto req_h = std::make_unique<nixlScalityObjBackendReqH>();

    for (int i = 0; i < local.descCount(); ++i) {
        scalityObjTransferRequestH req(local[i].addr, local[i].len, remote[i].addr);

        auto obj_key_search = devIdToObjKey_.find(remote[i].devId);
        if (obj_key_search == devIdToObjKey_.end()) {
            NIXL_ERROR << "The object segment key " << remote[i].devId
                       << " is not registered with the backend";
            return NIXL_ERR_INVALID_PARAM;
        }
        req.obj_key = obj_key_search->second;

        if (operation == NIXL_WRITE) {
            ssize_t cuda_status =
                cuClient_->cuObjPut(&req.ctx, (void *)req.addr, req.size, req.offset);
            if (cuda_status < 0) {
                NIXL_ERROR << "cuObjPut failed with status: " << cuda_status;
                return NIXL_ERR_BACKEND;
            }
        } else if (operation == NIXL_READ) {
            ssize_t cuda_status =
                cuClient_->cuObjGet(&req.ctx, (void *)req.addr, req.size, req.offset);
            if (cuda_status < 0) {
                NIXL_ERROR << "cuObjGet failed with status: " << cuda_status;
                return NIXL_ERR_BACKEND;
            }
        }
        req.rdma_desc = req.ctx.rdma_desc;
        // Log only the descriptor length, never the raw RDMA token (sensitive).
        NIXL_DEBUG << absl::StrFormat(
            "prepXfer: addr=0x%016x, size=%zu, offset=%zu, rdma_desc_len=%zu",
            req.addr,
            req.size,
            req.offset,
            req.rdma_desc.size());

        req_h->reqs_.push_back(req);
    }

    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               nixlBackendReqH *&handle,
                               const nixl_opt_b_args_t *opt_args) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlScalityObjBackendReqH *req_h = static_cast<nixlScalityObjBackendReqH *>(handle);

    for (const auto &req : req_h->reqs_) {
        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->statusFutures_.push_back(status_promise->get_future());

        if (operation == NIXL_WRITE) {
            connectorClient_->putObjectRdmaAsync(req.obj_key,
                                                 req.addr,
                                                 req.size,
                                                 req.offset,
                                                 req.rdma_desc,
                                                 [status_promise](bool success) {
                                                     status_promise->set_value(
                                                         success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
                                                 });
        } else {
            connectorClient_->getObjectRdmaAsync(req.obj_key,
                                                 req.addr,
                                                 req.size,
                                                 req.offset,
                                                 req.rdma_desc,
                                                 [status_promise](bool success) {
                                                     status_promise->set_value(
                                                         success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
                                                 });
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
ScalityObjEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlScalityObjBackendReqH *req_h = static_cast<nixlScalityObjBackendReqH *>(handle);
    return req_h->getOverallStatus();
}

nixl_status_t
ScalityObjEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlScalityObjBackendReqH *req_h = static_cast<nixlScalityObjBackendReqH *>(handle);
    delete req_h;
    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::queryMem(const nixl_reg_dlist_t &descs,
                               std::vector<nixl_query_resp_t> &resp) const {
    resp.assign(descs.descCount(), std::nullopt);

    if (!connectorClient_) {
        NIXL_ERROR << "queryMem: REST client not available";
        return NIXL_ERR_BACKEND;
    }

    struct perDescriptorState {
        std::shared_ptr<std::promise<void>> promise;
        std::shared_ptr<std::atomic<bool>> completed;
    };

    std::vector<std::future<void>> futures;
    std::vector<perDescriptorState> states;
    futures.reserve(descs.descCount());
    states.reserve(descs.descCount());
    // Shared so an async callback that fires after queryMem returns (e.g. a
    // timed-out request) never touches this stack frame.
    auto shared_resp =
        std::make_shared<std::vector<nixl_query_resp_t>>(descs.descCount(), std::nullopt);
    auto has_error = std::make_shared<std::atomic<bool>>(false);

    // One aggregate deadline for all checks: concurrent HEADs share a single
    // bound, so a hung endpoint can't make this block for 30s * descCount().
    constexpr auto kQueryTimeout = std::chrono::seconds(30);
    const auto deadline = std::chrono::steady_clock::now() + kQueryTimeout;

    for (int i = 0; i < descs.descCount(); ++i) {
        const auto &desc = descs[i];
        // Mirror the key convention used by registerMem: metaInfo, or the
        // device id when no key was supplied.
        std::string key = objKeyFor(desc.metaInfo, desc.devId);

        perDescriptorState state{std::make_shared<std::promise<void>>(),
                                 std::make_shared<std::atomic<bool>>(false)};
        futures.push_back(state.promise->get_future());
        states.push_back(state);

        connectorClient_->checkObjectExistsAsync(
            key,
            [shared_resp, has_error, i, promise = state.promise, completed = state.completed](
                std::optional<bool> exists) {
                if (completed->exchange(true)) {
                    return;
                }
                if (!exists.has_value()) {
                    (*shared_resp)[i] = std::nullopt;
                    has_error->store(true, std::memory_order_relaxed);
                } else {
                    (*shared_resp)[i] =
                        *exists ? nixl_query_resp_t{nixl_b_params_t{}} : std::nullopt;
                }
                promise->set_value();
            });
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        if (futures[i].wait_until(deadline) == std::future_status::timeout) {
            if (!states[i].completed->exchange(true)) {
                (*shared_resp)[i] = std::nullopt;
                has_error->store(true, std::memory_order_relaxed);
            }
        }
    }

    resp = *shared_resp;
    return has_error->load() ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}
