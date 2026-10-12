/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "client.h"
#include "object/engine_utils.h"
#include "object/rdma/rdma.h"
#include "obj_engine_registry.h"
#include "common/nixl_log.h"

namespace {

#ifdef HAVE_CUDA
/**
 * @brief Makes a buffer's CUDA device current for the lifetime of the object.
 *
 * cuObject registers device memory in the calling thread's current CUDA context.
 * A thread that never selected a device has none, so the buffer's device is
 * selected for the registration. The caller's context, or the absence of one, is
 * restored afterwards. If the caller's context cannot be read, the device is not
 * selected, so the caller's context is never lost.
 */
class scopedCudaDevice {
public:
    explicit scopedCudaDevice(int device) {
        const CUresult rc = cuCtxGetCurrent(&saved_);
        if (rc != CUDA_SUCCESS) {
            error_ = "cuCtxGetCurrent failed: " + std::to_string(rc);
            return;
        }
        restore_ = true;
        const cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            error_ =
                "cudaSetDevice(" + std::to_string(device) + ") failed: " + cudaGetErrorString(err);
        }
    }

    ~scopedCudaDevice() {
        if (restore_ && cuCtxSetCurrent(saved_) != CUDA_SUCCESS) {
            NIXL_WARN << "Failed to restore the caller's CUDA context";
        }
    }

    scopedCudaDevice(const scopedCudaDevice &) = delete;
    scopedCudaDevice &
    operator=(const scopedCudaDevice &) = delete;

    /**
     * @brief Why the device could not be selected.
     * @return An empty string on success, otherwise the failure.
     */
    [[nodiscard]] const std::string &
    error() const {
        return error_;
    }

private:
    CUcontext saved_ = nullptr;
    bool restore_ = false;
    std::string error_;
};
#endif

// Registered-memory metadata for the accel engine. It records what to clean up
// on deregister: an OBJ_SEG key mapping, or a DRAM/VRAM buffer pinned for RDMA.
class accelObjMetadata : public nixlBackendMD {
public:
    accelObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(std::move(obj_key)) {}

    accelObjMetadata(nixl_mem_t nixl_mem, uintptr_t addr)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          localAddr(addr),
          rdmaRegistered(true) {}

    ~accelObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId = 0;
    std::string objKey;
    uintptr_t localAddr = 0;
    bool rdmaRegistered = false;
};

// Register the standard-S3 engine as the default accelerated engine; the
// registry resolves a missing `type` to it, so `accelerated=true` with no type
// lands here too.
objAccelEngineRegistrar reg_s3_accel(
    default_accel_engine_type,
    [](const nixlBackendInitParams *p) { return std::make_unique<S3AccelObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p, std::shared_ptr<iS3Client> s3, std::shared_ptr<iS3Client>) {
        return std::make_unique<S3AccelObjEngineImpl>(p, std::move(s3));
    });

} // namespace

S3AccelObjEngineImpl::S3AccelObjEngineImpl(const nixlBackendInitParams *init_params)
    : DefaultObjEngineImpl(init_params) {
    s3Client_ = std::make_shared<awsS3AccelClient>(init_params->customParams, executor_);
    NIXL_INFO << "Object storage backend initialized with S3 Accel client";
}

S3AccelObjEngineImpl::S3AccelObjEngineImpl(const nixlBackendInitParams *init_params,
                                           std::shared_ptr<iS3Client> s3_client)
    : DefaultObjEngineImpl(init_params, s3_client, nullptr) {
    if (!s3_client) {
        s3Client_ = std::make_shared<awsS3AccelClient>(init_params->customParams, executor_);
    }
    NIXL_INFO << "Object storage backend initialized with S3 Accel client (injected)";
}

iS3Client *
S3AccelObjEngineImpl::getClient() const {
    return s3Client_.get();
}

bool
S3AccelObjEngineImpl::rdmaEngReady() const {
    const auto *client = dynamic_cast<const awsS3AccelClient *>(s3Client_.get());
    return client != nullptr && client->supportsRdma();
}

nixl_mem_list_t
S3AccelObjEngineImpl::getSupportedMems() const {
    // VRAM_SEG (GPU-direct) is advertised only when the RDMA fast path is ready,
    // so a GPU pointer can never reach the HTTP path.
    if (rdmaEngReady()) {
        return {DRAM_SEG, OBJ_SEG, VRAM_SEG};
    }
    return {DRAM_SEG, OBJ_SEG};
}

nixl_status_t
S3AccelObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                  const nixl_mem_t &nixl_mem,
                                  nixlBackendMD *&out) {
    const nixl_mem_list_t supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (nixl_mem == OBJ_SEG) {
        auto obj_md = std::make_unique<accelObjMetadata>(
            nixl_mem, mem.devId, mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo);
        devIdToObjKey_[mem.devId] = obj_md->objKey;
        out = obj_md.release();
        return NIXL_SUCCESS;
    }

    // DRAM_SEG / VRAM_SEG: pin the buffer so the client can mint a cuObject token
    // for it. The accelerated path is RDMA-only (no HTTP fallback), so when the
    // fast path is ready a registration failure is a hard error rather than a
    // silent success that fails later at transfer time.
    if (rdmaEngReady()) {
#ifdef HAVE_CUDA
        std::optional<scopedCudaDevice> device;
        if (nixl_mem == VRAM_SEG) {
            device.emplace(static_cast<int>(mem.devId));
            if (!device->error().empty()) {
                NIXL_ERROR << "Cannot register VRAM on device " << mem.devId << ": "
                           << device->error();
                return NIXL_ERR_BACKEND;
            }
        }
#endif
        auto *rdma = nixl_obj_rdma::SharedCuObjClient::instance();
        if (!rdma || !rdma->registerBuffer(reinterpret_cast<void *>(mem.addr), mem.len)) {
            NIXL_ERROR << "RDMA buffer registration failed; the accelerated path has no HTTP "
                          "fallback";
            return NIXL_ERR_BACKEND;
        }
        out = std::make_unique<accelObjMetadata>(nixl_mem, mem.addr).release();
        return NIXL_SUCCESS;
    }

    // Not RDMA-ready (e.g. an injected non-accel client in tests): VRAM is never
    // advertised, so this is DRAM, which needs no pinning.
    out = nullptr;
    return NIXL_SUCCESS;
}

nixl_status_t
S3AccelObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    auto *obj_md = static_cast<accelObjMetadata *>(meta);
    if (obj_md) {
        std::unique_ptr<accelObjMetadata> obj_md_ptr(obj_md);
        if (obj_md->nixlMem == OBJ_SEG) {
            devIdToObjKey_.erase(obj_md->devId);
        } else if (obj_md->rdmaRegistered) {
            if (auto *rdma = nixl_obj_rdma::SharedCuObjClient::instance()) {
                rdma->deregisterBuffer(reinterpret_cast<void *>(obj_md->localAddr));
            }
        }
    }
    return NIXL_SUCCESS;
}
