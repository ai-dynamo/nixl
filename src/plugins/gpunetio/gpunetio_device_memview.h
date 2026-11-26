// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_DEVICE_MEMVIEW_H
#define NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_DEVICE_MEMVIEW_H

#include <cuda_runtime.h>
#include "device/device_memview.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace nixl::doca::verbs {
class mr;
}
struct doca_gpu_dev_verbs_qp;

struct nixlGpunetioNativeStream {
    cudaStream_t stream = nullptr;

    ~nixlGpunetioNativeStream() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
};

struct nixlGpunetioNativeDeleter {
    std::shared_ptr<nixlGpunetioNativeStream> stream;

    void
    operator()(void *ptr) const noexcept {
        if (ptr) {
            cudaFreeAsync(ptr, stream->stream);
        }
    }
};

using nixlGpunetioNativeMem = std::unique_ptr<void, nixlGpunetioNativeDeleter>;

struct nixlGpunetioNativeViewStorage {
    std::vector<std::shared_ptr<nixl::doca::verbs::mr>> pins;
    bool remote = false;
    nixlGpunetioNativeMem elements;
    nixlGpunetioNativeMem header;
    nixlGpunetioNativeMem wrapper;
};

struct nixlGpunetioNativeState {
    std::mutex mutex;
    uint64_t cookie = 0;
    std::string peer;
    size_t remoteViews = 0;
    uint32_t laneCount = 1;
    std::vector<doca_gpu_dev_verbs_qp *> boundQps;
    // NIC-only fetch-add results; deregister before releasing the host storage.
    std::vector<uint64_t> atomicResults;
    std::shared_ptr<nixl::doca::verbs::mr> atomicMr;
    std::shared_ptr<nixlGpunetioNativeStream> stream;
    bool snapshotFailed = false;
    bool
    lanesIdle() const;
    nixlGpunetioNativeMem lane;
    nixlGpunetioNativeMem context;
    std::unordered_map<nixlMemViewH, nixlGpunetioNativeViewStorage> views;
};

#endif
