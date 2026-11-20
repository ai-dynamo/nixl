// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_GPUNETIO_DEVICE_MEMVIEW_H
#define NIXL_GPUNETIO_DEVICE_MEMVIEW_H

#include "device/device_ops.h"
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

struct nixlGpunetioNativeViewStorage {
    std::vector<std::shared_ptr<nixl::doca::verbs::mr>> pins;
    bool remote = false;
    uint32_t channel = 0;
    nixl::deviceMem elements;
    nixl::deviceMem header;
    nixl::deviceMem wrapper;
};

struct nixlGpunetioNativeState {
    std::mutex mutex;
    uint64_t cookie = 0;
    std::string peer;
    size_t remoteViews = 0;
    uint32_t laneCount = 1;
    std::vector<doca_gpu_dev_verbs_qp *> boundQps;
    nixl::deviceOps *ops = nullptr;
    nixl::deviceMem lane;
    nixl::deviceMem context;
    std::unordered_map<nixlMemViewH, nixlGpunetioNativeViewStorage> views;
};

#endif
