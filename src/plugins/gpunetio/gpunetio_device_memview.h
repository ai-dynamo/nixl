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

struct nixlGpunetioNativeViewStorage {
    std::vector<std::shared_ptr<nixl::doca::verbs::mr>> pins;
    bool remote = false;
    nixl::deviceMem elements;
    nixl::deviceMem header;
    nixl::deviceMem wrapper;
};

struct nixlGpunetioNativeState {
    std::mutex mutex;
    uint64_t cookie = 0;
    nixl::deviceOps *ops = nullptr;
    std::string peer;
    size_t remoteViews = 0;
    nixl::deviceMem lane;
    nixl::deviceMem context;
    std::unordered_map<nixlMemViewH, nixlGpunetioNativeViewStorage> views;
};

#endif
