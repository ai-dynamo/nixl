// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_GPUNETIO_DEVICE_MEMVIEW_H
#define NIXL_GPUNETIO_DEVICE_MEMVIEW_H

#include "device/device_allocator.h"
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
    nixlDeviceMem elements;
    nixlDeviceMem header;
};

struct nixlGpunetioNativeState {
    std::mutex mutex;
    uint64_t cookie = 0;
    std::string peer;
    size_t remoteViews = 0;
    nixlDeviceMem lane;
    nixlDeviceMem context;
    std::unordered_map<nixlMemViewH, nixlGpunetioNativeViewStorage> views;
};

#endif
