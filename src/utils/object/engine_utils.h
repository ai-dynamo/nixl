/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_UTILS_OBJECT_ENGINE_UTILS_H
#define OBJ_PLUGIN_UTILS_OBJECT_ENGINE_UTILS_H

#include "common/nixl_log.h"
#include "nixl_types.h"
#include <algorithm>
#include <thread>

inline std::size_t
getNumThreads(nixl_b_params_t *custom_params) {
    return custom_params && custom_params->count("num_threads") > 0 ?
        std::stoul(custom_params->at("num_threads")) :
        std::max(1u, std::thread::hardware_concurrency() / 2);
}

inline size_t
getCrtMinLimit(nixl_b_params_t *custom_params) {
    if (!custom_params) return 0;

    auto it = custom_params->find("crtMinLimit");
    if (it != custom_params->end()) {
        try {
            return std::stoull(it->second);
        }
        catch (const std::exception &e) {
            NIXL_WARN << "Invalid crtMinLimit value: " << it->second
                      << ", using default (CRT disabled)";
            return 0;
        }
    }
    return 0; // Disabled by default
}

inline bool
isAcceleratedRequested(nixl_b_params_t *custom_params) {
    if (!custom_params) return false;
    auto accel_it = custom_params->find("accelerated");
    return accel_it != custom_params->end() && accel_it->second == "true";
}

#endif // OBJ_PLUGIN_UTILS_OBJECT_ENGINE_UTILS_H
