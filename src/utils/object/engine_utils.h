/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_UTILS_OBJECT_ENGINE_UTILS_H
#define NIXL_SRC_UTILS_OBJECT_ENGINE_UTILS_H

#include "common/backend.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include <algorithm>
#include <thread>

[[nodiscard]] inline std::size_t
getNumThreads(nixl_b_params_t *custom_params) {
    const std::size_t fallback = std::max(1u, std::thread::hardware_concurrency() / 2);
    return nixl::getBackendParamDefaulted(custom_params, "num_threads", fallback);
}

[[nodiscard]] inline size_t
getCrtMinLimit(nixl_b_params_t *custom_params) {
    return nixl::getBackendParamDefaulted(custom_params, "crtMinLimit", size_t(0));
}

[[nodiscard]] inline bool
isAcceleratedRequested(nixl_b_params_t *custom_params) {
    return nixl::getBackendParamDefaulted(custom_params, "accelerated", false);
}

[[nodiscard]] inline bool
isDellOBSRequested(nixl_b_params_t *custom_params) {
    return isAcceleratedRequested(custom_params) &&
        (nixl::getBackendParamOptional<std::string>(custom_params, "type") == "dell");
}


#endif
