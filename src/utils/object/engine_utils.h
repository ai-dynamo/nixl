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
#include <string>
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

[[nodiscard]] inline std::string
getAccelType(nixl_b_params_t *custom_params) {
    return nixl::getBackendParamDefaulted(custom_params, "type", std::string());
}

// Standard, protocol-compliant S3-over-RDMA path: `accelerated=true` with no
// `type` (or `type=s3`). This is the vendor-neutral engine that speaks the
// published `x-amz-rdma-*` protocol and needs no per-vendor code, unlike a
// vendor engine selected by an explicit `type`.
//
// RDMA is asserted (not auto-probed): a server that silently ignores the
// `x-amz-rdma-token` would accept a body-less PUT as a 0-byte object, so the
// caller must opt in; on a decline/failure the transfer errors rather than
// silently falling back to HTTP.
[[nodiscard]] inline bool
isGenericAccelRequested(nixl_b_params_t *custom_params) {
    const std::string type = getAccelType(custom_params);
    return isAcceleratedRequested(custom_params) && (type.empty() || type == "s3");
}

#endif
