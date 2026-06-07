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
getAccelType(const nixl_b_params_t *custom_params) {
    return nixl::getBackendParamDefaulted(custom_params, "type", std::string());
}

// Standard, protocol-compliant S3-over-RDMA path: `accelerated=true` with no
// `type` (or `type=s3`). This is the preferred engine because it complies with
// the published `x-amz-rdma-*` S3-over-RDMA protocol and requires no per-vendor
// code, unlike the vendor-specific engines selected by an explicit `type` (e.g.
// `type=dell`) for servers that use vendor-specific RDMA headers.
//
// RDMA is asserted rather than auto-probed because the fallback handshake depends
// on the server returning `x-amz-rdma-reply: 501` (or omitting the reply header)
// when it cannot honor RDMA. A server that instead *silently ignores* the
// `x-amz-rdma-token` header would accept our body-less PUT as a 0-byte object —
// a chicken-and-egg the client cannot safely resolve by probing. Until server
// implementations reliably signal 501, the caller must assert RDMA support via
// `accelerated=true`; on a decline/failure the transfer errors rather than
// silently falling back. (See the fallback note in s3/client.cpp for the future
// path.)
[[nodiscard]] inline bool
isGenericAccelRequested(nixl_b_params_t *custom_params) {
    return isAcceleratedRequested(custom_params) &&
        (getAccelType(custom_params).empty() || getAccelType(custom_params) == "s3");
}

#endif // NIXL_SRC_UTILS_OBJECT_ENGINE_UTILS_H
