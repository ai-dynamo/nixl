/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "obj_engine_registry.h"

namespace {

// Register the standard-S3 engine under "s3" (the preferred, protocol-compliant
// S3-over-RDMA engine). obj_backend normalizes a missing `type` to "s3" before
// lookup, so `accelerated=true` with no type also resolves here. (We deliberately
// avoid registering under "" because the cuobj-gated s3_accel base also claims
// "", and the registry keeps the first-inserted entry — static-init order across
// translation units is unspecified, so a "" registration here would be a
// non-deterministic collision. Routing through "s3" is collision-free and
// deterministic in every build.)
//
// Built unconditionally; the RDMA transport itself is guarded by
// HAVE_CUOBJ_CLIENT in the standard client, which throws at construction if RDMA
// is requested but unavailable.
objAccelEngineRegistrar reg_generic(
    "s3",
    [](const nixlBackendInitParams *p) { return std::make_unique<GenericObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p,
       std::shared_ptr<iS3Client> s3,
       std::shared_ptr<iS3Client> s3_crt) {
        return std::make_unique<GenericObjEngineImpl>(p, std::move(s3), std::move(s3_crt));
    });

} // namespace
