/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client.h"
#include "common/nixl_log.h"

awsS3AccelClient::awsS3AccelClient(nixl_b_params_t *custom_params,
                                   std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    // Opt out of the base client's standard S3-over-RDMA fast path: vendor
    // accelerated engines (e.g. Dell) manage RDMA themselves via their own
    // cuObjClient, so the base must not also construct the shared one.
    : awsS3Client(custom_params, executor, /*enable_rdma_fast_path=*/false) {
    // Base class already initializes the S3 client, bucket name, and credentials.
    // Derived vendor clients can add their specific initialization here.
    NIXL_DEBUG << "S3 Accelerated client initialized";
}
