/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Marvell Technology, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/odm_consistency.h"

#include "worker/nixl/nixl_worker_odm.h"

OdmConsistencyContext::OdmConsistencyContext(
    const std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    (void)iov_lists;
    if (xferBenchConfig::backend != XFERBENCH_BACKEND_MARVELL_ODM ||
        xferBenchConfig::op_type != XFERBENCH_OP_WRITE) {
        return;
    }
    active = true;
}

bool
OdmConsistencyContext::fetchWriteBuffer(const xferBenchIOV &iov,
                                        void **addr_out,
                                        bool *allocated_out) {
    *addr_out = nullptr;
    *allocated_out = false;
    if (!active) {
        return false;
    }
    return xferBenchOdm::fetchWriteBufferForConsistency(iov, addr_out, allocated_out);
}
