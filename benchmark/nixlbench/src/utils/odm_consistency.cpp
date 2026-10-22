/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Marvell Technology, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
