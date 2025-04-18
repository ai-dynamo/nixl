/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __BACKEND_PLUGIN_DOCA_COMMON_H
#define __BACKEND_PLUGIN_DOCA_COMMON_H

#include <nixl_types.h>
#include <doca_rdma.h>

#define DOCA_XFER_REQ_SIZE 512

struct docaXferReqGpu {
    uintptr_t larr[DOCA_XFER_REQ_SIZE];
    uintptr_t rarr[DOCA_XFER_REQ_SIZE];
    size_t size[DOCA_XFER_REQ_SIZE];
    uint16_t num;
    nixl_xfer_op_t backendOp;           /* Needed only in case of GPU device transfer */
    struct doca_gpu_dev_rdma *rdma_gpu; /* Needed only in case of GPU device transfer */
};

#endif