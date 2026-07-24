/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuobj_rdma_token_client.h"
#include "common/nixl_log.h"

CuObjRdmaTokenClient::CuObjRdmaTokenClient(CUObjOps_t &ops)
    : inner_(std::make_shared<cuObjClient>(ops, CUOBJ_PROTO_RDMA_DC_V1)) {
    NIXL_INFO << "CuObjRdmaTokenClient initialized (DC transport)";
}

bool
CuObjRdmaTokenClient::isConnected() const {
    return inner_->isConnected();
}

cuObjErr_t
CuObjRdmaTokenClient::cuMemObjGetDescriptor(void *ptr, size_t size) {
    return inner_->cuMemObjGetDescriptor(ptr, size);
}

cuObjErr_t
CuObjRdmaTokenClient::cuMemObjPutDescriptor(void *ptr) {
    return inner_->cuMemObjPutDescriptor(ptr);
}

ssize_t
CuObjRdmaTokenClient::cuObjGet(void *ctx,
                               void *ptr,
                               size_t size,
                               loff_t offset,
                               loff_t buf_offset) {
    return inner_->cuObjGet(ctx, ptr, size, offset, buf_offset);
}

ssize_t
CuObjRdmaTokenClient::cuObjPut(void *ctx,
                               void *ptr,
                               size_t size,
                               loff_t offset,
                               loff_t buf_offset) {
    return inner_->cuObjPut(ctx, ptr, size, offset, buf_offset);
}
