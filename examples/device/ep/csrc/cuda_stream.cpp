/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_stream.hpp"

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/version.h>

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 10)
#error "nixl_ep requires PyTorch >=2.10 for torch_set_current_cuda_stream"
#endif

namespace nixl_ep::cuda_stream {
cudaStream_t
get_from_pool() {
    void *stream;
    TORCH_ERROR_CODE_CHECK(torch_get_cuda_stream_from_pool(true, -1, &stream));
    return static_cast<cudaStream_t>(stream);
}

cudaStream_t
get_current() {
    void *stream;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(-1, &stream));
    return static_cast<cudaStream_t>(stream);
}

void
set_current(cudaStream_t stream) {
    const int32_t device_index = torch::stable::accelerator::getCurrentDeviceIndex();
    TORCH_ERROR_CODE_CHECK(torch_set_current_cuda_stream(stream, device_index));
}
} // namespace nixl_ep::cuda_stream
