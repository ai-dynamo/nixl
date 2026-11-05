/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
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

#pragma once

#include <torch/python.h>
#include <nixl_ep_v2/common/exception.cuh>

namespace nixl_ep_v2::elastic {

static at::cuda::CUDAStream get_global_comm_stream() {
    static std::optional<at::cuda::CUDAStream> comm_stream = std::nullopt;
    if (not comm_stream.has_value())
        comm_stream = at::cuda::getStreamFromPool(true);
    return comm_stream.value();
}

template <int kNumDims>
static auto get_shape(const torch::Tensor& t) {
    EP_HOST_ASSERT(t.dim() == kNumDims);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<kNumDims>());
}

template <typename dtype_t = void>
static dtype_t* get_data_ptr(const std::optional<torch::Tensor>& t) {
    return t.has_value() ? t->data_ptr<dtype_t>() : nullptr;
}

}  // nixl_ep_v2::elastic
