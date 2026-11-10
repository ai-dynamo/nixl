// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <torch/python.h>
#include <nixl_ep_v2/common/exception.cuh>

namespace nixl_ep_v2::elastic {

static at::cuda::CUDAStream get_global_comm_stream() {
    static const auto comm_stream = at::cuda::getStreamFromPool(true);
    return comm_stream;
}

template <int kNumDims>
static auto get_shape(const torch::Tensor& t) {
    EP_HOST_ASSERT(t.dim() == kNumDims);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<kNumDims>());
}

}  // nixl_ep_v2::elastic
