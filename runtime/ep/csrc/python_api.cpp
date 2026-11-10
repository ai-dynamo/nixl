// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <nixl_ep_v2/common/compiled.cuh>

#include "jit/api.hpp"
#include "elastic/buffer.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NIXL EP v2: an efficient expert-parallel communication library";

    m.def("get_adaptive_num_launch_sms", []() { return nixl_ep_v2::elastic::topology::get_num_launch_sms(); });
    m.attr("min_num_active_ranks") = nixl_ep_v2::elastic::topology::kMinNumActiveRanks;
    m.def("is_fabric_handle_supported", &nixl_ep_v2::elastic::nixl::is_fabric_handle_supported);

    // The integer type of top-k indices
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<nixl_ep_v2::topk_idx_t>::value);

    // JIT API
    nixl_ep_v2::jit::register_apis(m);

    // Register elastic buffer (NIXL EP v2) APIs
    nixl_ep_v2::elastic::register_apis(m);
}
