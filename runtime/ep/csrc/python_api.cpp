// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <nixl_ep_v2/common/compiled.cuh>

#include "elastic/buffer.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NIXL EP: an efficient expert-parallel communication library";

    m.def("is_fabric_handle_supported", &nixl::runtime::ep::elastic::nixl::is_fabric_handle_supported);

    // The integer type of top-k indices
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<nixl::runtime::ep::topk_idx_t>::value);

    // Register elastic buffer APIs
    nixl::runtime::ep::elastic::register_apis(m);
}
