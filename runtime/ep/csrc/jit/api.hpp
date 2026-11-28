// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "compiler.hpp"
#include "include_parser.hpp"

namespace nixl::runtime::ep::jit {

static void init(const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
    Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
    IncludeParser::prepare_init(library_root_path);
}

static void register_apis(pybind11::module_& m) {
    m.def("init_jit", &init);
}

}  // namespace nixl::runtime::ep::jit
