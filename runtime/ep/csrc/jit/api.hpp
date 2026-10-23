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

#include "compiler.hpp"
#include "include_parser.hpp"
#include "kernel_runtime.hpp"

namespace nixl_ep_v2::jit {

static void init(const std::string& library_root_path,
                 const std::string& cuda_home_path_by_python, const std::string& nixl_root_path_by_python, const std::string& ucx_root_path_by_python) {
    Compiler::prepare_init(library_root_path, cuda_home_path_by_python, nixl_root_path_by_python, ucx_root_path_by_python);
    KernelRuntime::prepare_init(cuda_home_path_by_python);
    IncludeParser::prepare_init(library_root_path);
}

static void register_apis(pybind11::module_& m) {
    m.def("init_jit", &init);
}

}  // namespace nixl_ep_v2::jit
