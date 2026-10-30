# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mooncake_backend_dep = declare_dependency(link_with: mooncake_backend_lib, include_directories: [nixl_inc_dirs, '../../../../src/plugins/mooncake'])

if cuda_dep.found()
    mooncake_test_cuda_deps = [cuda_dep]
    mooncake_test_cpp_args = '-DHAVE_CUDA'
else
    mooncake_test_cuda_deps = []
    mooncake_test_cpp_args = '-UHAVE_CUDA'
endif

mooncake_backend_test = executable('mooncake_backend_test',
        'mooncake_backend_test.cpp',
        dependencies: [nixl_dep, nixl_infra, mooncake_backend_dep, thread_dep] + mooncake_test_cuda_deps,
        include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/mooncake'],
        cpp_args : mooncake_test_cpp_args,
        install: true)
