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

cpp = meson.get_compiler('cpp')

uccl_lib = cpp.find_library('uccl_p2p', dirs: ['/usr/local/lib'], required: false)
uccl_rdma_lib = cpp.find_library('rdma_plugin', dirs: ['/usr/local/lib'], required: false)

if not uccl_lib.found() or not uccl_rdma_lib.found()
    message('UCCL P2P libraries not found, skipping uccl_gtest build')
    subdir_done()
endif

python_dep = dependency('python3', required: false)
pybind11_dep = dependency('pybind11', required: false)

if not python_dep.found() or not pybind11_dep.found()
    message('Python dependencies not found, skipping uccl_gtest build')
    subdir_done()
endif

uccl_gtest_cpp_flags = []
uccl_gtest_cpp_flags += '-DBUILD_DIR="' + meson.project_build_root() + '"'

if cuda_dep.found()
    uccl_gtest_cuda_deps = [cuda_dep]
    uccl_gtest_cpp_flags += '-DHAVE_CUDA'
else
    uccl_gtest_cuda_deps = []
    uccl_gtest_cpp_flags += '-UHAVE_CUDA'
endif

uccl_test_exe = executable('uccl_gtest',
    sources : ['uccl_test.cpp', '../../main.cpp', '../../common.cpp'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, gtest_inc_dirs, '.'],
    cpp_args : uccl_gtest_cpp_flags,
    dependencies : [
        nixl_dep,
        nixl_infra,
        nixl_common_deps,
        thread_dep,
        python_dep,
        pybind11_dep,
        uccl_rdma_lib,
        uccl_lib,
        gtest_dep,
        gmock_dep,
        absl_strings_dep,
        absl_time_dep
    ] + uccl_gtest_cuda_deps,
    link_with: [nixl_build_lib],
    install : true
)

test('uccl_gtest', uccl_test_exe)
