# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ucx_utils_dep = declare_dependency(link_with: ucx_utils_lib, include_directories: inc_dir)
nixl_dep = declare_dependency(link_with: nixl_infra, include_directories: inc_dir)

compile_flags = []
if cuda_dep.found()
    compile_flags = [ '-DHAVE_CUDA' ]
endif

if 'UCX' in static_plugins
    ucx_backend_lib = static_library('UCX',
               'ucx_backend.cpp', 'ucx_backend.h', 'ucx_plugin.cpp',
               dependencies: [ucx_utils_dep, nixl_dep, serdes_interface, cuda_dep, ucx_dep],
               include_directories: inc_dir,
               install: true,
               cpp_args : compile_flags,
               name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
               #install_dir: plugin_install_dir)
else
    ucx_backend_lib = shared_library('UCX',
               'ucx_backend.cpp', 'ucx_backend.h', 'ucx_plugin.cpp',
               dependencies: [ucx_utils_dep, nixl_dep, serdes_interface, cuda_dep, ucx_dep],
               include_directories: inc_dir,
               install: true,
               cpp_args : compile_flags + ['-fPIC'],
               name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
               install_dir: plugin_install_dir)
    custom_target('copy_ucx_plugin',
                 input: ucx_backend_lib,
                 output: 'copied_ucx_plugin',
                 command: ['cp', '@INPUT@', plugin_build_dir / 'libplugin_UCX.so'],
                 build_by_default: true)
endif

ucx_backend_interface = declare_dependency(link_with: ucx_backend_lib)
