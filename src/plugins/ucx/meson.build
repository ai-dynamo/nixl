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

ucx_utils_dep = declare_dependency(link_with: ucx_utils_lib, include_directories: utils_inc_dirs )

compile_flags = []
if cuda_dep.found()
    compile_flags = [ '-DHAVE_CUDA' ]
endif

if 'UCX' in static_plugins
    ucx_backend_lib = static_library('UCX',
               'ucx_backend.cpp', 'ucx_backend.h', 'ucx_plugin.cpp',
               dependencies: [nixl_infra, ucx_utils_dep, serdes_interface, cuda_dep, ucx_dep, thread_dep],
               include_directories: nixl_inc_dirs,
               install: false,
               cpp_args : compile_flags,
               name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    ucx_backend_lib = shared_library('UCX',
               'ucx_backend.cpp', 'ucx_backend.h', 'ucx_plugin.cpp',
               dependencies: [nixl_infra, ucx_utils_dep, serdes_interface, cuda_dep, ucx_dep, thread_dep],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: true,
               cpp_args : compile_flags + ['-fPIC'],
               name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
               install_dir: plugin_install_dir)

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "UCX=' + ucx_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

ucx_backend_interface = declare_dependency(link_with: ucx_backend_lib)
