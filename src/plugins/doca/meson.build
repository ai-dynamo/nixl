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

doca_path = get_option('doca_path')
doca_inc_path = doca_path + '/include'

doca_dep0 = dependency('doca-common')
doca_dep1 = dependency('doca-rdma')
doca_dep2 = dependency('doca-gpunetio')

compile_flags += '-D DOCA_ALLOW_EXPERIMENTAL_API'
#if doca_path != ''
#  doca_lib_path = doca_path + '/lib/x86_64-linux-gnu'
#  doca_inc_path = doca_path + '/include'
#  doca_dep = declare_dependency(
#    #link_args : ['-L' + doca_lib_path, '-libdoca_rdma'], # -libdoca_common -libdoca_gpunetio -libdoca_gpunetio_device
#    include_directories : include_directories(doca_inc_path))
#endif

if 'DOCA' in static_plugins
    doca_backend_lib = static_library('DOCA',
               'doca_backend.cpp', 'doca_backend.h', 'doca_plugin.cpp', #'doca_utils.cpp',
               dependencies: [nixl_infra, serdes_interface, cuda_dep, doca_dep0, doca_dep1, doca_dep2],
               include_directories: nixl_inc_dirs, doca_inc_path,
               install: false,
               cpp_args : compile_flags,
               name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    doca_backend_lib = shared_library('DOCA',
               'doca_backend.cpp', 'doca_backend.h', 'doca_plugin.cpp', #'doca_utils.cpp',
               dependencies: [nixl_infra, serdes_interface, cuda_dep, doca_dep0, doca_dep1, doca_dep2],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: true,
               cpp_args : compile_flags + ['-fPIC'],
               name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
               install_dir: plugin_install_dir)

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "DOCA=' + doca_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

doca_backend_interface = declare_dependency(link_with: doca_backend_lib)
