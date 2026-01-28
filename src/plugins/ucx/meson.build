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

asio_dep = [dependency('asio', required: true)]

ucx_backend_includes = include_directories('.')

ucx_backend_sources = ['config.cpp',
                       'gpu_xfer_req_h.cpp',
                       'mem_list.cpp',
                       'rkey.cpp',
                       'ucx_backend.cpp',
                       'ucx_plugin.cpp',
                       'ucx_utils.cpp']

ucx_backend_dependencies = [asio_dep,
                            cuda_dep,
                            nixl_common_dep,
                            nixl_infra,
                            serdes_interface,
                            thread_dep,
                            ucx_dep]

ucx_backend_include_directories = [nixl_inc_dirs,
                                   utils_inc_dirs]

if 'UCX' in static_plugins
    ucx_backend_lib = static_library('UCX',
               ucx_backend_sources,
               dependencies: ucx_backend_dependencies,
               include_directories: ucx_backend_include_directories,
               install: false,
               name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    ucx_backend_lib = shared_library('UCX',
               ucx_backend_sources,
               dependencies: ucx_backend_dependencies,
               include_directories: ucx_backend_include_directories,
               install: true,
               cpp_args : ['-fPIC'],
               name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
               install_dir: plugin_install_dir,
               build_rpath: get_option('ucx_path') != '' ? get_option('ucx_path') + '/lib' : '',
               install_rpath: get_option('ucx_path') != '' ? '$ORIGIN/..:' + get_option('ucx_path') + '/lib' : '$ORIGIN/..')

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "UCX=' + ucx_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

ucx_backend_interface = declare_dependency(link_with: ucx_backend_lib, include_directories: ucx_backend_includes)
