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

# WQSKV plugin sources
wqskv_sources = [
    'wqskv_backend.cpp',
    'wqskv_backend.h',
    'wqskv_helpers.cpp',
    'wqskv_helpers.h',
    'wqskv_plugin.cpp',
]

# wqskv_lib_found, wqskv_inc_path, and jsoncpp_dep are set by the parent
# src/plugins/meson.build before calling subdir('wqskv'). Using the find_library
# result directly lets meson's dependency tracker handle RPATH correctly.
wqskv_inc_dirs_list = []
if wqskv_inc_path != ''
    wqskv_inc_dirs_list = [include_directories(wqskv_inc_path, is_system: true)]
endif
wqskv_inc_dep = declare_dependency(include_directories : wqskv_inc_dirs_list)

wqskv_plugin_deps = [
    nixl_infra,
    nixl_common_dep,
    wqskv_lib_found,
    wqskv_inc_dep,
    jsoncpp_dep,
    thread_dep,
]

if not is_variable('compile_flags')
    compile_flags = []
endif

if 'WQSKV' in static_plugins
    wqskv_backend_lib = static_library(
        'WQSKV',
        wqskv_sources,
        dependencies: wqskv_plugin_deps,
        cpp_args: compile_defs + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')
else
    wqskv_backend_lib = shared_library(
        'WQSKV',
        wqskv_sources,
        dependencies: wqskv_plugin_deps,
        cpp_args: compile_defs + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "WQSKV=' + wqskv_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

wqskv_backend_interface = declare_dependency(link_with: wqskv_backend_lib)
