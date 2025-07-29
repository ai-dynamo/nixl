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

gds_path = get_option('gds_path')
if gds_path != ''
    gds_lib_path = gds_path + '/lib64'
    gds_inc_path = gds_path + '/include'
    cufile_dep = declare_dependency(
        link_args: ['-L' + gds_lib_path, '-lcufile'],
        include_directories: include_directories(gds_inc_path))
else
    # Create a dummy dependency that does nothing
    cufile_dep = declare_dependency()
    subdir_done()
endif

if 'GDS' in static_plugins
    gds_backend_lib = static_library('GDS',
        'gds_utils.cpp', 'gds_utils.h',
        'gds_backend.cpp', 'gds_backend.h',
        'gds_plugin.cpp',
        dependencies: [nixl_infra, nixl_common_dep, cuda_dep, cufile_dep, file_utils_interface],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        cpp_args: compile_flags,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    gds_backend_lib = shared_library('GDS',
        'gds_utils.cpp', 'gds_utils.h',
        'gds_backend.cpp', 'gds_backend.h',
        'gds_plugin.cpp',
        dependencies: [nixl_infra, nixl_common_dep, cuda_dep, cufile_dep, file_utils_interface],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        cpp_args: ['-fPIC'],
        name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "GDS=' + gds_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

gds_backend_interface = declare_dependency(link_with: gds_backend_lib)
