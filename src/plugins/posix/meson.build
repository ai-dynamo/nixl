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

# Check for liburing dependency
liburing_dep = dependency('liburing', required: true)

if 'POSIX' in static_plugins
    posix_backend_lib = static_library('POSIX',
        'posix_backend.cpp', 'posix_backend.h',
        'posix_plugin.cpp',
        'posix_utils.cpp', 'posix_utils.h',
        dependencies: [nixl_infra, liburing_dep],
        link_args: ['-luring'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        cpp_args: compile_flags,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    posix_backend_lib = shared_library('POSIX',
        'posix_backend.cpp', 'posix_backend.h',
        'posix_plugin.cpp',
        'posix_utils.cpp', 'posix_utils.h',
        dependencies: [nixl_infra, liburing_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        link_args: ['-luring'],
        install: true,
        cpp_args: ['-fPIC'],
        name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
        install_dir: plugin_install_dir)
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "POSIX=' + posix_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

posix_backend_interface = declare_dependency(link_with: posix_backend_lib) 