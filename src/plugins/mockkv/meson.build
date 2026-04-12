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

# MOCKKV plugin sources
mockkv_sources = [
    'mockkv_backend.cpp',
    'mockkv_backend.h',
    'mockkv_plugin.cpp',
]

# MOCKKV has no external dependencies - it's a simple in-memory store
# Just needs NIXL infrastructure and common utilities
mockkv_plugin_deps = [nixl_infra, nixl_common_dep]

# Define compile_flags if not already defined
if not is_variable('compile_flags')
    compile_flags = []
endif

if 'MOCKKV' in static_plugins
    # Build as static library (linked into libnixl.so)
    mockkv_backend_lib = static_library(
        'MOCKKV',
        mockkv_sources,
        dependencies: mockkv_plugin_deps,
        cpp_args: compile_defs + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    # Build as shared library (loaded at runtime)
    mockkv_backend_lib = shared_library(
        'MOCKKV',
        mockkv_sources,
        dependencies: mockkv_plugin_deps,
        cpp_args: compile_defs + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "MOCKKV=' + mockkv_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

mockkv_backend_interface = declare_dependency(link_with: mockkv_backend_lib)
