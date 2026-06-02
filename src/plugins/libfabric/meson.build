# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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

# LibFabric plugin configuration

# Enable libfabric utils layer
libfabric_plugin_deps = [
    nixl_infra,
    nixl_common_dep,
    libfabric_dep,
    serdes_interface,
    libfabric_utils_dep,
    dependency('dl', required: true)
]

compile_flags = ['-DHAVE_LIBFABRIC']

# Add CUDA support if available
if cuda_dep.found()
    libfabric_plugin_deps += [cuda_dep]
    compile_flags += ['-DHAVE_CUDA']
endif

# Build as static or shared library based on configuration
if 'LIBFABRIC' in static_plugins
    libfabric_backend_lib = static_library(
        'LIBFABRIC',
        'libfabric_backend.cpp',
        'libfabric_backend.h',
        'libfabric_plugin.cpp',
        dependencies: libfabric_plugin_deps,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        cpp_args: compile_flags,
        name_prefix: 'libplugin_',
    )
else
    libfabric_backend_lib = shared_library(
        'LIBFABRIC',
        'libfabric_backend.cpp',
        'libfabric_backend.h',
        'libfabric_plugin.cpp',
        dependencies: libfabric_plugin_deps,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        cpp_args: compile_flags + ['-fPIC'],
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
    )

    # Add to plugin list for debug builds
    if get_option('buildtype') == 'debug'
        run_command(
            'sh',
            '-c',
            'echo "LIBFABRIC=' + libfabric_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true,
        )
    endif
endif

# Declare dependency interface
libfabric_backend_interface = declare_dependency(link_with: libfabric_backend_lib)
