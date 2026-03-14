# SPDX-FileCopyrightText: Copyright (c) 2026 Dell Technologies Inc. All rights reserved.
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

# libblkio plugin
# Reuse libblkio_dep from root meson.build to ensure consistent detection
# The root meson.build includes fallback find_library() detection

blkio_sources = [
    'libblkio_backend.h',
    'libblkio_backend.cpp',
    'libblkio_plugin.cpp',
]
compile_defs = []
if 'LIBBLKIO' in static_plugins
    libblkio_backend_lib = static_library(
        'LIBBLKIO',
        blkio_sources,
        dependencies: [nixl_infra, absl_log_dep, libblkio_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        cpp_args: compile_defs + compile_flags,
        name_prefix: 'libplugin_',
    )
else
    libblkio_backend_lib = shared_library(
        'LIBBLKIO',
        blkio_sources,
        dependencies: [nixl_infra, absl_log_dep, libblkio_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        cpp_args: compile_defs + ['-fPIC'],
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..',
        version: meson.project_version(),
    )
    if get_option('buildtype') == 'debug'
        run_command(
            'sh', '-c',
            'echo "LIBBLKIO=' + libblkio_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true,
        )
    endif
endif
message('Building libblkio plugin')
libblkio_backend_interface = declare_dependency(link_with: libblkio_backend_lib)
