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

# Check for libaio dependency (required)
compiler = meson.get_compiler('cpp')
libaio_dep = compiler.find_library('aio', required: false)

# If libaio is not found, skip building the POSIX plugin entirely
if not libaio_dep.found()
    message('libaio not found, POSIX plugin will not be built')
    subdir_done()
endif

# Configure compile flags and set AIO as default
compile_defs = ['-DHAVE_LIBAIO']
message('libaio found, building POSIX plugin')

# Try to find liburing (optional) - first try pkg-config
liburing_dep = dependency('liburing', required: false)

have_liburing = liburing_dep.found()
if have_liburing
    compile_defs += ['-DHAVE_LIBURING']
    message('liburing found, adding io_uring support')
else
    message('liburing not found, building with AIO support only')
endif

# Define base source files - conditionally include uring_queue.cpp
posix_sources = [
    'posix_backend.cpp',
    'posix_backend.h',
    'posix_plugin.cpp',
    'aio_queue.cpp'  # Always include AIO source since it's required
]

# Add io_uring source if available
if have_liburing
    posix_sources += ['uring_queue.cpp']
endif

# Prepare dependencies list - only include liburing if found
plugin_deps = [nixl_infra, nixl_common_dep, libaio_dep, rt_dep]
plugin_link_args = ['-laio']

if have_liburing
    plugin_deps += [liburing_dep]
    plugin_link_args += ['-luring']
endif

if 'POSIX' in static_plugins
    posix_backend_lib = static_library('POSIX',
        posix_sources,
        dependencies: plugin_deps,
        link_args: plugin_link_args,
        cpp_args: compile_defs + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    posix_backend_lib = shared_library('POSIX',
        posix_sources,
        dependencies: plugin_deps,
        link_args: plugin_link_args,
        cpp_args: compile_defs + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
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
