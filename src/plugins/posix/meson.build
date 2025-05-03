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

# If pkg-config fails, try finding it as a system library
if not liburing_dep.found()
    liburing_dep = compiler.find_library('uring', required: false)
endif

have_liburing = liburing_dep.found()

if have_liburing
    compile_defs += ['-DHAVE_LIBURING']
    message('liburing found, adding io_uring support')
else
    # Create a not found dependency to avoid build errors
    liburing_dep = dependency('', required: false)
    message('liburing not found, building with AIO support only')
endif

# Get Abseil dependencies
absl_log_dep = dependency('absl_log', required: true)

# Add pthread dependency
thread_dep = dependency('threads')

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
plugin_deps = [nixl_infra, nixl_common_dep, absl_log_dep, thread_dep, libaio_dep]
plugin_link_args = ['-laio']

if have_liburing
    plugin_deps += [liburing_dep]
    plugin_link_args += ['-luring']
endif

# Build the plugin using project-wide include directories
posix_backend_lib = shared_library('POSIX',
              posix_sources,
              dependencies: plugin_deps,
              link_args: plugin_link_args,
              cpp_args: compile_defs,
              name_prefix: 'libplugin_',
              include_directories: [nixl_inc_dirs, utils_inc_dirs],
              install: true,
              install_dir: plugin_install_dir)

if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "POSIX=' + posix_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
endif
