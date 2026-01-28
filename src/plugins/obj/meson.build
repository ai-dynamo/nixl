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

obj_sources = [
    'obj_backend.cpp',
    'obj_backend.h',
    'obj_plugin.cpp',
    '../../utils/object/engine_utils.h',
    's3/client.cpp',
    's3/client.h',
    's3/engine_impl.cpp',
    's3/engine_impl.h',
    's3_crt/engine_impl.cpp',
    's3_crt/engine_impl.h',
    's3_crt/client.cpp',
    's3_crt/client.h',
]

# Check if obj_utils library is available
if not is_variable('obj_utils_interface')
    warning('obj_utils library not available, skipping OBJ plugin build')
    subdir_done()
endif

plugin_deps += [obj_utils_interface, dependency('asio', required: true)]

if cuobj_dep.found()
    message('Found CUObjClient Library. Enabling S3 Accelerated engines')
    obj_sources += [
        's3_accel/client.cpp',
        's3_accel/client.h',
        's3_accel/engine_impl.cpp',
        's3_accel/engine_impl.h',
    ]
    plugin_deps += [ cuobj_dep ]
else
    message ('Could not find CUObjClient Library. Skipping S3 Accelerated engines')
endif

if 'OBJ' in static_plugins
    obj_backend_lib = static_library(
        'OBJ',
        obj_sources,
        dependencies: plugin_deps,
        cpp_args: compile_defs + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    obj_backend_lib = shared_library(
        'OBJ',
        obj_sources,
        dependencies: plugin_deps,
        cpp_args: compile_defs + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "OBJ=' + obj_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

obj_backend_interface = declare_dependency(link_with: obj_backend_lib)
