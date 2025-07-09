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

obj_sources = [
    'obj_backend.cpp',
    'obj_backend.h',
    'obj_plugin.cpp',
    'obj_s3_client.cpp',
    'obj_s3_client.h',
]

aws_s3 = dependency('aws-cpp-sdk-s3', static: false, required: false)
if aws_s3.found()
    # By default aws-cpp-sdk sets c++11 compile flag for the whole project
    partial_aws_s3 = aws_s3.partial_dependency(compile_args: false, includes: true, link_args: true, links: true)
    plugin_deps += [partial_aws_s3]
else
    subdir_done()
endif
plugin_deps += [dependency('asio', required: true)]

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
        install_dir: plugin_install_dir)
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "OBJ=' + obj_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

obj_backend_interface = declare_dependency(link_with: obj_backend_lib)
