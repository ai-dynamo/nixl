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

cc = meson.get_compiler('cpp')
gds_path = get_option('gds_path')
if gds_path != ''
    gds_lib_path = gds_path + '/lib64'
    gds_inc_path = gds_path + '/include'
    cuobjclient_lib_file = 'cuobjclient'
    cuobjclient_lib_found = cc.find_library(cuobjclient_lib_file, dirs: [gds_lib_path], required: false)

    if not cuobjclient_lib_found.found()
        cuobj_dep = declare_dependency()
        warning('CUDA CuObject Client library not found, skipping OBS_OBJ plugin build')
        subdir_done()
    endif

    cuobj_dep = declare_dependency(
        link_args: ['-L' + gds_lib_path, '-lcufile', '-lcuobjclient', '-lcudart'],
        include_directories: include_directories(gds_inc_path))
else
    # Create a dummy dependency that does nothing
    cuobj_dep = declare_dependency()
    warning('CUDA dependencies not found, skipping OBS_OBJ plugin build')
    subdir_done()
endif

obs_obj_sources = [
    'obs_obj_backend.cpp',
    'obs_obj_backend.h',
    'obs_obj_plugin.cpp',
    'obs_obj_s3_client.cpp',
    'obs_obj_s3_client.h',
]

plugin_deps = [nixl_infra, nixl_common_dep, s3_client_interface, cuda_dep, cuobj_dep]

aws_s3 = dependency('aws-cpp-sdk-s3', static: false, required: false)
aws_core = dependency('aws-cpp-sdk-core', required: false, static: false)
if aws_s3.found() and aws_core.found()
    # By default aws-cpp-sdk sets c++11 compile flag for the whole project
    partial_aws_s3 = aws_s3.partial_dependency(compile_args: false, includes: true, link_args: true, links: true)
    partial_aws_core = aws_core.partial_dependency(compile_args: false, includes: true, link_args: true, links: true)
    plugin_deps += [partial_aws_s3, partial_aws_core]
else
    warning('AWS SDK dependencies not found, skipping OBS_OBJ plugin build')
    subdir_done()
endif
plugin_deps += [dependency('asio', required: true)]

if 'OBS_OBJ' in static_plugins
    obs_obj_backend_lib = static_library(
        'OBS_OBJ',
        obs_obj_sources,
        dependencies: plugin_deps,
        cpp_args: compile_defs + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    obs_obj_backend_lib = shared_library(
        'OBS_OBJ',
        obs_obj_sources,
        dependencies: plugin_deps,
        cpp_args: compile_defs + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "OBS_OBJ=' + obs_obj_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

obj_backend_interface = declare_dependency(link_with: obj_backend_lib)