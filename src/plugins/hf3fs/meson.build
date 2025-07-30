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


# Use the hf3fs_lib_path from parent meson.build
threefs_inc_path = '/usr/include/hf3fs'
threefs_dep = declare_dependency(
  link_args : ['-L' + hf3fs_lib_path, '-l' + hf3fs_lib_file],
  include_directories : include_directories(threefs_inc_path))

if 'HF3FS' in static_plugins
  hf3fs_backend_lib = static_library('HF3FS',
                    'hf3fs_utils.cpp', 'hf3fs_utils.h',
                    'hf3fs_backend.cpp', 'hf3fs_backend.h',
                    'hf3fs_plugin.cpp',
                    dependencies: [nixl_infra, threefs_dep, nixl_common_dep, file_utils_interface],
                    include_directories: [nixl_inc_dirs, utils_inc_dirs],
                    install: false,
                    cpp_args : compile_flags,
                    name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
  hf3fs_backend_lib = shared_library('HF3FS',
                    'hf3fs_utils.cpp', 'hf3fs_utils.h',
                    'hf3fs_backend.cpp', 'hf3fs_backend.h',
                    'hf3fs_plugin.cpp',
                    dependencies: [nixl_infra, threefs_dep, nixl_common_dep, file_utils_interface],
                    include_directories: [nixl_inc_dirs, utils_inc_dirs],
                    install: true,
                    cpp_args : ['-fPIC'],
                    name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
                    install_dir: plugin_install_dir,
                    install_rpath: '$ORIGIN/..')
  if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "HF3FS=' + hf3fs_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

hf3fs_backend_interface = declare_dependency(link_with: hf3fs_backend_lib)
