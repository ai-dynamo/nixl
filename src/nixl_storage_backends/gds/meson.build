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

gds_path = get_option('gds_path')
if gds_path != ''
  gds_lib_path = gds_path + '/lib'
  gds_inc_path = gds_path + '/include'
  cufile_dep = declare_dependency(
    link_args : ['-L' + gds_lib_path, '-lcufile'],
    include_directories : include_directories(gds_inc_path))
endif

if 'GDS' in static_plugins
  gds_backend_lib = static_library('GDS',
                    'gds_utils.cpp', 'gds_utils.h',
                    'gds_backend.cpp', 'gds_backend.h',
                    'gds_plugin.cpp',
                    dependencies: [nixl_dep, cuda_dep, cufile_dep],
                    include_directories: inc_dir,
                    install: true,
                    cpp_args : compile_flags,
                    name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
                    install_dir: plugin_install_dir)
else
  gds_backend_lib = shared_library('GDS',
                    'gds_utils.cpp', 'gds_utils.h',
                    'gds_backend.cpp', 'gds_backend.h',
                    'gds_plugin.cpp',
                    dependencies: [nixl_dep, cuda_dep, cufile_dep],
                    include_directories: inc_dir,
                    install: true,
                    cpp_args : ['-fPIC'],
                    name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
                    install_dir: plugin_install_dir)
  custom_target('copy_gds_plugin',
               input: gds_backend_lib,
               output: 'copied_gds_plugin',
               command: ['cp', '@INPUT@', plugin_build_dir / 'libplugin_GDS.so'],
               build_by_default: true)
endif

gds_backend_interface = declare_dependency(link_with: gds_backend_lib)
