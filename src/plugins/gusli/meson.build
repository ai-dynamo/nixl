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


# Use the gusli_lib_path from parent meson.build
gusli_inc_path = '/usr/include'
gusli_dep = declare_dependency(
  link_args : ['-L' + gusli_lib_path, '-l' + gusli_lib_file],
  include_directories : include_directories(gusli_inc_path))

# Get Abseil dependencies
absl_log_dep = dependency('absl_log', required: true)
cpp_sources = ['gusli_plugin.cpp', 'gusli_backend.cpp', 'gusli_backend.h']

if 'GUSLI' in static_plugins
  gusli_backend_lib = static_library('GUSLI', cpp_sources,
                    dependencies: [nixl_infra, gusli_dep, absl_log_dep],
                    include_directories: [nixl_inc_dirs, utils_inc_dirs],
                    install: false,
                    cpp_args : compile_flags,
                    name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
  gusli_backend_lib = shared_library('GUSLI', cpp_sources,
                    dependencies: [nixl_infra, gusli_dep, absl_log_dep],
                    include_directories: [nixl_inc_dirs, utils_inc_dirs],
                    install: true,
                    cpp_args : ['-fPIC'],
                    name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
                    install_dir: plugin_install_dir)
  if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "GUSLI=' + gusli_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

gusli_backend_interface = declare_dependency(link_with: gusli_backend_lib)
