# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 DataDirect Networks, Inc.
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

infinia_path = get_option('infinia_path')
if infinia_path == ''
  error('INFINIA backend requires -Dinfinia_path to point to Infinia Client libraries installation')
endif

infinia_lib_path = infinia_path + '/lib'
infinia_inc_path = infinia_path + '/include'
infinia_dep = declare_dependency(
  link_args : ['-L' + infinia_lib_path,
               '-Wl,--push-state,--as-needed',
               '-Wl,-lstdc++',
               '-Wl,-rpath,' + infinia_lib_path,
               '-lred_client',
               '-lred_async',
               '-Wl,--pop-state'],
  include_directories: infinia_inc_path)

compile_flags = []
if cuda_dep.found()
    compile_flags = [ '-DHAVE_CUDA' ]
else
    compile_flags = [ '-UHAVE_CUDA' ]
endif

# RED async library requires C++20 coroutines
cpp_std_override = 'c++20'

# Get Abseil dependencies
absl_log_dep = dependency('absl_log', required: true)

if 'INFINIA' in static_plugins
  infinia_backend_lib = static_library('INFINIA',
    'infinia_backend.cpp', 'infinia_backend.h',
    'infinia_client.cpp', 'infinia_client.h',
    'infinia_plugin.cpp',
    dependencies: [ nixl_infra, cuda_dep, infinia_dep, absl_log_dep ],
    include_directories: [nixl_inc_dirs,utils_inc_dirs, infinia_inc_path],
    install: false,
    cpp_args : compile_flags + ['-fcoroutines'],
    override_options : ['cpp_std=c++20'],
    name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
  infinia_backend_lib = shared_library('INFINIA',
    'infinia_backend.cpp', 'infinia_backend.h',
    'infinia_client.cpp', 'infinia_client.h',
    'infinia_plugin.cpp',
    dependencies: [nixl_infra, cuda_dep, infinia_dep, absl_log_dep ],
    include_directories: [nixl_inc_dirs,utils_inc_dirs, infinia_inc_path],
    install: true,
    cpp_args : compile_flags + ['-fPIC', '-fcoroutines'],
    override_options : ['cpp_std=c++20'],
    name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
    install_dir: plugin_install_dir,
    install_rpath: '$ORIGIN/..')

  if get_option('buildtype') == 'debug'
    run_command('sh', '-c',
                'echo "INFINIA=' + infinia_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                check: true
    )
    endif
endif

infinia_backend_interface = declare_dependency(link_with: infinia_backend_lib)
