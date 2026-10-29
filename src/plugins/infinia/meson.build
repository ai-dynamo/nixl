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

if get_option('disable_infinia_backend')
    if is_explicit_enable or 'INFINIA' in static_plugins
        error('INFINIA plugin requested but Infinia backend is disabled')
    endif
    subdir_done()
endif

infinia_path = get_option('infinia_path')
if infinia_path == ''
    if is_explicit_enable or 'INFINIA' in static_plugins
        error('INFINIA plugin requested but -Dinfinia_path is empty; point it to the Infinia SDK (libred_async.so) installation')
    endif
    warning('INFINIA plugin disabled: -Dinfinia_path is empty')
    subdir_done()
endif

infinia_lib_path = infinia_path + '/lib'
infinia_inc_path = infinia_path + '/include'
infinia_lib_file = 'red_async'
infinia_lib_found = cc.find_library(infinia_lib_file, dirs: [infinia_lib_path], required: false)

# Infinia SDK 2.5+ symbols required by this plugin.
infinia_symbol_found = cpp.compiles('''
    #include <red/red_async.hpp>
    int main() {
        (void)red_async::RED_ASYNC_DEFAULT_MAX_RETRIES;
        (void)red_async::RED_ASYNC_DEFAULT_BATCH_SIZE;
        (void)red_async::RED_ASYNC_OP_CACHE_HEAD;
        (void)red_async::RED_ASYNC_OP_CACHE_GET;
        (void)red_async::RED_ASYNC_OP_CACHE_PUT;
        return 0;
    }
    ''',
    args: ['-I' + infinia_inc_path],
    name: 'Infinia SDK 2.5 symbols')

infinia_sdk_ok = infinia_symbol_found and infinia_lib_found.found()
if (not infinia_sdk_ok)
    if is_explicit_enable or 'INFINIA' in static_plugins
        if not infinia_lib_found.found()
            error('INFINIA plugin requested but red_async library not found')
        else
            error('INFINIA plugin requested but Infinia SDK is too old: needs release 2.5.0 or above')
        endif
    endif
    if not infinia_lib_found.found()
        warning('INFINIA plugin requested but red_async library not found')
    endif
    if not infinia_symbol_found
        warning('INFINIA plugin requested but Infinia SDK is too old: needs release 2.5.0 or above')
    endif
    warning('INFINIA plugin disabled')
    subdir_done()
endif

infinia_dep = declare_dependency(
  link_args : ['-L' + infinia_lib_path,
               '-Wl,--push-state,--as-needed',
               '-Wl,-lstdc++',
               '-Wl,-rpath,' + infinia_lib_path,
               '-lred_async',
               '-Wl,--pop-state'],
  include_directories: infinia_inc_path)

compile_flags = []
if cuda_dep.found()
    compile_flags = [ '-DHAVE_CUDA' ]
else
    compile_flags = [ '-UHAVE_CUDA' ]
endif

# Get Abseil dependencies
absl_log_dep = dependency('absl_log', required: true)

if 'INFINIA' in static_plugins
  infinia_backend_lib = static_library('INFINIA',
    'infinia_backend.cpp', 'infinia_backend.h',
    'infinia_client.cpp', 'infinia_client.h',
    'infinia_plugin.cpp',
    dependencies: [ nixl_infra, cuda_dep, infinia_dep, absl_log_dep, nixl_common_dep ],
    include_directories: [nixl_inc_dirs,utils_inc_dirs, infinia_inc_path],
    install: false,
    cpp_args : compile_flags + ['-fcoroutines'],
    name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
  infinia_backend_lib = shared_library('INFINIA',
    'infinia_backend.cpp', 'infinia_backend.h',
    'infinia_client.cpp', 'infinia_client.h',
    'infinia_plugin.cpp',
    dependencies: [nixl_infra, cuda_dep, infinia_dep, absl_log_dep, nixl_common_dep ],
    include_directories: [nixl_inc_dirs,utils_inc_dirs, infinia_inc_path],
    install: true,
    cpp_args : compile_flags + ['-fPIC', '-fcoroutines'],
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
