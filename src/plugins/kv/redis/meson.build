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

# REDIS implementation sources live in this directory; the plugin entrypoint
# stays in the shared KV layer to match the OBJ plugin layout.

if not enabled_plugins.get('REDIS')
    subdir_done()
endif

redis_sources = [
    '../kv_backend.cpp',
    '../kv_plugin.cpp',
    'redis_engine.cpp',
    'redis_engine.h',
    'redis_executor.h',
    'client.cpp',
    'client.h',
    'engine_impl.cpp',
    'engine_impl.h',
]

hiredis_async_dep = dependency('hiredis_async', required: false)
libevent_dep = dependency('libevent', required: false)
libevent_pthreads_dep = dependency('libevent_pthreads', required: false)

if not hiredis_async_dep.found()
    hiredis_async_dep = dependency('hiredis', required: false)
endif

if not hiredis_async_dep.found() or not libevent_dep.found()
    cc = meson.get_compiler('cpp')
    hiredis_lib = cc.find_library('hiredis', required: false)
    libevent_lib = cc.find_library('event', required: false)
    libevent_pthreads_lib = cc.find_library('event_pthreads', required: false)

    if hiredis_lib.found() and libevent_lib.found()
        if not hiredis_async_dep.found()
            hiredis_async_dep = hiredis_lib
        endif
        if not libevent_dep.found()
            libevent_dep = libevent_lib
        endif
        if libevent_pthreads_lib.found() and not libevent_pthreads_dep.found()
            libevent_pthreads_dep = libevent_pthreads_lib
        endif
    endif
endif

if not hiredis_async_dep.found() or not libevent_dep.found()
    if is_explicit_enable or 'REDIS' in static_plugins
        error('REDIS plugin requested but hiredis and libevent are required. Install libhiredis-dev and libevent-dev (or equivalent), then reconfigure.')
    endif
    warning('hiredis-async or libevent not available, skipping REDIS plugin build')
    subdir_done()
endif

compile_defs_redis = []
if is_variable('compile_defs')
    compile_defs_redis = compile_defs
endif
if hiredis_async_dep.found() and libevent_dep.found()
    compile_defs_redis += ['-DHAVE_HIREDIS_ASYNC']
endif

redis_plugin_deps = [nixl_infra, nixl_common_dep, hiredis_async_dep, libevent_dep, dependency('asio', required: true)]
if libevent_pthreads_dep.found()
    redis_plugin_deps += [libevent_pthreads_dep]
else
    cc = meson.get_compiler('cpp')
    event_pthreads = cc.find_library('event_pthreads', required: false)
    if event_pthreads.found()
        redis_plugin_deps += [event_pthreads]
    else
        warning('libevent_pthreads not found; REDIS plugin may fail to link (evthread_use_pthreads)')
    endif
endif

if not is_variable('compile_flags')
    compile_flags = []
endif

if 'REDIS' in static_plugins
    redis_backend_lib = static_library(
        'REDIS',
        redis_sources,
        dependencies: redis_plugin_deps,
        cpp_args: compile_defs_redis + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')
else
    redis_backend_lib = shared_library(
        'REDIS',
        redis_sources,
        dependencies: redis_plugin_deps,
        cpp_args: compile_defs_redis + compile_flags + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "REDIS=' + redis_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

redis_backend_interface = declare_dependency(
    link_with: redis_backend_lib,
    dependencies: redis_plugin_deps,
)
