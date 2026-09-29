# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if not enabled_plugins.get('REDIS')
    subdir_done()
endif

redis_sources = [
    'redis_backend.cpp',
    'redis_backend.h',
    'redis_client.cpp',
    'redis_client.h',
    'redis_plugin.cpp',
]

hiredis_async_dep = dependency('hiredis_async', required: false, version: '>=1.0.0')
libevent_dep = dependency('libevent', required: false)
libevent_pthreads_dep = dependency('libevent_pthreads', required: false)

if not hiredis_async_dep.found()
    hiredis_async_dep = dependency('hiredis', required: false, version: '>=1.0.0')
endif

if not hiredis_async_dep.found() or not libevent_dep.found() or not libevent_pthreads_dep.found()
    cc = meson.get_compiler('cpp')
    # find_library cannot enforce version; assumes system hiredis >= 1.0.0
    hiredis_lib = cc.find_library('hiredis', required: false)
    libevent_lib = cc.find_library('event', required: false)
    libevent_pthreads_lib = cc.find_library('event_pthreads', required: false)

    if not hiredis_async_dep.found() and hiredis_lib.found()
        hiredis_async_dep = hiredis_lib
    endif
    if not libevent_dep.found() and libevent_lib.found()
        libevent_dep = libevent_lib
    endif
    if not libevent_pthreads_dep.found() and libevent_pthreads_lib.found()
        libevent_pthreads_dep = libevent_pthreads_lib
    endif
endif

if not hiredis_async_dep.found() or not libevent_dep.found() or not libevent_pthreads_dep.found()
    if is_explicit_enable or 'REDIS' in static_plugins
        error('REDIS plugin requested but hiredis, libevent, and libevent_pthreads are required. Install libhiredis-dev, libevent-dev, and libevent-pthreads-dev (or equivalent), then reconfigure.')
    endif
    warning('hiredis-async, libevent, or libevent_pthreads not available, skipping REDIS plugin build')
    subdir_done()
endif

compile_defs_redis = ['-DHAVE_HIREDIS_ASYNC']

redis_plugin_deps = [nixl_infra, nixl_common_dep, hiredis_async_dep, libevent_dep, libevent_pthreads_dep]

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
