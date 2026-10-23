# SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

wqs_inc_path = get_option('wqs_inc_path')
wqs_lib_path = get_option('wqs_lib_path')
wqs_include_dirs = [nixl_inc_dirs, utils_inc_dirs]
wqs_header_args = []
if wqs_inc_path != ''
    wqs_include_dirs += [include_directories(wqs_inc_path, is_system: true)]
    wqs_header_args += ['-I' + wqs_inc_path]
endif
wqs_library_dirs = []
if wqs_lib_path != ''
    wqs_library_dirs += [wqs_lib_path]
endif
wqs_store_dep = cpp.find_library('wqs_store', dirs: wqs_library_dirs, required: false)
wqs_header_found = cpp.has_header('shared_client.h', args: wqs_header_args)
if not wqs_store_dep.found() or not wqs_header_found
    if is_explicit_enable or 'WQS' in static_plugins
        error('WQS requires shared_client.h and libwqs_store.so; set wqs_inc_path and wqs_lib_path')
    endif
    warning('WQS client headers or library unavailable, skipping WQS plugin')
    subdir_done()
endif

wqs_sources = [
    'wqs_backend.cpp',
    'wqs_buffer_pool.cpp',
    'wqs_client.cpp',
    'wqs_session.cpp',
    'wqs_plugin.cpp',
]
wqs_plugin_deps = [nixl_infra, nixl_common_dep, thread_dep, wqs_store_dep]
if not is_variable('compile_flags')
    compile_flags = []
endif

if 'WQS' in static_plugins
    wqs_backend_lib = static_library(
        'WQS',
        wqs_sources,
        dependencies: wqs_plugin_deps,
        include_directories: wqs_include_dirs,
        cpp_args: compile_flags,
        name_prefix: 'libplugin_',
        install: false,
    )
else
    wqs_backend_lib = shared_library(
        'WQS',
        wqs_sources,
        dependencies: wqs_plugin_deps,
        include_directories: wqs_include_dirs,
        cpp_args: compile_flags,
        name_prefix: 'libplugin_',
        install: true,
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..',
    )
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "WQS=' + wqs_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true,
        )
    endif
endif

wqs_backend_interface = declare_dependency(
    link_with: wqs_backend_lib,
    dependencies: wqs_plugin_deps,
)
