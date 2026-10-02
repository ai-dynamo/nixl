# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

daos_sources = [
    'daos_backend.cpp',
    'daos_backend.h',
    'daos_client.cpp',
    'daos_client.h',
    'daos_plugin.cpp',
]

daos_deps = [
    nixl_infra,
    nixl_common_dep,
    thread_dep,
    daos_dfs_dep,
    daos_client_dep,
]

if 'DAOS' in static_plugins
    daos_backend_lib = static_library(
        'DAOS',
        daos_sources,
        dependencies: daos_deps,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')
    daos_backend_interface = declare_dependency(link_whole: daos_backend_lib)
else
    daos_backend_lib = shared_library(
        'DAOS',
        daos_sources,
        dependencies: daos_deps,
        cpp_args: ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    daos_backend_interface = declare_dependency(link_with: daos_backend_lib)

    if get_option('buildtype') == 'debug'
        run_command(
            'sh', '-c',
            'echo "DAOS=' + daos_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true)
    endif
endif
