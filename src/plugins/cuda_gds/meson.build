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

# Unified GDS implementation: the batch and per-request TaskFlow engines share
# one abstract base. The GDS entry point links both engines and selects one from
# its "mode" parameter; GDS_MT remains a thin MT-only compatibility entry point.

gds_path = get_option('gds_path')
if gds_path != ''
    gds_lib_path = gds_path + '/lib64'
    gds_inc_path = gds_path + '/include'
    cufile_dep = declare_dependency(
        link_args: ['-L' + gds_lib_path, '-lcufile'],
        include_directories: include_directories(gds_inc_path))
else
    # Create a dummy dependency that does nothing
    cufile_dep = declare_dependency()
    subdir_done()
endif

gds_incs = [nixl_inc_dirs, utils_inc_dirs]
gds_engine_deps = [
    nixl_infra,
    nixl_common_dep,
    cuda_dep,
    cufile_dep,
    file_utils_interface,
    taskflow_proj,
    thread_dep,
]
gds_engine_sources = [
    'gds_utils.cpp', 'gds_utils.h',
    'gds_backend.cpp', 'gds_backend.h',
    'gds_mt_engine.cpp', 'gds_mt_engine.h',
]
if enabled_plugins.get('GDS')
    gds_engine_sources += ['gds_batch_engine.cpp', 'gds_batch_engine.h']
endif

gds_engine_lib = static_library('gds_engine',
    gds_engine_sources,
    dependencies: gds_engine_deps,
    include_directories: gds_incs,
    pic: true,
    install: false)

gds_engine_dep = declare_dependency(
    link_with: gds_engine_lib,
    dependencies: gds_engine_deps,
    include_directories: gds_incs)

# "GDS" backend name entry point (batch by default, MT when mode=mt).
if enabled_plugins.get('GDS')
    gds_sources = ['gds_plugin.cpp']
    gds_deps = [gds_engine_dep]
    if 'GDS' in static_plugins
        gds_backend_lib = static_library('GDS',
            gds_sources,
            dependencies: gds_deps,
            include_directories: gds_incs,
            install: false,
            name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
    else
        gds_backend_lib = shared_library('GDS',
            gds_sources,
            dependencies: gds_deps,
            include_directories: gds_incs,
            install: true,
            cpp_args: ['-fPIC'],
            name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
            install_dir: plugin_install_dir,
            install_rpath: '$ORIGIN/..')
        if get_option('buildtype') == 'debug'
            run_command('sh', '-c',
                'echo "GDS=' + gds_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                check: true
            )
        endif
    endif
    gds_backend_interface = declare_dependency(
        link_with: gds_backend_lib,
        dependencies: [taskflow_proj, thread_dep])
endif

# "GDS_MT" backend name entry point (per-request TaskFlow strategy).
if enabled_plugins.get('GDS_MT')
    gds_mt_sources = ['gds_mt_plugin.cpp']
    gds_mt_deps = [gds_engine_dep]
    if 'GDS_MT' in static_plugins
        gds_mt_backend_lib = static_library('GDS_MT',
            gds_mt_sources,
            dependencies: gds_mt_deps,
            include_directories: gds_incs,
            install: false,
            name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
    else
        gds_mt_backend_lib = shared_library('GDS_MT',
            gds_mt_sources,
            dependencies: gds_mt_deps,
            include_directories: gds_incs,
            install: true,
            cpp_args: ['-fPIC'],
            name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
            install_dir: plugin_install_dir,
            install_rpath: '$ORIGIN/..')
        if get_option('buildtype') == 'debug'
            run_command('sh', '-c',
                'echo "GDS_MT=' + gds_mt_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                check: true
            )
        endif
    endif
    gds_mt_backend_interface = declare_dependency(
        link_with: gds_mt_backend_lib,
        dependencies: [taskflow_proj, thread_dep])
endif
