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

# Unified GDS plugin: the "GDS" batch API and "GDS_MT" per-request TaskFlow
# backends share one abstract base engine (nixlGdsEngine), compiled once into
# gds_common_lib. Each backend name builds a thin entry point plus its concrete
# engine subclass.

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

gds_common_incs = [nixl_inc_dirs, utils_inc_dirs]
# Shared engine deps exclude TaskFlow because the common registration and
# metadata path does not create workers.
gds_common_deps = [nixl_infra, nixl_common_dep, cuda_dep, cufile_dep, file_utils_interface]

# Shared base engine + utils compiled exactly once (PIC) so it can be embedded
# into a shared plugin or linked statically into the core without duplicate
# symbols. No TaskFlow here.
gds_common_lib = static_library('gds_common',
    'gds_utils.cpp', 'gds_utils.h',
    'gds_backend.cpp', 'gds_backend.h',
    dependencies: gds_common_deps,
    include_directories: gds_common_incs,
    cpp_args: ['-fPIC'],
    install: false)

gds_common_dep = declare_dependency(
    link_with: gds_common_lib,
    dependencies: gds_common_deps,
    include_directories: gds_common_incs)

gds_sources = ['gds_plugin.cpp', 'gds_batch_engine.cpp', 'gds_batch_engine.h']
gds_deps = [gds_common_dep]

# "GDS" backend name entry point (cuFile batch engine).
if enabled_plugins.get('GDS')
    if 'GDS' in static_plugins
        gds_backend_lib = static_library('GDS',
            gds_sources,
            dependencies: gds_deps,
            include_directories: gds_common_incs,
            install: false,
            name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
    else
        gds_backend_lib = shared_library('GDS',
            gds_sources,
            dependencies: gds_deps,
            include_directories: gds_common_incs,
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
    gds_backend_interface = declare_dependency(link_with: gds_backend_lib)
endif

# "GDS_MT" backend name entry point (per-request TaskFlow strategy).
if enabled_plugins.get('GDS_MT')
    if not taskflow_proj.found()
        if is_explicit_enable
            error('GDS_MT plugin requested but Taskflow dependency not found')
        endif
        # Otherwise silently skip building the GDS_MT entry point.
    else
        gds_mt_sources = ['gds_mt_plugin.cpp', 'gds_mt_engine.cpp', 'gds_mt_engine.h']
        gds_mt_deps = [gds_common_dep, taskflow_proj, thread_dep]
        if 'GDS_MT' in static_plugins
            gds_mt_backend_lib = static_library('GDS_MT',
                gds_mt_sources,
                dependencies: gds_mt_deps,
                include_directories: gds_common_incs,
                install: false,
                name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
        else
            gds_mt_backend_lib = shared_library('GDS_MT',
                gds_mt_sources,
                dependencies: gds_mt_deps,
                include_directories: gds_common_incs,
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
endif
