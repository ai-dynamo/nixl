# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Backend source list, exported so the unit-test executable
# (test/unit/plugins/doca_memos) can build the same translation units against
# the DOCA mocks. Defined unconditionally; everything below is gated on the
# DOCA dependencies being available, but the test only needs the file list.
doca_memos_backend_sources = files(
    'doca_memos_backend.cpp',
    'doca_memos_backend.h',
    'doca_memos_progress_engine.cpp',
    'doca_memos_progress_engine.h',
)

doca_memos_sources = [
    doca_memos_backend_sources,
    files('doca_memos_plugin.cpp'),
]

if not (doca_kv_dep.found() and doca_common_dep.found())
    if is_explicit_enable
        if not doca_kv_dep.found()
            error('DOCA_MEMOS plugin requested but doca-kv dependency not found')
        else
            error('DOCA_MEMOS plugin requested but doca-common dependency not found')
        endif
    endif
    subdir_done()
endif

plugin_deps = [nixl_infra, nixl_common_dep]
doca_plugin_deps = plugin_deps + [doca_kv_dep, doca_common_dep]

compile_defs = []
compile_flags = []
doca_memos_cpp_args = compile_defs + ['-DDOCA_ALLOW_EXPERIMENTAL_API']

if 'DOCA_MEMOS' in static_plugins
    doca_memos_backend_lib = static_library(
        'DOCA_MEMOS',
        doca_memos_sources,
        dependencies: doca_plugin_deps,
        cpp_args: doca_memos_cpp_args + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    doca_memos_backend_lib = shared_library(
        'DOCA_MEMOS',
        doca_memos_sources,
        dependencies: doca_plugin_deps,
        cpp_args: doca_memos_cpp_args + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "DOCA_MEMOS=' + doca_memos_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

doca_memos_backend_interface = declare_dependency(link_with: doca_memos_backend_lib)
