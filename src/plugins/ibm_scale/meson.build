# SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation. All rights reserved.
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

if not has_io_uring
    message('IBM_SCALE plugin requires liburing — skipping')
    subdir_done()
endif

ibm_scale_path = get_option('ibm_scale_path')

compile_flags = []
link_flags = []
ibm_scale_extra_deps = []

if ibm_scale_path != ''
    ibm_scale_lib_path = ibm_scale_path + '/lib'
    ibm_scale_inc_path = ibm_scale_path + '/include'
    # Optional: IBM Storage Scale GPFS fcntl hints (gpfs_fcntl.h).
    # When found, the plugin can fire PREFETCH / ACCESS_RANGE hints on
    # registered file descriptors.  The plugin is fully functional without
    # this library — io_uring I/O works on any filesystem.
    gpfs_lib = cc.find_library('gpfs', dirs: [ibm_scale_lib_path], required: false)
    if gpfs_lib.found() and cc.has_header('gpfs_fcntl.h',
                                           args: '-I' + ibm_scale_inc_path)
        compile_flags += ['-DHAVE_GPFS_FCNTL', '-I' + ibm_scale_inc_path]
        link_flags    += ['-L' + ibm_scale_lib_path, '-lgpfs',
                          '-Wl,-rpath,' + ibm_scale_lib_path]
        ibm_scale_extra_deps += [gpfs_lib]
        message('IBM Storage Scale GPFS hints enabled (ibm_scale_path=' + ibm_scale_path + ')')
    else
        message('IBM Storage Scale GPFS hints disabled (gpfs_fcntl.h not found at ' + ibm_scale_path + ')')
    endif
endif

ibm_scale_sources = [
    'ibm_scale_backend.cpp',
    'ibm_scale_backend.h',
    'ibm_scale_plugin.cpp',
]

if 'IBM_SCALE' in static_plugins
    ibm_scale_backend_lib = static_library('IBM_SCALE',
        ibm_scale_sources,
        dependencies: [nixl_infra, nixl_common_dep, file_utils_interface,
                       io_uring_dep] + ibm_scale_extra_deps,
        link_args: link_flags,
        cpp_args: compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')
else
    ibm_scale_backend_lib = shared_library('IBM_SCALE',
        ibm_scale_sources,
        dependencies: [nixl_infra, nixl_common_dep, file_utils_interface,
                       io_uring_dep] + ibm_scale_extra_deps,
        link_args: link_flags,
        cpp_args: compile_flags + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "IBM_SCALE=' + ibm_scale_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

ibm_scale_backend_interface = declare_dependency(link_with: ibm_scale_backend_lib)
