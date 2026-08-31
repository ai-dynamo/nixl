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

# ODM dma-buf export of GPU VRAM (cuMemGetHandleForAddressRange) requires the
# CUDA driver API. Without CUDA the plugin still builds but refuses to
# initialize at runtime (see nixlOdmEngine constructor).

odm_sources = ['odm_plugin.cpp', 'odm_backend.cpp', 'odm_backend.h', 'odm_ioctl.h']
odm_deps = [nixl_infra, nixl_common_dep]
odm_compile_flags = []

if cuda_dep.found()
    odm_deps += [cuda_dep]
    odm_compile_flags += ['-DHAVE_CUDA']
    message('ODM plugin: CUDA found, enabling GPU VRAM dma-buf export (ODM controller path)')
else
    # ODM has no fallback: without CUDA the plugin still builds but refuses to
    # initialize at runtime (see nixlOdmEngine constructor).
    message('ODM plugin: CUDA NOT found — plugin will refuse to initialize at runtime (no fallback)')
endif

if 'ODM' in static_plugins
    odm_backend_lib = static_library('ODM',
        odm_sources,
        dependencies: odm_deps,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        cpp_args: odm_compile_flags,
        name_prefix: 'libplugin_')
else
    odm_backend_lib = shared_library('ODM',
        odm_sources,
        dependencies: odm_deps,
        cpp_args: odm_compile_flags + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "ODM=' + odm_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

odm_backend_interface = declare_dependency(link_with: odm_backend_lib)
