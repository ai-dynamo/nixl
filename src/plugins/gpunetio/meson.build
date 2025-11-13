# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

plugin_gpunetio_deps = [dependency('libibverbs'), dependency('libmlx5'), dependency('doca-common'), dependency('doca-verbs'), dependency('doca-gpunetio')]

#Get Abseil dependencies
absl_log_dep = dependency('absl_log', required: true)

gpu_cuda_args = []
compile_flags += '-D DOCA_ALLOW_EXPERIMENTAL_API'

if 'GPUNETIO' in static_plugins
    gpunetio_backend_lib = static_library('GPUNETIO',
               'gpunetio_backend.cpp', 'gpunetio_backend.h', 'gpunetio_backend_aux.h', 'gpunetio_plugin.cpp', 'gpunetio_utils.cpp', 'gpunetio_kernels.cu',
               dependencies: [nixl_infra, serdes_interface, cuda_dep, plugin_gpunetio_deps, absl_log_dep],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: true,
               cpp_args : compile_flags,
               cuda_args : gpu_cuda_args + ['-rdc=true'],
               name_prefix: 'libplugin_',
               install_dir: plugin_install_dir)  # Custom prefix for plugin libraries
else
    gpunetio_backend_lib = shared_library('GPUNETIO',
               'verbs/verbs.cpp', 'gpunetio_backend.cpp', 'gpunetio_backend.h', 'gpunetio_backend_aux.h', 'gpunetio_plugin.cpp', 'gpunetio_utils.cpp', 'gpunetio_kernels.cu',
               dependencies: [nixl_infra, serdes_interface, cuda_dep, plugin_gpunetio_deps, absl_log_dep],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: true,
               cpp_args : compile_flags + ['-fPIC'],
               cuda_args : gpu_cuda_args,
               name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
               install_dir: plugin_install_dir)

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "GPUNETIO=' + gpunetio_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

gpunetio_backend_interface = declare_dependency(link_with: gpunetio_backend_lib)
