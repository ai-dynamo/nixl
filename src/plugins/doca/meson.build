#SPDX - FileCopyrightText : Copyright(c) 2025 NVIDIA CORPORATION &AFFILIATES.All rights reserved.
#SPDX - License - Identifier : Apache - 2.0
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http: // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

gpunetio_device_path = dependency('doca-gpunetio').get_variable(pkgconfig : 'libdir')
dependency_gpunetio_device = declare_dependency(compile_args : '-Wl,--whole-archive',
                                                link_args : ['-L' + gpunetio_device_path , '-ldoca_gpunetio_device'],)

plugin_doca_deps = [dependency('doca-common'), dependency('doca-rdma'), dependency('doca-gpunetio'), dependency_gpunetio_device]

#Get Abseil dependencies
absl_log_dep = dependency('absl_log', required: true)

gpu_cuda_args = []
compile_flags += '-D DOCA_ALLOW_EXPERIMENTAL_API'

if 'DOCA' in static_plugins
    doca_backend_lib = static_library('DOCA',
               'doca_backend.cpp', 'doca_backend.h', 'doca_backend_aux.h', 'doca_plugin.cpp', 'doca_utils.cpp', 'kernels.cu',
               dependencies: [nixl_infra, serdes_interface, cuda_dep, plugin_doca_deps, absl_log_dep],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: true,
               cpp_args : compile_flags,
               cuda_args : gpu_cuda_args + ['-rdc=true'],
               name_prefix: 'libplugin_',
               install_dir: plugin_install_dir)  # Custom prefix for plugin libraries
else
    doca_backend_lib = shared_library('DOCA',
               'doca_backend.cpp', 'doca_backend.h', 'doca_backend_aux.h', 'doca_plugin.cpp', 'doca_utils.cpp', 'kernels.cu',
               dependencies: [nixl_infra, serdes_interface, cuda_dep, plugin_doca_deps, absl_log_dep],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: true,
               cpp_args : compile_flags + ['-fPIC'],
               cuda_args : gpu_cuda_args,
               name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
               install_dir: plugin_install_dir)

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "DOCA=' + doca_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

doca_backend_interface = declare_dependency(link_with: doca_backend_lib)
