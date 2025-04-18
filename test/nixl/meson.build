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

ucx_backend_dep = declare_dependency(link_with: ucx_backend_lib, include_directories: [nixl_inc_dirs, '../../src/plugins/ucx'])

if cuda_dep.found()
    cuda_dependencies = [cuda_dep]
    cpp_args = '-DUSE_VRAM'
    nvtx_dep = nvcc.find_library('nvToolsExt', dirs: '/usr/local/cuda/lib64', required: true)
else
    cuda_dependencies = []
    cpp_args = '-UUSE_VRAM'
endif


desc_example = executable('desc_example',
           'desc_example.cpp',
           dependencies: [nixl_dep, cuda_dependencies, nixl_infra],
           include_directories: [nixl_inc_dirs, utils_inc_dirs],
           link_with: [serdes_lib],
           install: true)

agent_example = executable('agent_example',
           'agent_example.cpp',
           dependencies: [nixl_dep, nixl_infra, ucx_backend_dep, ucx_dep] + cuda_dependencies,
           include_directories: [nixl_inc_dirs, utils_inc_dirs],
           link_with: [serdes_lib],
           install: true)

nixl_ucx_app  = executable('nixl_test', 'nixl_test.cpp',
                           dependencies: [nixl_dep, nixl_infra, stream_interface] + cuda_dependencies,
                           include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../src/utils/serdes'],
                           link_with: [serdes_lib], install: true)

nixl_doca_stream_app  = executable('nixl_doca_stream_test', 'nixl_doca_stream_test.cu',
                           dependencies: [nixl_dep, nixl_infra, stream_interface] + cuda_dep + nvtx_dep,
                           include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../src/utils/serdes'],
                           link_with: [serdes_lib], install: true)

if dependency('doca-gpunetio').found()
    gpunetio_device_path = dependency('doca-gpunetio').get_variable(pkgconfig : 'libdir')
    dependency_gpunetio_device = declare_dependency(compile_args : '-Wl,--whole-archive',
                                                link_args : ['-L' + gpunetio_device_path , '-ldoca_gpunetio_device'],)

    plugin_doca_deps = [dependency('doca-common'), dependency('doca-rdma'), dependency('doca-gpunetio'), dependency_gpunetio_device]

    nixl_doca_device_app  = executable('nixl_doca_device_test', 'nixl_doca_device_test.cu',
                           dependencies: [nixl_dep, nixl_infra, stream_interface] + plugin_doca_deps + cuda_dep + nvtx_dep,
                           include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../src/utils/serdes'],
                           link_with: [serdes_lib], install: true)
endif


plugin_test = executable('test_plugin',
                        'test_plugin.cpp',
                        dependencies: [nixl_dep, cuda_dep],
                        include_directories: [nixl_inc_dirs, utils_inc_dirs],
                        install: true)
