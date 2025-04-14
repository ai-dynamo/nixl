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

ucx_backend_dep = declare_dependency(link_with: ucx_backend_lib, include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../src/plugins/ucx'])

if cuda_dep.found()
    cuda_dependencies = [cuda_dep]
    cpp_args = '-DUSE_VRAM'
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

plugin_test = executable('test_plugin',
                        'test_plugin.cpp',
                        dependencies: [nixl_dep, cuda_dep],
                        include_directories: [nixl_inc_dirs, utils_inc_dirs],
                        install: true)
