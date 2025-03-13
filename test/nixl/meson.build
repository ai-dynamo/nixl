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

nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: inc_dir)
ucx_backend_dep = declare_dependency(link_with: ucx_backend_lib, include_directories: [inc_dir, '../../src/plugins/ucx'])

if cuda_dep.found()
    cuda_dependencies = [cuda_dep]
    cpp_args = '-DUSE_VRAM'
else
    cuda_dependencies = []
    cpp_args = '-UUSE_VRAM'
endif

desc_example = executable('desc_example',
           'desc_example.cpp',
           dependencies: [nixl_dep] + cuda_dependencies,
           include_directories: [inc_dir, '../../src/utils/serdes'],
           link_with: [serdes_lib],
           install: true)

md_streamer = executable('md_streamer',
            'metadata_streamer.cpp',
            dependencies: [nixl_dep] + cuda_dependencies,
            include_directories: [inc_dir],
            install: true)

agent_example = executable('agent_example',
           'agent_example.cpp',
           dependencies: [nixl_dep, ucx_backend_dep, ucx_dep] + cuda_dependencies,
           include_directories: [inc_dir, '../../src/utils/serdes', '../../src/plugins/ucx'],
           link_with: [serdes_lib],
           install: true)

p2p_socket = executable('p2p_test',
            'p2p_socket_test.cpp',
            dependencies: [nixl_dep] + cuda_dependencies,
            include_directories: [inc_dir],
            install: true)

nixl_ucx_app  = executable('nixl_test', 'nixl_test.cpp',
                           dependencies: [nixl_dep] + cuda_dependencies,
                           include_directories: [inc_dir, '../../src/utils/serdes'],
                           link_with: [serdes_lib], install: true)

map_perf_test = executable('map_perf_test',
                           'map_perf.cpp',
                           include_directories: [inc_dir],
                           install: true)
