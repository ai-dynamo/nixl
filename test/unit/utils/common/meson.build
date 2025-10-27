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

if cuda_dep.found()
    cuda_dependencies = [cuda_dep]
    cpp_args = '-DUSE_VRAM'
else
    cuda_dependencies = []
    cpp_args = '-UUSE_VRAM'
endif

p2p_socket = executable('p2p_test',
            'p2p_socket_test.cpp',
            dependencies: [nixl_dep, nixl_infra, stream_interface] + cuda_dependencies,
            include_directories: [nixl_inc_dirs, utils_inc_dirs],
            install: true)

map_perf_test = executable('map_perf_test',
                           'map_perf.cpp',
                           dependencies: [nixl_test_utils_dep],
                           include_directories: [nixl_inc_dirs, utils_inc_dirs],
                           install: true)
