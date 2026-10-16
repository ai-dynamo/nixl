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

device_utils_lib = shared_library('device_utils',
           ['device_ops.cpp', 'device_ops.h',
            'host_published_mem.cpp', 'host_published_mem.h',
            'device_memview.cpp', 'device_memview.h'],
           # nixl_device_inc_dirs resolves <gpu/device_types.cuh>: the memview
           # wrapper is the device API's dispatch handle, laid out there.
           include_directories: [nixl_inc_dirs, utils_inc_dirs, nixl_device_inc_dirs],
           dependencies: [nixl_common_dep, dl_dep],
           install: true)

device_utils_interface = declare_dependency(link_with: device_utils_lib,
                                                include_directories: [nixl_device_inc_dirs])

if cuda_dep.found()
    shared_library('nixl_device_ops_cuda',
               ['device_ops_cuda.cpp',
                'device_ops_cuda_gdrcopy.cpp', 'device_ops_cuda_gdrcopy.h'],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               dependencies: [cuda_dep, nixl_common_dep, dl_dep],
               install: true)
endif

subdir('proxy')
