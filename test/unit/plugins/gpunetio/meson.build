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
        cpp_args = '-DHAVE_CUDA'
        nvtx_dep = nvcc.find_library (
        'nvToolsExt', dirs
        : '/usr/local/cuda/lib64', required
        : true)
else
        cuda_dependencies = []
        cpp_args = '-UHAVE_CUDA'
        nvtx_dep = []
endif

nixl_gpunetio_stream_app = executable ('nixl_gpunetio_stream_test', 'nixl_gpunetio_stream_test.cu',
        dependencies: [nixl_dep, nixl_infra, stream_interface] + cuda_dep + nvtx_dep,
        include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/utils/serdes'],
        link_with: [serdes_lib],
        install: true)
