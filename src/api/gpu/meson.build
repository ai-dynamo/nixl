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

# proxy/nixl_device_proxy.cu defines g_nixl_proxy_ctx, the __device__ global
# that ProxyRuntime publishes via cudaMemcpyToSymbol and device kernels read
# through load_proxy_context(). It must be compiled as a separate CUDA TU so
# that the symbol exists at device link time.
if cuda_dep.found()
    nixl_gpu_proxy_lib = static_library('nixl_gpu_proxy',
        'proxy/nixl_device_proxy.cu',
        dependencies: [cuda_dep],
        include_directories: [nixl_inc_dirs],
        install: false)

    nixl_gpu_proxy_dep = declare_dependency(link_with: nixl_gpu_proxy_lib)

    if gpu_device_api_is_proxy
        gpu_device_api_link_with = [nixl_gpu_proxy_lib]
    endif
endif
