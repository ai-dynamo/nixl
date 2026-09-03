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

# device_allocator.cpp drives a fake allocator, so the handle contract is
# covered without a GPU and without the CUDA implementation.
device_allocator_test_sources = ['device_allocator.cpp']
device_allocator_test_deps = [nixl_common_dep]

# device_allocator_gpu.cpp drives the real CUDA allocator, so it needs the CUDA
# runtime headers and the implementation behind nixlGetDeviceAllocator().
# Declared here rather than relying on another suite in the same binary to have
# pulled them in. Every case in it skips when no GPU is present.
if cuda_dep.found() and is_variable('device_allocator_interface')
    device_allocator_test_sources += ['device_allocator_gpu.cpp']
    device_allocator_test_deps += [cuda_dep, device_allocator_interface]
endif

device_allocator_unit_test_dep = declare_dependency(
    sources: device_allocator_test_sources,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, gtest_inc_dirs],
    dependencies: device_allocator_test_deps,
)
