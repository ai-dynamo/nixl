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

worker_sources = [
  'worker.cpp',
  'worker.h',
]

worker_deps = [
  cuda_dep,
  openmp_dep,
  etcd_dep
]

worker_lib = static_library('worker',
  worker_sources,
  dependencies: [worker_deps],
  include_directories: inc_dir
)

worker_dep = declare_dependency(
  link_with: worker_lib,
  dependencies: [worker_deps],
  include_directories: inc_dir
)

# Include subdirectories
subdir('nixl')
subdir('nvshmem')

# Combine all worker libraries
worker_libs = [worker_lib, nixl_worker_lib]
if nvshmem_available
  worker_libs += [nvshmem_worker_lib]
endif
