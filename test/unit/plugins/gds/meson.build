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

if cuda_dep.found()
    cuda_dependencies = [cuda_dep]
    cpp_args = '-DUSE_VRAM'
else
    cuda_dependencies = []
    cpp_args = '-UUSE_VRAM'
endif

disable_gds_backend = get_option('disable_gds_backend')
if not disable_gds_backend
    nixl_gds_app = executable('nixl_gds_test', 'nixl_gds_test.cpp',
                              dependencies: [nixl_dep, cuda_dep],
                              include_directories: [inc_dir, '../../../../src/utils/serdes'],
                              install: true)
endif


