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

cuda_utils_depends_from = [ nixl_common_dep ]

compile_flags = [ ]
if cuda_dep.found()
    compile_flags += '-DHAVE_CUDA'
    cuda_utils_depends_from += [ cuda_dep ]

    func ='cuMemRetainAllocationHandle'
    if cpp.has_function(func, dependencies: cuda_dep)
        compile_flags += '-DHAVE_' + func.underscorify().to_upper() + '=1'
    endif
endif

cuda_utils_lib = library('cuda_utils',
                         'cuda_utils.cpp', 'cuda_utils.h',
                         dependencies: cuda_utils_depends_from,
                         cpp_args : compile_flags,
                         include_directories: [ nixl_inc_dirs ],
                         install: true)

cuda_utils_interface = declare_dependency(link_with: cuda_utils_lib,
                                          include_directories: utils_inc_dirs,
                                          dependencies: cuda_utils_depends_from )