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

# Check if VMM is supported
func ='cuMemCreate'
if cpp.has_function(func, dependencies: cuda_dep)
    compile_flags += '-DHAVE_CUDA_VMM'
endif


if get_option('buildtype') != 'release'
    if cuda_dep.found()
        cuda_ptr_bin = executable('cuda_ptr_test',
                                  'cuda_ptr_test.cpp',
                                  dependencies: [ cuda_utils_interface, nixl_common_dep ],
                                  include_directories: [nixl_inc_dirs],
                                  link_with: cuda_utils_lib,
                                  cpp_args: compile_flags,
                                  install: true)
    endif
endif
