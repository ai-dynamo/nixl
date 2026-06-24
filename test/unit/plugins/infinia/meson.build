# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 DataDirect Networks, Inc.
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

compile_flags = []
if cuda_dep.found()
    compile_flags = [ '-DHAVE_CUDA' ]
else
    compile_flags = [ '-UHAVE_CUDA' ]
endif

infinia_nixl_app = executable('infinia_nixl_test', 'infinia_nixl_test.cpp',
                          dependencies: [nixl_dep, nixl_infra, cuda_dep],
                          cpp_args : compile_flags,
                          include_directories: [nixl_inc_dirs, utils_inc_dirs],
                          install: true)
