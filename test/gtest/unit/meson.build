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

unit_test_deps = [
    nixl_dep,
    gtest_dep,
]

aws_s3 = dependency('aws-cpp-sdk-s3', static: false, required: false)
if aws_s3.found()
    subdir('obj')
    unit_test_deps += [obj_unit_test_dep]
endif

unit_test_exe = executable('unit',
    sources : [
        'main.cpp',
    ],
    cpp_args : cpp_args,
    dependencies : unit_test_deps,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
    ],
    link_with: [nixl_build_lib],
    install : true
)

test('unit', unit_test_exe)
