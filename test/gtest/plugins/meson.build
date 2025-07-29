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

aws_s3 = dependency('aws-cpp-sdk-s3', static: false, required: false)
if not aws_s3.found()
    message('aws-cpp-sdk-s3 not found, skipping plugins_gtest build')
    subdir_done()
endif

plugins_test_exe = executable('plugins_gtest',
    sources : ['../main.cpp', 'obj_plugin.cpp'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, plugins_inc_dirs, '.'],
    dependencies : [nixl_dep, gtest_dep, absl_strings_dep, absl_time_dep, plugin_deps,
                    obj_backend_interface],
    link_with: [nixl_build_lib],
    install : true
)

test('plugins_gtest', plugins_test_exe)
