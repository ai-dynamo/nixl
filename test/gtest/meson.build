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

gtest_dep = dependency('gtest', version : '>=1.11.0')

if cuda_dep.found()
    cuda_dependencies = [cuda_dep]
    cpp_args = '-DUSE_VRAM'
else
    cuda_dependencies = []
    cpp_args = '-UUSE_VRAM'
endif

plugin_dir = get_option('tests_plugin_dir')
if plugin_dir != ''
  plugin_dir_arg = '--tests_plugin_dir=' + plugin_dir
else
  plugin_dir_arg = ''
endif

test_exe = executable('test_executable',
    sources : ['main.cpp', 'plugin_manager.cpp'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies : [nixl_dep, cuda_dep, gtest_dep]
)

test('gtest', test_exe, args: [plugin_dir_arg])
