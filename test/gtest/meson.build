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

gtest_dep = dependency('gtest', version : '>=1.11.0', required : false)

if not gtest_dep.found()
    message('GTest not found, skipping gtest build')
    subdir_done()
endif

if cuda_dep.found()
    cuda_dependencies = [cuda_dep]
else
    cuda_dependencies = []
endif

subdir('mocks')

plugin_dirs_arg = '--tests_plugin_dirs=' + mocks_dep.get_variable('path')

cpp_flags = []

if get_option('test_all_plugins')
  cpp_flags+='-DTEST_ALL_PLUGINS'
endif

test_exe = executable('gtest',
    sources : ['main.cpp', 'plugin_manager.cpp'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    cpp_args : cpp_flags,
    dependencies : [nixl_dep, cuda_dep, gtest_dep],
    install : true
)

test('gtest', test_exe, args: [plugin_dirs_arg])

if get_option('b_sanitize').split(',').contains('thread')
    test_env = environment()
    test_env.set('TSAN_OPTIONS', 'halt_on_error=1')
    test_env.set('NIXL_PLUGIN_DIR', mocks_dep.get_variable('path'))

    mt_test_exe = executable('mt_test',
        sources : ['main.cpp', 'multi_threading.cpp'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        cpp_args : cpp_flags,
        dependencies : [nixl_dep, nixl_infra, cuda_dep, gtest_dep],
        link_with: [nixl_build_lib],
    )

    test('mt_test', mt_test_exe, is_parallel: false, env: test_env)
endif
