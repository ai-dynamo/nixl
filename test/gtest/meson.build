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
absl_strings_dep = abseil_proj.get_variable('absl_strings_dep')
absl_time_dep = abseil_proj.get_variable('absl_time_dep')

if not gtest_dep.found()
    message('GTest not found, skipping gtest build')
    subdir_done()
endif

gmock_dep = dependency('gmock', required : false)

if not gmock_dep.found()
    message('GMock not found, skipping gtest build')
    subdir_done()
endif

gtest_inc_dirs = include_directories('.')

subdir('mocks')
subdir('unit')
subdir('plugins')

if ucx_gpu_device_api_available
    subdir('device_api')
endif

plugin_dirs_arg = '--tests_plugin_dirs=' + mocks_dep.get_variable('path')

cpp_flags = []

if cuda_dep.found()
    cpp_flags += '-DHAVE_CUDA'
    cuda_dependencies = [cuda_dep]
else
    cuda_dependencies = []
endif

if get_option('test_all_plugins')
  cpp_flags+='-DTEST_ALL_PLUGINS'
endif

# for loading plugins from build directory
cpp_flags += '-DBUILD_DIR="' + meson.project_build_root() + '"'

gtest_sources = [
    'main.cpp',
    'mocks/gmock_engine.cpp',
    'plugin_manager.cpp',
    'error_handling.cpp',
    'test_transfer.cpp',
    'metadata_exchange.cpp',
    'common.cpp',
    'query_mem.cpp',
    'telemetry_test.cpp'
    ]

if ucx_gpu_device_api_available
    gtest_sources += device_api_test_sources
    device_api_inc = [nixl_gpu_inc_dirs, include_directories('device_api')]
    device_api_dep = ucx_dep
else
    device_api_inc = []
    device_api_dep = []
endif

test_exe = executable('gtest',
    sources : gtest_sources,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, device_api_inc],
    cpp_args : cpp_flags,
    dependencies : [nixl_dep, nixl_common_dep, cuda_dependencies, device_api_dep, gtest_dep, gmock_dep, absl_strings_dep, absl_time_dep, file_utils_interface],
    link_with: [nixl_build_lib],
    install : true
)

test('gtest', test_exe, args: [plugin_dirs_arg])

if get_option('b_sanitize').split(',').contains('thread')
    test_env = environment()
    test_env.set('TSAN_OPTIONS', 'halt_on_error=1')
    test_env.set('NIXL_PLUGIN_DIR', mocks_dep.get_variable('path'))

    mt_test_exe = executable('mt_test',
        sources : ['main.cpp', 'mocks/gmock_engine.cpp', 'multi_threading.cpp'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        cpp_args : cpp_flags,
        dependencies : [nixl_dep, nixl_infra, cuda_dependencies, gtest_dep, gmock_dep],
        link_with: [nixl_build_lib],
    )

    test('mt_test', mt_test_exe, is_parallel: false, env: test_env)
endif
