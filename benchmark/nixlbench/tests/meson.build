# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gtest_dep = dependency('gtest',
                       main: true,
                       required: true,
                       fallback: ['gtest', 'gtest_main_dep'])

nixlbench_tests = executable(
  'nixlbench_tests',
  ['raw_cli_test.cpp', 'posix_integration_test.cpp'],
  include_directories: inc_dir,
  link_with: [utils_lib, nixlbench_runtimes],
  dependencies: deps + [gtest_dep],
)

nixlbench_test_env = environment()
nixlbench_test_env.set('NIXLBENCH_BINARY', nixlbench_target.full_path())
nixlbench_test_env.set('NIXL_PLUGIN_DIR', join_paths(nixl_lib_path, 'plugins'))
test('nixlbench_tests', nixlbench_tests,
     env: nixlbench_test_env,
     timeout: 180,
     suite: 'nixlbench')
