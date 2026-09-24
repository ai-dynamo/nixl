# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gtest_dep = dependency('gtest',
                       main: true,
                       required: false)

if gtest_dep.found()
  test_sources = ['consistency_test.cpp']
  if raw_cli_available
    test_sources += ['raw_cli_test.cpp', 'posix_integration_test.cpp']
  endif
  nixlbench_tests = executable(
    'nixlbench_tests',
    test_sources,
    include_directories: inc_dir,
    link_with: [utils_lib, nixlbench_runtimes],
    dependencies: deps + [gtest_dep],
    install: true,
  )

  nixlbench_test_env = environment()
  nixlbench_test_env.set('NIXLBENCH_BINARY', nixlbench_target.full_path())
  nixlbench_test_env.set('NIXL_PLUGIN_DIR', join_paths(nixl_lib_path, 'plugins'))
  test('nixlbench_tests', nixlbench_tests,
       env: nixlbench_test_env,
       timeout: 180,
       suite: 'nixlbench')
else
  message('GoogleTest not found. Skipping NIXLBench tests.')
endif
