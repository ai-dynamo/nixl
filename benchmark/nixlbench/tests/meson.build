# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

memos_key_tests = executable(
  'memos_key_tests',
  'memos_key_test.cpp',
  include_directories: inc_dir,
)
test('memos_key_tests', memos_key_tests, suite: 'nixlbench')

memos_key_pool_tests = executable(
  'memos_key_pool_tests',
  'memos_key_pool_test.cpp',
  include_directories: inc_dir,
)
test('memos_key_pool_tests', memos_key_pool_tests, suite: 'nixlbench')

memos_pipeline_tests = executable(
  'memos_pipeline_tests',
  'memos_pipeline_test.cpp',
  include_directories: inc_dir,
)
test('memos_pipeline_tests', memos_pipeline_tests, suite: 'nixlbench')

memos_buffer_tests = executable(
  'memos_buffer_tests',
  'memos_buffer_test.cpp',
  include_directories: inc_dir,
)
test('memos_buffer_tests', memos_buffer_tests, suite: 'nixlbench')

memos_duration_tests = executable(
  'memos_duration_tests',
  'memos_duration_test.cpp',
  include_directories: inc_dir,
)
test('memos_duration_tests', memos_duration_tests, suite: 'nixlbench')

memos_cleanup_tests = executable(
  'memos_cleanup_tests',
  'memos_cleanup_test.cpp',
  include_directories: inc_dir,
)
test('memos_cleanup_tests', memos_cleanup_tests, suite: 'nixlbench')

memos_reporting_tests = executable(
  'memos_reporting_tests',
  'memos_reporting_test.cpp',
  include_directories: inc_dir,
  link_with: [utils_lib, nixlbench_runtimes],
  dependencies: deps,
)
test('memos_reporting_tests', memos_reporting_tests, suite: 'nixlbench')

gtest_dep = dependency('gtest',
                       main: true,
                       required: false)

if gtest_dep.found()
  test_sources = ['print_stats_test.cpp']
  if raw_cli_available
    test_sources += ['raw_cli_test.cpp', 'posix_integration_test.cpp']
  endif
  nixlbench_tests = executable(
    'nixlbench_tests',
    test_sources,
    include_directories: inc_dir,
    link_with: [utils_lib, nixlbench_runtimes],
    dependencies: deps + [gtest_dep],
    link_language: 'cpp', # Let users pass -Wl,... in LDFLAGS (nvcc rejects it)
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
