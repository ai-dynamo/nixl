gtest_dep = dependency('gtest',
                       main: true,
                       required: false,
                       fallback: ['gtest', 'gtest_main_dep'])

if gtest_dep.found()
  nixlbench_unit_tests = executable(
    'nixlbench_unit_tests',
    [
      'benchmark_config_test.cpp',
      'xfer_bench_config_test.cpp',
    ],
    include_directories: inc_dir,
    link_with: [nixlbench_runtimes],
    dependencies: [gtest_dep, utils_dep],
  )

  test('nixlbench_unit_tests', nixlbench_unit_tests)
else
  warning('GoogleTest dependency not found; nixlbench unit tests are disabled')
endif
