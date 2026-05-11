gtest_dep = dependency('gtest',
                       main: true,
                       required: false,
                       fallback: ['gtest', 'gtest_main_dep'])

if gtest_dep.found()
  nixlbench_unit_tests = executable(
    'nixlbench_unit_tests',
    [
      'benchmark_config_test.cpp',
      'benchmark_config_adapter_test.cpp',
      'backend_option_inventory_test.cpp',
      'nixl_backend_params_test.cpp',
      'benchmark_runner_config_test.cpp',
      'benchmark_executor_test.cpp',
      'benchmark_memory_allocator_test.cpp',
      'transfer_descriptor_strategy_test.cpp',
      'worker_config_test.cpp',
      'xfer_bench_config_test.cpp',
    ],
    include_directories: inc_dir,
    link_with: [nixlbench_runtimes, benchmark_framework_lib, worker_lib, nixl_worker_lib],
    dependencies: [gtest_dep, utils_dep, benchmark_framework_dep],
  )

  test('nixlbench_unit_tests', nixlbench_unit_tests)
else
  warning('GoogleTest dependency not found; nixlbench unit tests are disabled')
endif
