# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

gmock_dep = dependency('gmock', required : false)

if not gmock_dep.found()
    message('GMock not found, skipping gtest build')
    subdir_done()
endif

gtest_inc_dirs = include_directories('.')

# Sanitizer runtime configuration. Built once and shared by every instrumented
# test (tagged with the 'sanitizer' suite) so `meson test --suite sanitizer`
# runs them with halt-on-error semantics and the third-party suppression lists.
# When -Dsanitizer=none the environment stays empty, so tagging tests with the
# suite is harmless for normal builds.
sanitizer_opts = get_option('sanitizer').split(',')
sanitizer_env = environment()
sanitizer_supp_dir = meson.current_source_dir() / 'sanitizer'
if sanitizer_opts.contains('address')
    sanitizer_env.set('ASAN_OPTIONS', 'halt_on_error=1:abort_on_error=1:detect_leaks=1')
    sanitizer_env.set('LSAN_OPTIONS', 'suppressions=' + sanitizer_supp_dir / 'lsan.supp')
endif
if sanitizer_opts.contains('undefined')
    sanitizer_env.set('UBSAN_OPTIONS', 'halt_on_error=1:print_stacktrace=1:suppressions=' + sanitizer_supp_dir / 'ubsan.supp')
endif
if sanitizer_opts.contains('thread')
    sanitizer_env.set('TSAN_OPTIONS', 'halt_on_error=1:suppressions=' + sanitizer_supp_dir / 'tsan.supp')
endif

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

if libfabric_dep.found()
    cpp_flags += '-DHAVE_LIBFABRIC'
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
    'telemetry_test.cpp',
    'telemetry_prometheus_test.cpp',
    'telemetry_benchmark.cpp',
    'nixl_duration_test.cpp',
    'configuration.cpp'
    ]

if ucx_gpu_device_api_available
    gtest_sources += device_api_test_sources
    device_api_inc = [nixl_gpu_inc_dirs, include_directories('device_api')]
    device_api_dep = [ucx_dep, doca_gpunetio_dep]
else
    device_api_inc = []
    device_api_dep = []
endif

if ucx_dep.found() and is_variable('ucx_backend_interface')
    gtest_sources += 'hw_warning_test.cpp'
    ucx_hw_warning_dep = [ucx_backend_interface, ucx_dep]
else
    ucx_hw_warning_dep = []
endif

test_exe = executable('gtest',
    sources : gtest_sources,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, device_api_inc, include_directories('../doca-telemetry'), include_directories('../../src/plugins/telemetry/common')],
    cpp_args : cpp_flags,
    dependencies : [nixl_dep, nixl_common_dep, cuda_dependencies, device_api_dep, gtest_dep, gmock_dep, absl_strings_dep, absl_time_dep, ucx_hw_warning_dep, file_utils_interface],
    link_with: [nixl_build_lib],
    install : true
)

# MetadataExchangeTestFixture rapidly creates and tears down comm-thread agents,
# which trips the GCC 13.3 libsanitizer thread-registry crash (a flaky SIGSEGV in
# the ASan/TSan runtime on thread create/destroy -- a toolchain bug, not a NIXL
# issue). Exclude it from the sanitizer run until the CI toolchain is bumped; it
# still runs in the normal (non-sanitizer) CI.
gtest_args = [plugin_dirs_arg]
if get_option('sanitizer') != 'none'
    gtest_args += '--gtest_filter=-MetadataExchangeTestFixture.*'
endif
test('gtest', test_exe, args: gtest_args, env: sanitizer_env, suite: 'sanitizer')

# Telemetry hot-path microbenchmark (NVBug 5876558). The cases are DISABLED_ so
# they never run in the normal 'gtest' suite; this dedicated entry opts them in.
# Run on demand: `ctest --test-dir build -R telemetry_benchmark -V`. Excluded
# from the sanitizer suite (timing under ASan/TSan is meaningless).
if get_option('sanitizer') == 'none'
    test('telemetry_benchmark',
         test_exe,
         args: [plugin_dirs_arg,
                '--gtest_also_run_disabled_tests',
                '--gtest_filter=telemetryBenchmark.*'],
         is_parallel: false,
         timeout: 300)
endif

# Best-effort NVTX capture: when nsys is available, profile the tracing tests
# and write a .nsys-rep that CI can publish as an artifact. Skips (exit 77) when
# profiling is denied, so it never fails CI on its own.
if nixl_trace_nvtx_enabled
    nsys_prog = find_program('nsys', required: false)
    if nsys_prog.found()
        nsys_wrapper = find_program(meson.current_source_dir() / 'run_nsys_tracing.sh')
        test('tracing_nsys',
             nsys_wrapper,
             args: [nsys_prog.full_path(), test_exe, meson.current_build_dir(), plugin_dirs_arg],
             is_parallel: false,
             timeout: 600)
    endif
endif

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
