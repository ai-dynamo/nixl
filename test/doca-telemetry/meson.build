# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Standalone DOCA telemetry tests. They intentionally avoid linking nixl core
# (no etcd/gRPC), so each process carries only the DOCA exporter's gRPC (no duplicate-flag
# clash with nixl's gRPC). Built under the same gate as the exporter plugin in
# src/plugins/telemetry/meson.build: the doca-telemetry-exporter/doca-common
# pkg-config dependencies plus the TELEMETRY_DOCA plugin toggle.
#  - doca_test:      exercises the raw DOCA Metrics API directly.
#  - doca_nixl_test: exercises the NIXL DOCA exporter (compiled in directly, so
#                    it pulls only nixl_infra/nixl_common - no gRPC).

if not gtest_dep.found()
    subdir_done()
endif

# DOCA discovery via pkg-config + the TELEMETRY_DOCA plugin toggle - identical to
# the exporter plugin in src/plugins/telemetry/meson.build, so the tests build
# under the exact same condition (and honor -Ddisable_plugins=TELEMETRY_DOCA).
doca_telemetry_dep = dependency('doca-telemetry-exporter', required: false)
doca_common_dep = dependency('doca-common', required: false)
build_doca_tests = enabled_plugins.get('TELEMETRY_DOCA') and doca_telemetry_dep.found() and doca_common_dep.found()

if build_doca_tests
    doca_test_exe = executable('doca_test',
        sources: ['telemetry_doca_test.cpp'],
        dependencies: [gtest_dep, doca_telemetry_dep, doca_common_dep],
        # DOCA telemetry metrics APIs are marked experimental (deprecated
        # attribute); mirror the DOCA sample Makefile and the plugin sources.
        cpp_args: ['-Wno-deprecated-declarations'],
        install: true,
    )

    test('doca_test', doca_test_exe, is_parallel: false)

    doca_exporter_dir = '../../src/plugins/telemetry/doca'
    doca_nixl_test_exe = executable('doca_nixl_test',
        sources: [
            'telemetry_doca_nixl_test.cpp',
            doca_exporter_dir / 'doca_exporter.cpp',
        ],
        include_directories: [
            nixl_inc_dirs,
            utils_inc_dirs,
            include_directories(doca_exporter_dir),
            include_directories('../../src/plugins/telemetry/common'),
        ],
        dependencies: [gtest_dep, doca_telemetry_dep, doca_common_dep, nixl_infra, nixl_common_dep, absl_log_dep, absl_strings_dep],
        cpp_args: ['-Wno-deprecated-declarations'],
        install: true,
    )

    test('doca_nixl_test', doca_nixl_test_exe, is_parallel: false)

    # Exporter volume/stress case, kept in its own source/binary so the sustained
    # high-volume run stays isolated from the functional exporter tests.
    doca_nixl_stress_test_exe = executable('doca_nixl_stress_test',
        sources: [
            'telemetry_doca_stress_test.cpp',
            doca_exporter_dir / 'doca_exporter.cpp',
        ],
        include_directories: [
            nixl_inc_dirs,
            utils_inc_dirs,
            include_directories(doca_exporter_dir),
            include_directories('../../src/plugins/telemetry/common'),
        ],
        dependencies: [gtest_dep, doca_telemetry_dep, doca_common_dep, nixl_infra, nixl_common_dep, absl_log_dep, absl_strings_dep],
        cpp_args: ['-Wno-deprecated-declarations'],
        install: true,
    )

    # scrapeUntilValue polls up to 15s; give generous headroom over Meson's 30s
    # default for the high-volume run plus setup/flush on loaded or sanitizer CI.
    test('doca_nixl_stress_test', doca_nixl_stress_test_exe, is_parallel: false, timeout: 120)
    message('Building standalone DOCA telemetry exporter tests')

    cmake = import('cmake')
    cmake_opts = cmake.subproject_options()
    cmake_opts.add_cmake_defines({
        'ENABLE_PULL': 'ON',
        'ENABLE_PUSH': 'OFF',
        'ENABLE_COMPRESSION': 'OFF',
        'ENABLE_TESTING': 'OFF',
        'USE_THIRDPARTY_LIBRARIES': 'ON',
        'BUILD_SHARED_LIBS': 'ON',
        'CMAKE_POSITION_INDEPENDENT_CODE': 'ON',
        'CMAKE_LINK_DEPENDS_USE_LINKER': 'OFF',
        'CMAKE_C_FLAGS': '-Wno-unused-but-set-variable',
        'CMAKE_CXX_FLAGS': '-Wno-unused-but-set-variable'
    })
    prometheus_sub = cmake.subproject('prometheus-cpp', required: false, options: cmake_opts)
    if prometheus_sub.found()
        prometheus_core_dep = prometheus_sub.dependency('core')
        prometheus_pull_dep = prometheus_sub.dependency('pull')
        prometheus_exporter_dir = '../../src/plugins/telemetry/prometheus'
        histogram_parity_test_exe = executable('histogram_parity_test',
            sources: [
                'telemetry_histogram_parity_test.cpp',
                doca_exporter_dir / 'doca_exporter.cpp',
                prometheus_exporter_dir / 'prometheus_exporter.cpp',
            ],
            include_directories: [
                nixl_inc_dirs,
                utils_inc_dirs,
                include_directories(doca_exporter_dir),
                include_directories(prometheus_exporter_dir),
                include_directories('../../src/plugins/telemetry/common'),
            ],
            dependencies: [
                gtest_dep,
                doca_telemetry_dep,
                doca_common_dep,
                nixl_infra,
                nixl_common_dep,
                absl_log_dep,
                prometheus_core_dep,
                prometheus_pull_dep,
            ],
            cpp_args: ['-Wno-deprecated-declarations'],
            install: true,
        )
        test('histogram_parity_test', histogram_parity_test_exe, is_parallel: false)
        message('Building Prometheus/DOCA histogram parity test')
    endif
endif
