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

# Standalone DOCA telemetry exporter test. It links the DOCA telemetry exporter
# directly and intentionally pulls in NO nixl libraries, so the process carries
# only CollectX's gRPC (no duplicate-flag clash with nixl's gRPC). Built only
# when the DOCA telemetry headers are present, mirroring the plugin gate in
# src/plugins/telemetry/doca/meson.build.

if not gtest_dep.found()
    subdir_done()
endif

# DOCA discovery - identical to src/plugins/telemetry/doca/meson.build.
doca_found = false
doca_base_path = '/opt/mellanox/doca'
doca_inc_path = doca_base_path / 'include'
doca_arch = host_machine.cpu_family() + '-linux-gnu'
doca_lib_path = doca_base_path / 'lib' / doca_arch

if cpp.has_header('doca_telemetry_exporter.h', args: '-I' + doca_inc_path)
    doca_dep = declare_dependency(
        include_directories: include_directories(doca_inc_path, is_system: true),
        link_args: [
            '-L' + doca_lib_path,
            '-ldoca_telemetry_exporter',
            '-ldoca_common',
            '-Wl,-rpath,' + doca_lib_path
        ]
    )
    doca_found = true
endif

if doca_found
    doca_test_exe = executable('doca_test',
        sources: ['telemetry_doca_test.cpp'],
        dependencies: [gtest_dep, doca_dep],
        # DOCA telemetry metrics APIs are marked experimental (deprecated
        # attribute); mirror the DOCA sample Makefile and the plugin sources.
        cpp_args: ['-Wno-deprecated-declarations'],
        install: true,
    )

    test('doca_test', doca_test_exe, is_parallel: false)
    message('Building standalone DOCA telemetry exporter test')
else
    warning('DOCA Telemetry Exporter headers not found. DOCA telemetry exporter test will not be built.')
endif
