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
    shared_library(
        'libtelemetry_exporter_doca',
        'doca_plugin.cpp',
        'doca_exporter.cpp',
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        dependencies: [nixl_infra, absl_log_dep, doca_dep],
        install: true,
        install_dir: plugin_install_dir,
        name_prefix: '',
    )
    message('Building DOCA telemetry exporter plugin')
else
    warning('DOCA Telemetry Exporter headers not found. DOCA telemetry exporter will not be built.')
endif
