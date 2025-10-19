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

buffer_exporter_plugin = shared_library(
    'libtelemetry_exporter_buffer',
    'buffer_plugin.cpp',
    'buffer_exporter.cpp',
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [nixl_infra, absl_log_dep],
    install: true,
    install_dir: get_option('libdir') / 'nixl' / 'telemetry_exporters',
    name_prefix: '',
)
message('Building shared buffer telemetry exporter plugin')

# Add to telemetry exporters list for debug builds
if get_option('buildtype') == 'debug'
    run_command('sh', '-c',
                'echo "buffer=' + buffer_exporter_plugin.full_path() + '" >> ' + plugin_build_dir + '/telemetry_exporters_list',
                check: true
    )
endif

