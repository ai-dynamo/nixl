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

if not enable_dte
    error('DOCA Telemetry Exporter plugin requested but is disabled')
endif

# DOCA Telemetry Exporter Plugin
csv_exporter_plugin = shared_library(
    'libtelemetry_exporter_dte',
    'dte_plugin.cpp',
    'dte_exporter.cpp',
    cpp_args: compile_defs + ['-Wno-deprecated-declarations'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [nixl_infra, absl_log_dep, dte_dep],
    install: true,
    install_dir: plugin_install_dir,
    name_prefix: '',
)
