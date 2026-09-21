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

# Entered only when TELEMETRY_DOCA is enabled and doca_telemetry_dep /
# doca_common_dep are found (gated in ../meson.build, mirroring the GPUNETIO
# backend in src/plugins/gpunetio/meson.build). doca_telemetry_dep and
# doca_common_dep come from the parent scope.
shared_library(
    'libtelemetry_exporter_doca',
    'doca_plugin.cpp',
    'doca_exporter.cpp',
    include_directories: [nixl_inc_dirs, utils_inc_dirs, include_directories('../common')],
    dependencies: [nixl_infra, nixl_common_dep, absl_log_dep, absl_strings_dep, doca_telemetry_dep, doca_common_dep],
    install: true,
    install_dir: plugin_install_dir,
    name_prefix: '',
)
message('Building DOCA telemetry exporter plugin')
