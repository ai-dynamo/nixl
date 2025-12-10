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

# Use CMake subproject directly
cmake = import('cmake')
prometheus_found = false

# Try to get prometheus-cpp as a CMake subproject
# Pass CMake options directly to ensure they're applied
cmake_opts = cmake.subproject_options()
cmake_opts.add_cmake_defines({
    'ENABLE_PULL': 'ON',
    'ENABLE_PUSH': 'OFF',
    'ENABLE_COMPRESSION': 'OFF',
    'ENABLE_TESTING': 'OFF',
    'USE_THIRDPARTY_LIBRARIES': 'ON',
    'BUILD_SHARED_LIBS': 'ON',
    'CMAKE_POSITION_INDEPENDENT_CODE': 'ON',
    'CMAKE_C_FLAGS': '-Wno-unused-but-set-variable',
    'CMAKE_CXX_FLAGS': '-Wno-unused-but-set-variable'
})

prometheus_sub = cmake.subproject('prometheus-cpp', required: false, options: cmake_opts)
if prometheus_sub.found()
    # Get the CMake targets
    prometheus_core_dep = prometheus_sub.dependency('core')
    prometheus_pull_dep = prometheus_sub.dependency('pull')
    prometheus_found = true
endif

if prometheus_found
    # Prometheus Exporter Plugin
    prometheus_exporter_plugin = shared_library(
        'libtelemetry_exporter_prometheus',
        'prometheus_plugin.cpp',
        'prometheus_exporter.cpp',
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        dependencies: [nixl_infra, absl_log_dep, prometheus_core_dep, prometheus_pull_dep],
        install: true,
        install_dir: plugin_install_dir,
        name_prefix: '',
    )
    message('Building Prometheus telemetry exporter plugin')
else
    warning('Prometheus C++ library not found. Prometheus telemetry exporter will not be built.')
endif

