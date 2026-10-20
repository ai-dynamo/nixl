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
    'CMAKE_LINK_DEPENDS_USE_LINKER': 'OFF',
    'CMAKE_C_FLAGS': '-Wno-unused-but-set-variable',
    'CMAKE_CXX_FLAGS': '-Wno-unused-but-set-variable'
})

prometheus_sub = cmake.subproject('prometheus-cpp', required: false, options: cmake_opts)
if prometheus_sub.found()
    # Get the CMake targets
    prometheus_core_dep = prometheus_sub.dependency('core')
    prometheus_pull_dep = prometheus_sub.dependency('pull')
    prometheus_found = true
else
    warning('Prometheus C++ library not found. Prometheus telemetry exporter will not be built.')
endif

subdir('prometheus_mp')

# DOCA telemetry exporter: an optional DOCA pkg-config dependency, gated exactly
# like the GPUNETIO backend (src/plugins/meson.build). Built by default when the
# DOCA SDK is present; skipped silently when it is absent; errors only if it is
# explicitly requested via -Denable_plugins=TELEMETRY_DOCA without the SDK; and
# can be turned off with -Ddisable_plugins=TELEMETRY_DOCA. The exporter links
# doca_common, so request it explicitly like GPUNETIO does.
doca_telemetry_dep = dependency('doca-telemetry-exporter', required: false)
doca_common_dep = dependency('doca-common', required: false)
if enabled_plugins.get('TELEMETRY_DOCA')
    if (not doca_telemetry_dep.found() or not doca_common_dep.found()) and is_explicit_enable
        error('TELEMETRY_DOCA plugin requested but doca-telemetry-exporter/doca-common dependency not found')
    elif doca_telemetry_dep.found() and doca_common_dep.found()
        subdir('doca')
    endif
endif
