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

# Multiprocess metric-state store: a small, prometheus-cpp-free component (mmap
# per-process store) shared by the prometheus_mp plugin and its unit tests. Built
# unconditionally so the CTest coverage runs even where prometheus-cpp is absent;
# the plugin shared library (which needs prometheus-cpp) is added in a later step.
prometheus_mp_inc_dirs = include_directories('.')

telemetry_mp_store_lib = static_library(
    'nixl_telemetry_mp_store',
    'mp_store.cpp',
    include_directories: [nixl_inc_dirs, utils_inc_dirs, prometheus_mp_inc_dirs],
    dependencies: [nixl_common_dep],
    install: false,
    pic: true,
)

nixl_telemetry_mp_store_dep = declare_dependency(
    link_with: telemetry_mp_store_lib,
    include_directories: prometheus_mp_inc_dirs,
    dependencies: [nixl_common_dep],
)

# Scrape-time collector: converts peer store snapshots into prometheus-cpp metric
# families. Needs prometheus-cpp (core), so it is gated on it being available
# (same condition as the prometheus plugin, set in the sibling subdir above).
if prometheus_found
    telemetry_mp_collector_lib = static_library(
        'nixl_telemetry_mp_collector',
        'mp_collector.cpp',
        include_directories: [nixl_inc_dirs, utils_inc_dirs, prometheus_mp_inc_dirs],
        dependencies: [nixl_common_dep, prometheus_core_dep, nixl_telemetry_mp_store_dep],
        install: false,
        pic: true,
    )

    nixl_telemetry_mp_collector_dep = declare_dependency(
        link_with: telemetry_mp_collector_lib,
        include_directories: prometheus_mp_inc_dirs,
        dependencies: [nixl_common_dep, prometheus_core_dep, nixl_telemetry_mp_store_dep],
    )
endif
