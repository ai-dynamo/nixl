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

nixl_example = executable('nixl_example',
        'nixl_example.cpp',
        dependencies: [nixl_dep, nixl_infra, nixl_common_deps, nixl_test_utils_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        link_with: [serdes_lib],
        install: true)

if etcd_dep.found()
    etcd_example = executable('nixl_etcd_example',
               'nixl_etcd_example.cpp',
               dependencies: [nixl_dep, nixl_infra, nixl_common_deps, nixl_test_utils_dep],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               link_with: [serdes_lib],
               install: true)
endif

telemetry_reader = executable('telemetry_reader',
           'telemetry_reader.cpp',
           dependencies: [nixl_dep, nixl_common_deps],
           include_directories: [nixl_inc_dirs, utils_inc_dirs],
           install: true)
