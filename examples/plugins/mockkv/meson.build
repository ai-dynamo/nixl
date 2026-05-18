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

mockkv_sources = [
    'mockkv_backend.cpp',
    'mockkv_backend.h',
    'mockkv_plugin.cpp',
]

mockkv_example_plugin = shared_library(
    'MOCKKV',
    mockkv_sources,
    dependencies: [nixl_infra, nixl_common_dep],
    cpp_args: ['-fPIC'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    install: false,
    name_prefix: 'libplugin_')
