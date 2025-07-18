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

# Get required Abseil dependencies from the subproject
absl_log_dep = abseil_proj.get_variable('absl_log_dep')
absl_strings_dep = abseil_proj.get_variable('absl_strings_dep')
absl_status_dep = abseil_proj.get_variable('absl_status_dep')
absl_strings_dep = abseil_proj.get_variable('absl_strings_dep')
absl_synchronization_dep = abseil_proj.get_variable('absl_synchronization_dep')

nixl_common_inc = include_directories('.')

nixl_common_deps = [
    absl_log_dep,
    absl_strings_dep,
    absl_status_dep,
    absl_strings_dep,
    absl_synchronization_dep,
]

# Define a shared library for common utilities
nixl_common_lib = shared_library('nixl_common',
    'nixl_log.cpp',
    'uuid_v4.cpp',
    dependencies: nixl_common_deps,
    include_directories: nixl_common_inc,
    install: true,
)

# Make the library and includes available to other parts of the build
nixl_common_dep = declare_dependency(
    include_directories: nixl_common_inc,
    link_with: nixl_common_lib,
    dependencies: nixl_common_deps
)
