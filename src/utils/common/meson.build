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

tomlplusplus_dep = dependency('tomlplusplus', required: true, default_options: ['build_lib=false', 'compile_library=false'])

nixl_common_inc = include_directories('.')

nixl_common_deps = [
    absl_log_dep,
    absl_log_initialize_dep,
    absl_flat_hash_map_dep,
    absl_strings_dep,
    absl_status_dep,
    absl_statusor_dep,
    absl_synchronization_dep,
    tomlplusplus_dep,
    dependency('asio', required: true),
]

# vcs_tag() resolves the revision on every build, where run_command() froze it
# at configure time. '-dirty' marks an uncommitted tree; --exclude='*' reports
# the hash rather than a tag.
nixl_git_hash_h = vcs_tag(
    command: ['git', 'describe', '--always', '--abbrev=8', '--dirty', '--exclude=*'],
    input: 'nixl_git_hash.h.in',
    output: 'nixl_git_hash.h',
    fallback: 'unknown',
)

# Define a shared library for common utilities
nixl_common_lib = shared_library('nixl_common',
    'hw_info.cpp',
    'nixl_log.cpp',
    'uuid_v4.cpp',
    'configuration.cpp',
    'nixl_duration.cpp',
    nixl_git_hash_h,
    dependencies: nixl_common_deps,
    include_directories: [nixl_common_inc, nixl_inc_dirs],
    cpp_args: [
        '-DNIXL_VERSION="' + meson.project_version() + '"',
    ],
    install: true,
)

# Make the library and includes available to other parts of the build
nixl_common_dep = declare_dependency(
    include_directories: nixl_common_inc,
    link_with: nixl_common_lib,
    dependencies: nixl_common_deps
)
