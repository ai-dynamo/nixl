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

# ── Shared KV backend abstraction layer ────────────────────────────────────
#
# nixl_kv_base is a static utility library that provides the shared base class
# (nixlKVBackendBase), shared request handle (nixlKVReqH), and the standard
# completion callback (nixlKVXferCallback) for all KV-family backend plugins.
#
# Plugins link against nixl_kv_base_dep:
#
#   src/plugins/kv/redis/   REDIS plugin    links nixl_kv_base_dep
#   src/plugins/kv/wqskv/   WQSKV plugin    links nixl_kv_base_dep
#   src/plugins/kv/<next>/  future backends  links nixl_kv_base_dep
#
# Each plugin subdirectory is added via its own conditional subdir() call below.
# The subdir() calls are intentionally commented out until each PR lands.

kv_base_sources = [
    'kv_backend.cpp',
    'kv_backend.h',
    'kv_req_handle.h',
]

kv_base_compile_flags = []
if is_variable('compile_flags')
    kv_base_compile_flags = compile_flags
endif

nixl_kv_base_lib = static_library(
    'nixl_kv_base',
    kv_base_sources,
    dependencies: [nixl_infra, nixl_common_dep],
    cpp_args: kv_base_compile_flags,
    include_directories: [nixl_inc_dirs, utils_inc_dirs,
                          include_directories('.')],
    install: false,
)

# Dependency object consumed by plugin subdirectories.
nixl_kv_base_dep = declare_dependency(
    link_with: nixl_kv_base_lib,
    dependencies: [nixl_infra, nixl_common_dep],
    include_directories: [nixl_inc_dirs, utils_inc_dirs,
                          include_directories('.')],
)

# ── Plugin subdirectories ──────────────────────────────────────────────────
# Uncomment each line when the corresponding PR is merged and the plugin
# directory is moved under src/plugins/kv/.
#
# if enabled_plugins.get('REDIS')
#     subdir('redis')
# endif
#
# if enabled_plugins.get('WQSKV')
#     subdir('wqskv')
# endif
