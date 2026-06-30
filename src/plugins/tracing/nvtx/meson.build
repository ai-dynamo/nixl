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

# Reached only when nixl_trace_nvtx_enabled is set (root meson.build), which also
# provides nvtx_inc_dep.
shared_library(
    'libtrace_backend_nvtx',
    'nvtx_plugin.cpp',
    'nvtx_backend.cpp',
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    # dl_dep: nvtx3 injection thunks call dlopen/dlsym/dlclose, so with --no-undefined
    # libdl must be linked where dl* isn't merged into libc (e.g. the manylinux wheel build).
    dependencies: [nixl_infra, nixl_common_dep, absl_log_dep, nvtx_inc_dep, dl_dep],
    install: true,
    install_dir: plugin_install_dir,
    name_prefix: '',
)
message('Building NVTX trace backend plugin')
