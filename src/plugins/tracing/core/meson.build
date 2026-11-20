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

trace_core_inc_dirs = [nixl_inc_dirs, utils_inc_dirs, plugins_inc_dirs]
trace_core_deps = [serdes_interface, nixl_common_dep, absl_log_dep]

trace_core_lib = static_library('trace_core',
    'scope.cpp',
    'trace_context.cpp',
    'trace_sink.cpp',
    'tracer.cpp',
    include_directories: trace_core_inc_dirs,
    dependencies: trace_core_deps,
    gnu_symbol_visibility: 'hidden',
    pic: true,
    install: false)

trace_core_dep = declare_dependency(link_with: trace_core_lib,
                                    include_directories: plugins_inc_dirs,
                                    dependencies: trace_core_deps)

shared_library('libtrace_core',
    'trace_core_plugin.cpp',
    link_whole: trace_core_lib,
    include_directories: trace_core_inc_dirs,
    dependencies: trace_core_deps,
    install: true,
    install_dir: plugin_install_dir,
    name_prefix: '')
