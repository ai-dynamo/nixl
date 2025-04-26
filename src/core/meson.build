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

# Add dependency on the common utility library which brings in logging deps
nixl_lib_deps = [nixl_infra, serdes_interface, stream_interface, dl_dep, nixl_common_dep]

if 'UCX' in static_plugins
    nixl_lib_deps += [ ucx_backend_interface, cuda_dep ]
endif

if 'UCX_MO' in static_plugins
    nixl_lib_deps += [ ucx_mo_backend_interface, cuda_dep ]
endif


disable_gds_backend = get_option('disable_gds_backend')
if not disable_gds_backend and cuda_dep.found()
    nixl_lib_deps += [ gds_backend_interface, cuda_dep ]
endif

nixl_lib = library('nixl',
                   'nixl_agent.cpp',
                   'nixl_plugin_manager.cpp',
                   'nixl_listener.cpp',
                   include_directories: [ nixl_inc_dirs, utils_inc_dirs ],
                   dependencies: nixl_lib_deps,
                   install: true)

nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: nixl_inc_dirs)
