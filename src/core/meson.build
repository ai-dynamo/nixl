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

nixl_lib_deps = [nixl_infra, serdes_interface, stream_interface, dl_dep, nixl_common_dep, thread_dep]

if etcd_dep.found()
    nixl_lib_deps += [ etcd_dep ]
endif

if 'UCX' in static_plugins
    nixl_lib_deps += [ ucx_backend_interface, cuda_dep ]
endif

if 'UCX_MO' in static_plugins
    nixl_lib_deps += [ ucx_mo_backend_interface, cuda_dep ]
endif

if 'POSIX' in static_plugins
    nixl_lib_deps += [ posix_backend_interface ]
endif

if 'OBJ' in static_plugins
    nixl_lib_deps += [ obj_backend_interface ]
endif

disable_gds_backend = get_option('disable_gds_backend')
if not disable_gds_backend and 'GDS' in static_plugins
    nixl_lib_deps += [ gds_backend_interface, cuda_dep ]
endif

cc = meson.get_compiler('cpp')
libtransfer_engine = cc.find_library('transfer_engine', required: false)
disable_mooncake_backend = get_option('disable_mooncake_backend')
if libtransfer_engine.found() and not disable_mooncake_backend and 'Mooncake' in static_plugins
    nixl_lib_deps += [ mooncake_backend_interface, cuda_dep ]
endif

nixl_lib = library('nixl',
                   'nixl_agent.cpp',
                   'nixl_plugin_manager.cpp',
                   'nixl_listener.cpp',
                   include_directories: [ nixl_inc_dirs, utils_inc_dirs ],
                   link_args: ['-lstdc++fs'],
                   dependencies: nixl_lib_deps,
                   install: true)

nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: nixl_inc_dirs)
