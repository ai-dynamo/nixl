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

gdrcopy_lib = cpp.find_library('gdrapi', required: false)
device_proxy_cpp_args = []
device_proxy_deps = [nixl_common_dep, device_allocator_interface]
if gdrcopy_lib.found() and cpp.has_header('gdrapi.h')
    device_proxy_cpp_args += ['-DHAVE_GDRCOPY']
    device_proxy_deps += [gdrcopy_lib]
else
    message('GDRCopy not found; device proxy will use host-mapped control-memory fallback')
endif

device_proxy_lib = library('nixl_device_proxy',
           ['proxy_config.cpp', 'proxy_config.h',
            'proxy_control_buffer.cpp', 'proxy_control_buffer.h',
            'proxy_protocol.h', 'proxy_backend_ops.h'],
           include_directories: [nixl_inc_dirs, utils_inc_dirs],
           cpp_args: device_proxy_cpp_args,
           dependencies: device_proxy_deps,
           install: true)

if get_option('install_headers')
    # Device code includes the wire contract as <device/proxy/proxy_protocol.h>.
    install_headers('proxy_protocol.h', install_dir: prefix_inc + '/device/proxy')
endif

# The proxy headers hand out nixlDeviceAllocator, so consumers link it too.
device_proxy_interface = declare_dependency(link_with: device_proxy_lib,
                                            dependencies: [device_allocator_interface])
