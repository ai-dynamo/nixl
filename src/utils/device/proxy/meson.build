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

device_proxy_lib = library('nixl_device_proxy',
           ['proxy_config.cpp', 'proxy_config.h'],
           include_directories: [nixl_inc_dirs, utils_inc_dirs],
           dependencies: [nixl_common_dep, nixl_infra, device_allocator_interface],
           install: true)

if get_option('install_headers')
    # Device code includes the wire contract as <device/proxy/proxy_protocol.h>.
    install_headers('proxy_protocol.h', install_dir: prefix_inc + '/device/proxy')
endif

# The proxy headers hand out nixlDeviceAllocator, so consumers link it too.
device_proxy_interface = declare_dependency(link_with: device_proxy_lib,
                                            dependencies: [device_allocator_interface])
