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

nixl_build_lib = library('nixl_build',
                        'nixl_descriptors.cpp',
                        'nixl_memory_section.cpp',
                        include_directories: [ nixl_inc_dirs, utils_inc_dirs ],
                        dependencies: [serdes_interface, nixl_common_dep],
                        install: true)

nixl_infra = declare_dependency(link_with: nixl_build_lib)

# Test utilities library that can depend on nixl_dep (created after nixl_dep is defined)
# This will be defined in a separate meson file to avoid circular dependencies
