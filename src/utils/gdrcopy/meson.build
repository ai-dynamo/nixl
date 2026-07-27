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

gdrcopy_utils_deps = [cuda_dep, nixl_common_dep]
if have_gdrcopy
    gdrcopy_utils_deps += [gdrcopy_dep]
endif

gdrcopy_utils_lib = library(
    'nixl_gdrcopy_utils',
    'nixl_gdr_buffer.cpp',
    'nixl_gdr_buffer.h',
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: gdrcopy_utils_deps,
    install: true,
)

gdrcopy_utils_interface = declare_dependency(link_with: gdrcopy_utils_lib)
