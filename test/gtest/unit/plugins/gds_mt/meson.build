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

gds_mt_unit_test_dep = declare_dependency(
    sources: [
        'gds_mt_offset_test.cpp',
    ],
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        include_directories('../../../../../src/plugins/gds_mt'),
    ],
    dependencies: [
        nixl_dep,
        nixl_infra,
        nixl_common_dep,
        cuda_dep,
        cufile_dep,
        taskflow_proj,
        file_utils_interface,
        gds_mt_backend_interface,
    ],
)
