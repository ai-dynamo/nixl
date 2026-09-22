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

gds_mt_interior_pointer_test = executable(
    'gds_mt_interior_pointer_test',
    sources: [
        'gds_mt_interior_pointer_test.cpp',
        '../../../main.cpp',
        '../../../common.cpp',
    ],
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        include_directories('../../../'),
        include_directories('../../../../../src/plugins/gds_mt'),
    ],
    cpp_args: [
        '-DHAVE_CUDA',
        '-DBUILD_DIR="' + meson.project_build_root() + '"',
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
        gtest_dep,
        gmock_dep,
        absl_strings_dep,
        absl_time_dep,
    ],
    link_with: [nixl_build_lib],
    install: true,
)

gds_mt_interior_pointer_test_env = environment()
gds_mt_interior_pointer_test_env.set(
    'NIXL_PLUGIN_DIR',
    meson.project_build_root() / 'src/plugins/gds_mt',
)
gds_mt_interior_pointer_test_env.set(
    'NIXL_GDS_TEST_DIR',
    meson.current_build_dir(),
)

test(
    'gds_mt_interior_pointer_integration',
    gds_mt_interior_pointer_test,
    env: gds_mt_interior_pointer_test_env,
    is_parallel: false,
    timeout: 600,
    suite: 'gds_mt_integration',
)
