# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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

rados_unit_test_dep = declare_dependency(
    sources: [
        'rados_config_test.cpp',
        'rados_engine_test.cpp',
    ],
    include_directories: [
        nixl_inc_dirs,
        '../../../../../src/plugins/rados',
    ],
    dependencies: [
        rados_dep,
    ],
    link_with: rados_backend_lib,
)

if 'RADOS' not in static_plugins
    rados_live_exe = executable(
        'rados_live',
        ['../../main.cpp', 'rados_discovery_test.cpp'],
        dependencies: [nixl_dep, nixl_infra, nixl_common_dep, gtest_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        cpp_args: ['-DRADOS_TEST_PLUGIN_DIR="' +
                   meson.project_build_root() / 'src/plugins/rados' + '"'],
        link_with: [nixl_build_lib],
        install: true,
    )

    test(
        'rados_live',
        rados_live_exe,
        depends: rados_backend_lib,
        timeout: 120,
    )
endif
