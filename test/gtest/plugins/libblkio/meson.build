# SPDX-FileCopyrightText: Copyright (c) 2026 Dell Technologies Inc. All rights reserved.
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

libblkio_gtest_exe = executable(
    'libblkio_gtest',
    sources: [
        'libblkio_plugin_test.cpp',
        '../../unit/main.cpp',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, gtest_inc_dirs],
    dependencies: [
        nixl_dep,
        nixl_infra,
        gtest_dep,
        absl_log_dep,
    ],
    link_with: [nixl_build_lib],
    install: true,
)

test('libblkio_gtest', libblkio_gtest_exe,
    env: ['NIXL_PLUGIN_DIR=' + meson.project_build_root() + '/src/plugins/libblkio'],
)
