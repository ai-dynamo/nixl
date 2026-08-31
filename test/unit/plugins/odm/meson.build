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

disable_odm_backend = get_option('disable_odm_backend')
if disable_odm_backend or not cuda_dep.found()
    subdir_done()
endif

odm_nixl_test = executable('odm_nixl_test',
    'odm_nixl_test.cpp',
    dependencies: [nixl_dep, nixl_infra, cuda_dep, nixl_common_dep, nixl_test_utils_dep],
    include_directories: [nixl_inc_dirs, utils_inc_dirs,
                          include_directories('../../../../src/plugins/odm')],
    install: true)

# Hardware-only: run manually with `meson test --suite odm-hw` when /dev/odm0 is present.
if get_option('enable_odm_hw_tests')
    test('odm_plugin_test', odm_nixl_test, suite: 'odm-hw')
endif
