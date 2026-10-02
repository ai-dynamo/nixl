# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

if not rocm_dep.found()
    subdir_done()
endif

nixl_ais_mt_app = executable('nixl_ais_mt_test',
    'nixl_ais_mt_test.cpp',
    dependencies: [nixl_dep, nixl_infra, rocm_dep, nixl_common_dep],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, include_directories('..')],
    install: true)

# Fails on hipFile: the shared harness registers host DRAM buffers, and
# hipFileBufRegister takes device memory only (5013, hipFileHipMemoryTypeInvalid).
test('ais_mt_path_mode_smoke', nixl_ais_mt_app)
