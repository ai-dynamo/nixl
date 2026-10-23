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

mooncake_batch_test = executable('mooncake_batch_test',
    'mooncake_batch_test.cpp',
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/mooncake'],
    dependencies: [nixl_dep, nixl_infra, nixl_common_dep, mooncake_backend_interface, thread_dep,
                   dependency('gtest', main: true)])

# Keep the capacity regression CPU-only and independent of RDMA hardware.
mooncake_batch_env = environment()
mooncake_batch_env.set('MC_FORCE_TCP', '1')
mooncake_batch_env.set('MC_DISABLE_HIP', '1')
mooncake_batch_env.set('NIXL_MOONCAKE_IP_ADDR', '127.0.0.1')
test('mooncake_batch_test', mooncake_batch_test,
     env: mooncake_batch_env, timeout: 120)
