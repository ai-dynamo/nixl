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

redis_unit_test_dep = declare_dependency(
    sources: [
        'redis.cpp',
    ],
    include_directories: [
        nixl_inc_dirs, utils_inc_dirs,
        '../../../../src/plugins/redis',
    ],
    dependencies: [
        redis_backend_interface,
    ],
)

# Keep scheduling interposition local to this executable and compile the real client.
redis_event_loop_test = executable(
    'redis_event_loop_test',
    sources: ['../main.cpp', 'redis_event_loop.cpp', '../../../../src/plugins/redis/redis_client.cpp'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/redis'],
    cpp_args: redis_compile_flags,
    dependencies: [redis_plugin_deps, gtest_dep],
    link_args: ['-Wl,--wrap=event_base_once', '-Wl,--wrap=event_base_dispatch', '-Wl,--wrap=event_base_loop'],
)

test('redis_event_loop', redis_event_loop_test, env: sanitizer_env, suite: 'sanitizer')
