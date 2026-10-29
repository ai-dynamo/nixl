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

libfabric_test_dep = [
    nixl_infra,
    nixl_common_dep,
    libfabric_dep,
    serdes_interface,
    libfabric_utils_dep,
    libfabric_backend_interface,
    dependency('dl', required: false),
]

libfabric_backend_test_dep = [nixl_dep, nixl_infra, thread_dep] + libfabric_test_dep + nixl_test_utils_dep
libfabric_test_include_directories = [nixl_inc_dirs, utils_inc_dirs]

libfabric_backend_test = executable('libfabric_backend_test',
        'libfabric_backend_test.cpp',
        dependencies: libfabric_backend_test_dep,
        include_directories: libfabric_test_include_directories,
        cpp_args: ['-DHAVE_LIBFABRIC'],
        install: true)

test('libfabric_backend_test', libfabric_backend_test, timeout: 180)
