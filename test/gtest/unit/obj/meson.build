# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

obj_unit_test_dep = declare_dependency(
    sources: [
        'obj.cpp',
    ],
    include_directories: [
        nixl_inc_dirs,
        '../../../../src/plugins/obj',
    ],
    dependencies: [
        aws_s3.partial_dependency(compile_args: false, includes: true, link_args: true, links: true),
        dependency('asio', required: true),
    ],
    link_with: obj_backend_lib,
)
