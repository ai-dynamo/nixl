# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
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

# Pure (SPDK-free) unit tests for the RADOS_NKV key derivation. Compiles the
# key source directly so it builds and runs without the SPDK KV shim.
rados_nkv_plugin_dir = '../../../../src/plugins/rados-nkv'
rados_nkv_unit_test_dep = declare_dependency(
    sources: [
        'test_rados_nkv_key.cpp',
        rados_nkv_plugin_dir / 'rados_nkv_key.cpp',
    ],
    include_directories: [
        nixl_inc_dirs, utils_inc_dirs,
        rados_nkv_plugin_dir,
    ],
)
