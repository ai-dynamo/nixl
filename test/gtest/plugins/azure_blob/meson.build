# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Microsoft Corporation.
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
cpp = meson.get_compiler('cpp')
azure_core = cpp.find_library('azure-core', required: false)
azure_storage_common = cpp.find_library('azure-storage-common', required: false)
azure_storage_blobs = cpp.find_library('azure-storage-blobs', required: false)
azure_identity = cpp.find_library('azure-identity', required: false)

if not azure_core.found() or not azure_storage_common.found() or not azure_storage_blobs.found() or not azure_identity.found()
    message('Azure SDK dependencies not found, skipping azure_blob_gtest build')
    subdir_done()
endif

azure_blob_test_exe = executable('azure_blob_gtest',
    sources : [
        'azure_blob_test.cpp',
        '../../main.cpp',
        '../../common.cpp',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, plugins_inc_dirs, '.'],
    dependencies : [
        nixl_dep,
        gtest_dep,
        absl_strings_dep,
        absl_time_dep,
        azure_core,
        azure_storage_common,
        azure_storage_blobs,
        azure_identity,
        plugin_deps,
        azure_blob_backend_interface
    ],
    link_with: [nixl_build_lib],
    install : true
)

test('azure_blob_gtest', azure_blob_test_exe)
