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

azure_sources = [
    'azure_blob_backend.cpp',
    'azure_blob_backend.h',
    'azure_blob_plugin.cpp',
    'azure_blob_client.cpp',
    'azure_blob_client.h',
]

cpp = meson.get_compiler('cpp')
azure_core = cpp.find_library('azure-core', required: false)
azure_storage_common = cpp.find_library('azure-storage-common', required: false)
azure_storage_blobs = cpp.find_library('azure-storage-blobs', required: false)
azure_identity = cpp.find_library('azure-identity', required: false)

if azure_core.found() and azure_storage_common.found() and azure_storage_blobs.found() and azure_identity.found()
    plugin_deps += [azure_core, azure_storage_common, azure_storage_blobs, azure_identity]
else
    warning('Azure SDK dependencies not found, skipping AZURE_BLOB plugin build')
    subdir_done()
endif
plugin_deps += [dependency('asio', required: true)]

if 'AZURE_BLOB' in static_plugins
    azure_blob_backend_lib = static_library(
        'AZURE_BLOB',
        azure_sources,
        dependencies: plugin_deps,
        cpp_args: compile_defs + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    azure_blob_backend_lib = shared_library(
        'AZURE_BLOB',
        azure_sources,
        dependencies: plugin_deps,
        cpp_args: compile_defs + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "AZURE_BLOB=' + azure_blob_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

azure_blob_backend_interface = declare_dependency(link_with: azure_blob_backend_lib)
