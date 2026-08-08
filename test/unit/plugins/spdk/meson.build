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

# When the plugin is baked into libnixl (static_plugins), it (and the SPDK
# closure) is already inside nixl_dep; linking spdk_backend_interface as well
# would whole-archive a second copy into the test and duplicate SPDK's global
# state. So depend on the plugin interface only in the dynamic-plugin build.
spdk_test_deps = [nixl_dep, nixl_infra]
if 'SPDK' not in static_plugins
    spdk_test_deps += spdk_backend_interface
endif

nixl_spdk_test = executable(
    'nixl_spdk_test',
    'nixl_spdk_test.cpp',
    dependencies: spdk_test_deps,
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    link_args: ['-L' + spdk_library_path,
                '-Wl,-rpath,' + spdk_library_path,
                '-Wl,-rpath-link,' + spdk_library_path],
    install: true)

test('spdk_plugin_test',
     nixl_spdk_test,
     env: {'LD_LIBRARY_PATH': spdk_library_path})

nixl_spdk_runtime_test = executable(
    'nixl_spdk_runtime_test',
    'nixl_spdk_runtime_test.cpp',
    dependencies: spdk_test_deps,
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    link_args: ['-L' + spdk_library_path,
                '-Wl,-rpath,' + spdk_library_path,
                '-Wl,-rpath-link,' + spdk_library_path],
    install: true)

test('spdk_plugin_runtime_test',
     nixl_spdk_runtime_test,
     env: {'LD_LIBRARY_PATH': spdk_library_path},
     timeout: 120)
