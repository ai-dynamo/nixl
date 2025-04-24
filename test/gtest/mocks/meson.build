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

mock_basic_plugin = shared_library('MOCK_BASIC', 'mock_basic_plugin.cpp',
               dependencies: [nixl_infra],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               link_with : [ucx_backend_lib],
               name_prefix: 'libplugin_',
               install: true,
               install_dir: plugin_install_dir)

mock_dram_sources = ['mock_dram_plugin.cpp', 'mock_dram_engine.cpp']
mock_dram_plugin = shared_library('MOCK_DRAM', mock_dram_sources,
               dependencies: [nixl_infra],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               link_with : [ucx_backend_lib],
               name_prefix: 'libplugin_',
               install: true,
               install_dir: plugin_install_dir)
run_command('sh', '-c',
            'echo "MOCK_BASIC=' + mock_basic_plugin.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                check: true
            )
run_command('sh', '-c',
            'echo "MOCK_DRAM=' + mock_dram_plugin.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                check: true
            )

source_root = meson.project_source_root()
mocks_dep = declare_dependency(variables : {'path' : meson.current_source_dir().split(source_root + '/')[1]})
