# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Amazon.com, Inc. and affiliates.
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

# The test recompiles the engine sources into its own binary (see below), which would collide with
# the copy already inside libnixl when the plugin is linked statically.
if get_option('buildtype') != 'release' and 'LIBFABRIC' not in static_plugins

    libfabric_plugin_dir = '../../../../src/plugins/libfabric'

    libfabric_backend_test_dep = [
        nixl_dep,
        nixl_infra,
        nixl_common_dep,
        libfabric_dep,
        serdes_interface,
        libfabric_utils_dep,
        absl_log_dep,
        thread_dep,
        dependency('dl', required: true),
    ]

    if cuda_dep.found()
        libfabric_backend_test_dep += [cuda_dep]
    endif

    libfabric_backend_test_inc = [
        nixl_inc_dirs,
        utils_inc_dirs,
        # The engine under test and the fabric stubs shared with the utils layer tests.
        include_directories(libfabric_plugin_dir),
        include_directories('../../utils/libfabric'),
    ]

    # The plugin sources are compiled into the test binary rather than linked from the plugin
    # library: --wrap only redirects calls in objects the executable links directly, so the engine
    # has to be part of this link to see the mocked fabric. libfabric_plugin.cpp is left out because
    # the plugin entry points are not exercised here.
    libfabric_agent_index_test_bin = executable('libfabric_agent_index_test',
               sources: [
                   'libfabric_agent_index_test.cpp',
                   libfabric_plugin_dir / 'libfabric_backend.cpp',
                   libfabric_plugin_dir / 'libfabric_connection.cpp',
                   libfabric_plugin_dir / 'libfabric_handshake.cpp',
               ],
               dependencies: libfabric_backend_test_dep,
               include_directories: libfabric_backend_test_inc,
               cpp_args: ['-DHAVE_LIBFABRIC'],
               link_args: [
                   '-Wl,--wrap=fi_getinfo',
                   '-Wl,--wrap=fi_fabric',
                   '-Wl,--wrap=numa_max_node',
                   '-Wl,--wrap=numa_num_configured_nodes',
               ],
               install: true)

    test('libfabric_agent_index_test', libfabric_agent_index_test_bin)

endif
