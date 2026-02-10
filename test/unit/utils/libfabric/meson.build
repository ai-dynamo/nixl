# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
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

libfabric_utils_dep = [ libfabric_dep, nixl_common_deps, serdes_interface ]
libfabric_test_cpp_args = []

if cuda_dep.found()
    libfabric_utils_dep += [ cuda_dep ]
    libfabric_test_cpp_args += [ '-DCUDA_FOUND' ]
endif

if get_option('buildtype') != 'release'

    # mock function calls for testing
    libfabric_test_link_args = [
        '-Wl,--wrap=fi_getinfo', 
        '-Wl,--wrap=numa_max_node', 
        '-Wl,--wrap=numa_num_configured_nodes', 
        '-Wl,--wrap=numa_distance',
        '-Wl,--wrap=get_mempolicy',
        '-Wl,--wrap=fi_fabric' ]

    libfabric_topology_test_bin = executable('libfabric_topology_test',
               'libfabric_topology_test.cpp',
               dependencies: libfabric_utils_dep,
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               link_with: libfabric_utils_lib,
               cpp_args: libfabric_test_cpp_args,
               link_args: libfabric_test_link_args,
               install: true)

    # install topology test files
    install_subdir('topo', strip_directory : true, install_dir : get_option('bindir'))

endif
