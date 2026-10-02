# SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
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

gtest_dep = dependency('gtest', version: '>=1.11.0', required: false)

if not gtest_dep.found()
    message('GTest not found, skipping tcpxo test build')
    subdir_done()
endif

gmock_dep = dependency('gmock', required: false)

if not gmock_dep.found()
    message('GMock not found, skipping tcpxo test build')
    subdir_done()
endif

deps = [
    nixl_dep,
    nixl_infra,
    nixl_common_dep,
    tcpxo_backend_interface,
    gtest_dep,
    gmock_dep,
]

if not get_option('disable_gpudirect_tcpxo_cuda') and cuda_dep.found()
    deps += [cuda_dep]
    cpp_args = ['-DHAVE_CUDA']
else
    cpp_args = ['-UHAVE_CUDA']
endif

if get_option('disable_rxdm_dxs')
    cpp_args += ['-DTCPXO_STUB_RXDM_DXS']
endif

tcpxo_params_test = executable(
    'tcpxo_params_test',
    'tcpxo_params_test.cpp',
    dependencies: deps,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        '../../../../src/plugins/gpudirect_tcpxo',
    ],
    cpp_args: cpp_args,
    install: true,
)

test('tcpxo_params_test', tcpxo_params_test)

tcpxo_validation_test = executable(
    'tcpxo_validation_test',
    'tcpxo_validation_test.cpp',
    dependencies: deps,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        '../../../../src/plugins/gpudirect_tcpxo',
    ],
    cpp_args: cpp_args,
    install: true,
)

test('tcpxo_validation_test', tcpxo_validation_test)

tcpxo_backend_test = executable(
    'tcpxo_backend_test',
    'tcpxo_backend_test.cpp',
    dependencies: deps,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        '../../../../src/plugins/gpudirect_tcpxo',
    ],
    cpp_args: cpp_args,
    install: true,
)

test('tcpxo_backend_test', tcpxo_backend_test)

mpmc_queue_test = executable(
    'mpmc_queue_test',
    'mpmc_queue_test.cpp',
    dependencies: deps,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        '../../../../src/plugins/gpudirect_tcpxo',
    ],
    cpp_args: cpp_args,
    install: true,
)

test('mpmc_queue_test', mpmc_queue_test)

control_channel_test = executable(
    'control_channel_test',
    'control_channel_test.cpp',
    dependencies: deps,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        '../../../../src/plugins/gpudirect_tcpxo',
    ],
    cpp_args: cpp_args,
    install: true,
)

test('control_channel_test', control_channel_test)

host_connection_test = executable(
    'host_connection_test',
    'host_connection_test.cpp',
    dependencies: deps,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        '../../../../src/plugins/gpudirect_tcpxo',
    ],
    cpp_args: cpp_args,
    install: true,
)

test('host_connection_test', host_connection_test)

tcpxo_nixl_memory_metadata_test = executable(
    'tcpxo_nixl_memory_metadata_test',
    'tcpxo_nixl_memory_metadata_test.cpp',
    dependencies: deps,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        '../../../../src/plugins/gpudirect_tcpxo',
    ],
    cpp_args: cpp_args,
    install: true,
)

test('tcpxo_nixl_memory_metadata_test', tcpxo_nixl_memory_metadata_test)

dxs_endpoint_test = executable(
    'dxs_endpoint_test',
    'dxs_endpoint_test.cpp',
    dependencies: deps,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        '../../../../src/plugins/gpudirect_tcpxo',
    ],
    cpp_args: cpp_args,
    install: true,
)

test('dxs_endpoint_test', dxs_endpoint_test)

trap_add_test = find_program(meson.current_source_dir() / 'trap_add_test.sh')
test('trap_add_test', trap_add_test)
