# SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Google LLC
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

# GPUDirect TCPXO Plugin Configuration

cpp = meson.get_compiler('cpp')
cmake = import('cmake')

tcpxo_plugin_deps = [nixl_infra, serdes_interface, nixl_common_dep]
tcpxo_lib_name = 'TCPXO'
tcpxo_inc_dirs = [nixl_inc_dirs, utils_inc_dirs]
lib_prefix = 'libplugin_' # Custom prefix for plugin libraries

compile_flags = []
# Add CUDA support if available
if not get_option('disable_cuda') and cuda_dep.found()
    tcpxo_plugin_deps += [cuda_dep]
    compile_flags += ['-DHAVE_CUDA']
endif

proto_args = [
    '--proto_path=@CURRENT_SOURCE_DIR@',
]

if get_option('disable_rxdm_dxs')
    compile_flags += ['-DTCPXO_STUB_RXDM_DXS']
else
    # Get the execution root from bazel
    googleapis_path = tcpxo_rxdm_proj.get_variable('googleapis_path')
    googleapis_gen_path = tcpxo_rxdm_proj.get_variable('googleapis_gen_path')

    # Link WebRTC library
    webrtc_prefix = '/usr/local/webrtc'
    webrtc_lib = cpp.find_library(
        'webrtc',
        dirs: [webrtc_prefix],
        required: true,
    )
    webrtc_dep = declare_dependency(dependencies: [webrtc_lib])

    # Get dependencies from RxDM and DXS
    tcpxo_rxdm_dxs_dep = tcpxo_rxdm_proj.get_variable(
        'tcpxo_rxdm_dxs_dep',
    )
    tcpxo_plugin_deps += [
        tcpxo_rxdm_dxs_dep,
        webrtc_dep,
    ]

    proto_args += [
        '--proto_path=' + googleapis_path,
        '--proto_path=' + googleapis_gen_path,
    ]
endif

proto_args += [
    '--cpp_out=@BUILD_DIR@',
    '@INPUT@',
]

# Build protobuf files
protoc = find_program('protoc', required: true)

if tcpxo_lib_name in static_plugins
    tcpxo_plugin_deps += [dependency('protobuf', required: true, static: true)]
else
    tcpxo_plugin_deps += [dependency('protobuf', required: true, static: false)]
endif

protoc_gen = generator(
    protoc,
    output: ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
    arguments: proto_args,
)
proto_files = protoc_gen.process('control_channel.proto')

tcpxo_sources = [
    'tcpxo_backend.cpp',
    'tcpxo_plugin.cpp',
    'tcpxo_common.cpp',
    'tcpxo_nic_common.cpp',
    'tcpxo_nixl_memory_metadata.cpp',
    'nixl_cuda/cuda_common.cpp',
    'host_connection.cpp',
    'control_channel.cpp',
    'dxs_endpoint.cpp',
    proto_files,
]

if tcpxo_lib_name in static_plugins
    tcpxo_backend_lib = static_library(
        tcpxo_lib_name,
        tcpxo_sources,
        dependencies: tcpxo_plugin_deps,
        include_directories: tcpxo_inc_dirs,
        install: false,
        cpp_args: compile_flags,
        name_prefix: lib_prefix,
    )
else
    tcpxo_backend_lib = shared_library(
        tcpxo_lib_name,
        tcpxo_sources,
        dependencies: tcpxo_plugin_deps,
        include_directories: tcpxo_inc_dirs,
        install: true,
        cpp_args: compile_flags + ['-fPIC'],
        name_prefix: lib_prefix,
        install_dir: plugin_install_dir,
    )

    if get_option('buildtype') == 'debug'
        run_command(
            'sh',
            '-c',
            'echo "TCPXO=' + tcpxo_backend_lib.full_path() + '" >> ' + plugin_build_dir +
            '/pluginlist',
            check: true,
        )
    endif
endif

tcpxo_backend_interface = declare_dependency(
    link_with: tcpxo_backend_lib,
    dependencies: tcpxo_plugin_deps,
)
