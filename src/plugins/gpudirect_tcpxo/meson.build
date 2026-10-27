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

# GPUDirect TCPXO Plugin Configuration

cpp = meson.get_compiler('cpp')
cmake = import('cmake')

tcpxo_plugin_deps = [nixl_infra, serdes_interface, nixl_common_dep]
tcpxo_lib_name = 'TCPXO'
tcpxo_inc_dirs = [nixl_inc_dirs, utils_inc_dirs]
lib_prefix = 'libplugin_' # Custom prefix for plugin libraries
tcpxo_is_explicit = is_explicit_enable or (tcpxo_lib_name in static_plugins)

compile_flags = []
# Add CUDA support if available
if not get_option('disable_gpudirect_tcpxo_cuda') and cuda_dep.found()
    tcpxo_plugin_deps += [cuda_dep]
    compile_flags += ['-DHAVE_CUDA']
endif

proto_args = [
    '--proto_path=@CURRENT_SOURCE_DIR@',
]

if get_option('disable_rxdm_dxs')
    compile_flags += ['-DTCPXO_STUB_RXDM_DXS']
else
    # Link WebRTC library
    webrtc_prefix = get_option('webrtc_path')
    if webrtc_prefix == ''
        webrtc_prefix = '/usr/local/webrtc'
    endif
    webrtc_lib = cpp.find_library(
        'webrtc',
        dirs: [webrtc_prefix],
        required: false,
    )
    if not webrtc_lib.found()
        if tcpxo_is_explicit
            error(
                'TCPXO plugin requested but WebRTC library not found in ' + webrtc_prefix,
            )
        endif
        message(
            'WebRTC library not found in ' + webrtc_prefix + ', skipping TCPXO plugin',
        )
        subdir_done()
    endif
    webrtc_dep = declare_dependency(dependencies: [webrtc_lib])

    # Get dependencies from RxDM and DXS
    rxdm_dxs_prefix = get_option('rxdm_dxs_path')
    rxdm_dxs_bazel_bin = rxdm_dxs_prefix / 'bazel-bin'

    rxdm_dxs_lib = cpp.find_library(
        'rxdm_dxs',
        dirs: [rxdm_dxs_bazel_bin / 'buffer_mgmt_daemon'],
        required: false,
    )
    if not rxdm_dxs_lib.found()
        if tcpxo_is_explicit
            error(
                'TCPXO plugin requested but rxdm_dxs library not found in ' + (
                    rxdm_dxs_bazel_bin / 'buffer_mgmt_daemon'
                ),
            )
        endif
        message('rxdm_dxs library not found, skipping TCPXO plugin')
        subdir_done()
    endif

    rxdm_dxs_inc_dirs = [
        include_directories(rxdm_dxs_prefix),
        include_directories(rxdm_dxs_bazel_bin),
    ]

    googleapis_find = run_command(
        'sh',
        '-c',
        'ls -d ' + rxdm_dxs_bazel_bin + '/external/googleapis* 2>/dev/null | head -n 1',
        check: false,
    )
    googleapis_gen_path = googleapis_find.stdout().strip()
    if googleapis_gen_path != '' and fs.is_dir(googleapis_gen_path)
        rxdm_dxs_inc_dirs += include_directories(googleapis_gen_path)
    endif

    tcpxo_rxdm_dxs_dep = declare_dependency(
        include_directories: rxdm_dxs_inc_dirs,
        dependencies: [rxdm_dxs_lib],
    )

    tcpxo_plugin_deps += [
        tcpxo_rxdm_dxs_dep,
        webrtc_dep,
    ]
endif

proto_args += [
    '--cpp_out=@BUILD_DIR@',
    '@INPUT@',
]

# Build protobuf files
protoc = find_program('protoc', required: false)
if not protoc.found()
    if tcpxo_is_explicit
        error('TCPXO plugin requested but protoc program not found')
    endif
    message('protoc not found, skipping TCPXO plugin')
    subdir_done()
endif

is_static_tcpxo = tcpxo_lib_name in static_plugins
protobuf_dep = dependency(
    'protobuf',
    required: false,
    static: is_static_tcpxo,
)
if not protobuf_dep.found()
    if tcpxo_is_explicit
        error('TCPXO plugin requested but protobuf dependency not found')
    endif
    message('protobuf dependency not found, skipping TCPXO plugin')
    subdir_done()
endif
tcpxo_plugin_deps += [protobuf_dep]

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
