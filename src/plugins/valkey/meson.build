# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if not enabled_plugins.get('VALKEY')
    subdir_done()
endif

valkey_sources = [
    'valkey_backend.cpp',
    'valkey_backend.h',
    'valkey_client.cpp',
    'valkey_client.h',
    'valkey_plugin.cpp',
]

# From valkey-glide (v2.6.0-rc2 or later, for RDMA), under glide_path:
#   lib/libglide_ffi.a              cd ffi && cargo build --release --features rdma
#   include/glide_ffi.h             cbindgen --config cbindgen.toml --crate glide-ffi --lang c
#   proto/connection_request.proto  glide-core/src/protobuf/connection_request.proto
# The plugin builds the connection request with code generated from that .proto
# (as GLIDE's C++ client does), so it also needs protoc and libprotobuf.
# libfabric itself is loaded at runtime.
valkey_cc = meson.get_compiler('cpp')
valkey_fs = import('fs')
valkey_glide_path = get_option('glide_path')
valkey_glide_inc = valkey_glide_path != '' ? [valkey_glide_path / 'include'] : []
valkey_glide_lib = valkey_cc.find_library('glide_ffi',
    dirs: valkey_glide_path != '' ? [valkey_glide_path / 'lib'] : [],
    static: true,
    required: false)
valkey_proto = valkey_glide_path / 'proto' / 'connection_request.proto'
valkey_protoc = find_program('protoc', required: false)
valkey_protobuf_dep = dependency('protobuf', required: false)
valkey_deps_found = valkey_glide_lib.found() and valkey_cc.has_header('glide_ffi.h',
    args: valkey_glide_inc.length() > 0 ? ['-I' + valkey_glide_inc[0]] : []) and \
    valkey_fs.is_file(valkey_proto) and valkey_protoc.found() and valkey_protobuf_dep.found()

if not valkey_deps_found
    if is_explicit_enable or 'VALKEY' in static_plugins
        error('VALKEY plugin requested but its dependencies were not found: lib/libglide_ffi.a, include/glide_ffi.h and proto/connection_request.proto under glide_path (valkey-glide ffi built with --features rdma), protoc and libprotobuf.')
    endif
    warning('VALKEY plugin dependencies not found, skipping VALKEY plugin build (see src/plugins/valkey/README.md)')
    subdir_done()
endif
valkey_glide_dep = declare_dependency(
    dependencies: valkey_glide_lib,
    include_directories: include_directories(valkey_glide_inc, is_system: true))

# Generated code is compiled on its own, without the project's -Werror.
valkey_proto_src = custom_target('valkey_connection_request_pb',
    input: valkey_proto,
    output: ['connection_request.pb.cc', 'connection_request.pb.h'],
    command: [valkey_protoc, '--proto_path=' + (valkey_glide_path / 'proto'),
              '--cpp_out=@OUTDIR@', '@INPUT@'])
valkey_proto_lib = static_library('valkey_connection_request',
    valkey_proto_src,
    dependencies: valkey_protobuf_dep,
    override_options: ['werror=false', 'warning_level=0'],
    pic: true)
valkey_proto_dep = declare_dependency(
    sources: valkey_proto_src[1],
    link_with: valkey_proto_lib,
    dependencies: valkey_protobuf_dep)

valkey_plugin_deps = [
    nixl_infra,
    nixl_common_dep,
    valkey_glide_dep,
    valkey_proto_dep,
    dl_dep,
    valkey_cc.find_library('m'),
    thread_dep,
]
# Keep GLIDE's symbols private to the library that links it (the plugin, or
# libnixl for a static plugin), so they never bind to another GLIDE in the
# process (e.g. the Python client's libglide_ffi.so).
valkey_link_args = ['-Wl,--exclude-libs,libglide_ffi.a']

if 'VALKEY' in static_plugins
    valkey_backend_lib = static_library(
        'VALKEY',
        valkey_sources,
        dependencies: valkey_plugin_deps,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')
else
    valkey_backend_lib = shared_library(
        'VALKEY',
        valkey_sources,
        dependencies: valkey_plugin_deps,
        cpp_args: ['-fPIC'],
        link_args: valkey_link_args,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "VALKEY=' + valkey_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

valkey_backend_interface = declare_dependency(
    link_with: valkey_backend_lib,
    dependencies: valkey_plugin_deps,
    link_args: valkey_link_args,
)
