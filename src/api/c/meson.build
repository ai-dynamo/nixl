# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Libraries exposing the nixl C API (nixl_capi_* symbols). The shared flavor is
# built whatever default_library is set to, because the Rust bindings'
# stubs.cpp dlopens libnixl_capi.so by name when a downstream Rust binary was
# compiled without nixl present.
nixl_capi_inc = include_directories('.', '../cpp', '../../infra', '../../core')

# Limit the dynamic export set to exactly the nixl_capi_* contract; hidden
# visibility alone still leaks weak C++ template instantiations.
nixl_capi_map = meson.current_source_dir() / 'nixl_capi.map'
nixl_capi_link_args = []
if cpp.has_link_argument('-Wl,--version-script=' + nixl_capi_map)
    nixl_capi_link_args += ['-Wl,--version-script=' + nixl_capi_map]
endif

nixl_capi_libs = both_libraries('nixl_capi',
    sources: ['nixl_capi.cpp'],
    include_directories: [nixl_capi_inc],
    link_with: [nixl_lib, nixl_build_lib],
    gnu_symbol_visibility: 'inlineshidden',
    link_args: nixl_capi_link_args,
    link_depends: nixl_capi_map,
    version: meson.project_version(),
    soversion: '0',
    install: true,
    # A consumer's own RUNPATH is not searched when resolving this library's
    # dependency on libnixl.so, so it needs one of its own.
    install_rpath: join_paths(get_option('prefix'), get_option('libdir'))
  )

pkg = import('pkgconfig')
pkg.generate(nixl_capi_libs.get_shared_lib(),
    name: 'nixl_capi',
    description: 'C API for the NVIDIA Inference Xfer Library (NIXL)',
    url: 'https://github.com/ai-dynamo/nixl',
    # Static consumers must also link the underlying NIXL C++ libraries.
    libraries_private: [nixl_lib, nixl_build_lib]
  )
