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

rustc = find_program('rustc', required: false)
cargo = find_program('cargo', required: false)


nixl_include_dir = include_directories('../../api/cpp', '../../infra', '../../core')
nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: nixl_include_dir)

wrapper_sources = ['wrapper.cpp']
wrapper_lib = static_library('nixl_wrapper',
    sources: wrapper_sources,
    include_directories: nixl_include_dir,
    link_with: nixl_lib,
    dependencies: [nixl_dep]
  )

if rustc.found() and cargo.found()
  rust_lib = custom_target(
    'rust',
    command: [
      cargo, 'build', '--release',
      '--manifest-path', join_paths(meson.project_source_root(), 'src', 'bindings', 'rust', 'Cargo.toml')
    ],
    input: [],
    output: 'libnixl-sys.so',
    depends: [wrapper_lib],
    install: false,
    install_dir: get_option('libdir'),
    env: [
      'NIXL_PREFIX=' + get_option('prefix')
    ]
  )
else
  nixl_rust_bindings_lib = disabler()
endif
