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

# Find cargo
cargo = find_program('cargo', required: true)

# Build the Rust library
rust_lib = custom_target('nixl-sys',
  output: ['libnixl_sys.a'],
  input: ['Cargo.toml', 'build.rs', 'wrapper.h', 'wrapper.cpp'] + files('src/lib.rs'),
  command: [cargo, 'build', '--release', '--manifest-path', '@INPUT0@'],
  build_by_default: true,
  install: true,
  install_dir: get_option('libdir')
)

# Install the Rust library header
install_headers('wrapper.h', install_dir: get_option('includedir'))

# Create a dependency for other parts of the project
nixl_rust_dep = declare_dependency(
  link_with: rust_lib,
  include_directories: include_directories('.'),
  dependencies: [nixl_dep]  # Assuming nixl_dep is defined in the main meson.build
)

