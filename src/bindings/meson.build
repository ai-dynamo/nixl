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

subdir('python')

if get_option('build_cute')
  subdir('cute')
  # A second install target keeps the same bitcode in the CUDA backend wheel.
  foreach arch, bitcode : cute_bitcodes
    custom_target('cute-wheel-bitcode-' + arch,
      input: bitcode,
      output: '@PLAINNAME@',
      command: [py, '-c', 'import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])',
                '@INPUT@', '@OUTPUT@'],
      install: true,
      install_dir: py.get_install_dir() / cuda_wheel_dir / 'lib',
    )
  endforeach
endif

# Shared library exposing the nixl C API (nixl_capi_* symbols).
# Always built so LD_PRELOAD can override stub symbols at runtime when
# downstream Rust binaries were compiled without nixl present.
nixl_capi_inc = include_directories('../api/cpp', '../infra', '../core')
shared_library('nixl_capi',
    sources: ['rust/wrapper.cpp'],
    include_directories: [nixl_capi_inc],
    link_with: [nixl_lib, nixl_build_lib],
    dependencies: [declare_dependency(link_with: [nixl_lib, nixl_build_lib], include_directories: nixl_capi_inc)],
    install: true
  )

if get_option('rust')
  subdir('rust')
endif
