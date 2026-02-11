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

if not cuda_dep.found()
    warning('CUDA not found, skipping nixl_ep build')
    subdir_done()
endif

py = import('python').find_installation('python3', pure: false)

# Check if PyTorch is available
torch_check = run_command(py, '-c', 'import torch', check: false)
if torch_check.returncode() != 0
    warning('PyTorch not found, skipping nixl_ep build')
    subdir_done()
endif

# Get PyTorch base directory from Python
torch_dir = run_command(py, '-c', 'import torch, os; print(os.path.dirname(torch.__file__))', check: true).stdout().strip()
torch_include_base = join_paths(torch_dir, 'include')
torch_lib_dir = join_paths(torch_dir, 'lib')

# All Torch include directories
torch_inc_dirs = [
    include_directories(torch_include_base),
    include_directories(join_paths(torch_include_base, 'torch')),
    include_directories(join_paths(torch_include_base, 'torch', 'csrc', 'api', 'include')),
    include_directories(join_paths(torch_include_base, 'ATen')),
]

# Create Torch dependency with includes and libraries
torch_dep = declare_dependency(
    include_directories: torch_inc_dirs,
    link_args: ['-L' + torch_lib_dir, '-ltorch', '-ltorch_python', '-ltorch_cuda']
)

pybind_dep = dependency('pybind11', required: false)
if not pybind_dep.found()
    warning('pybind11 not found, skipping nixl_ep build')
    subdir_done()
endif

nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: nixl_inc_dirs)
nixl_lib_dir = join_paths(meson.project_build_root(), 'src', 'core')

ucx_build_deps = []
if ucx_dep.found()
    ucx_build_deps = [ucx_dep]
endif


nixl_ep_sources = [
    'csrc/nixl_ep.cpp',
    'csrc/kernels/nixl_ep.cu',
]

nixl_ep_inc_dirs = [
    nixl_inc_dirs,
    nixl_gpu_inc_dirs,
    include_directories('csrc'),
    include_directories('csrc/kernels'),
    torch_inc_dirs,
]

doca_build_deps = []
if doca_gpunetio_dep.found()
    doca_build_deps = [doca_gpunetio_dep]
endif

nixl_ep_cpp_args = [
    '-DHAVE_CUDA',
    '-DTORCH_EXTENSION_NAME=nixl_ep_cpp',
    '-Wno-deprecated-declarations',
    '-Wno-unused-variable',
    '-Wno-sign-compare',
    '-Wno-reorder',
    '-Wno-attributes',
]

nixl_ep_cuda_args = [
    '-DHAVE_CUDA',
    '-DTORCH_EXTENSION_NAME=nixl_ep_cpp',
    '--expt-relaxed-constexpr',  # Allow calling constexpr __host__ functions from __device__ functions
    '-arch=sm_90',  # Only compile for sm90 (overrides global -gencode flags)
    '--ptxas-options=--register-usage-level=10',  # Allow more register usage (matches setup.py)
    '-Xcompiler', '-Wno-deprecated-declarations',
    '-Xcompiler', '-Wno-unused-variable',
    '-Xcompiler', '-Wno-sign-compare',
    '-Xcompiler', '-Wno-reorder',
    '-Xcompiler', '-Wno-attributes',
]

topk_idx_bits = meson.get_external_property('topk_idx_bits', '64')
nixl_ep_cpp_args += ['-DTOPK_IDX_BITS=' + topk_idx_bits]
nixl_ep_cuda_args += ['-DTOPK_IDX_BITS=' + topk_idx_bits]

# NVCC 12.9 workaround for bug (https://nvbugspro.nvidia.com/bug/5595631)
# This workaround is needed because NVCC 12.9 has a bug which fails UCX compilation
nvcc = meson.get_compiler('cuda')
if nvcc.version().version_compare('>=12.9') and nvcc.version().version_compare('<12.10')
    nixl_ep_cpp_args += ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE']
    nixl_ep_cuda_args += ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE']
endif

nixl_ep_rpath = torch_lib_dir
nixl_ep_rpath += ':' + nixl_lib_dir
nixl_ep_rpath += ':' + join_paths(nixl_lib_dir, 'core')
nixl_ep_rpath += ':' + join_paths(nixl_lib_dir, 'plugins')

# For now, nixl ep cannot be built with -G due to register usage limits
nixl_ep_override_options = []
if get_option('buildtype') == 'debug'
    nixl_ep_override_options = ['optimization=3']
    nixl_ep_cpp_args += ['-g']
endif

enable_fast_debug = get_option('buildtype') != 'release'

if enable_fast_debug
    nixl_ep_cuda_args += ['-DENABLE_FAST_DEBUG']
endif

nixl_ep_install_rpath = join_paths(get_option('prefix'), get_option('libdir'))
nixl_ep_install_rpath += ':' + join_paths(get_option('prefix'), get_option('libdir'), 'plugins')
nixl_ep_install_rpath += ':' + torch_lib_dir

nixl_ep_ext = py.extension_module('nixl_ep_cpp',
    nixl_ep_sources,
    subdir: 'nixl_ep',
    dependencies: [
        nixl_dep,
        pybind_dep,
        cuda_dep,
        torch_dep,
    ] + ucx_build_deps + doca_build_deps,
    include_directories: nixl_ep_inc_dirs,
    cpp_args: nixl_ep_cpp_args,
    cuda_args: nixl_ep_cuda_args,
    link_with: [nixl_lib],
    build_rpath: nixl_ep_rpath,
    install_rpath: nixl_ep_install_rpath,
    override_options: nixl_ep_override_options,
    install: true,
)

nixl_ep_py_files = files(
    'nixl_ep/__init__.py',
    'nixl_ep/buffer.py',
    'nixl_ep/utils.py',
)

custom_target('nixl_ep_py_copy',
    output: 'nixl_ep_py.stamp',
    input: nixl_ep_py_files,
    command: [
        'bash', '-c',
        'cp -r @0@/nixl_ep @1@/ && cp @2@ @1@/nixl_ep/ && touch @3@'.format(
            meson.current_source_dir(),
            meson.current_build_dir(),
            nixl_ep_ext.full_path(),
            join_paths(meson.current_build_dir(), 'nixl_ep_py.stamp')
        )
    ],
    depends: nixl_ep_ext,
    build_by_default: true,
)

py.install_sources(
    'nixl_ep/__init__.py',
    'nixl_ep/buffer.py',
    'nixl_ep/utils.py',
    subdir: 'nixl_ep',
    pure: false,
)
