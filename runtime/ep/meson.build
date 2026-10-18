# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if not cuda_dep.found()
    warning('CUDA not found, skipping NIXL-EP V2 build')
    subdir_done()
endif

if not ucx_dep.found()
    warning('UCX not found, skipping NIXL-EP V2 build')
    subdir_done()
endif

if not ucx_gpu_device_api_available
    warning('UCX GPU device API is unavailable, skipping NIXL-EP V2 build')
    subdir_done()
endif

py = import('python').find_installation('python3', pure: false)
torch_check = run_command(py, '-c', 'import torch', check: false)
if torch_check.returncode() != 0
    warning('PyTorch not found, skipping NIXL-EP V2 build')
    subdir_done()
endif

pybind_dep = dependency('pybind11', required: false)
if not pybind_dep.found()
    warning('pybind11 not found, skipping NIXL-EP V2 build')
    subdir_done()
endif

if not cpp.has_header('format', args: ['-std=c++20'])
    warning('C++20 <format> is unavailable, skipping NIXL-EP V2 build')
    subdir_done()
endif

if cuda_inc_path != ''
    v2_cuda_include_root = cuda_inc_path
else
    v2_cuda_include_root = run_command(
        py, '-c',
        'import os, sys; print(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[1]))), "include"))',
        nvcc_prog.full_path(),
        check: true,
    ).stdout().strip()
endif
v2_cccl_include_root = join_paths(v2_cuda_include_root, 'cccl')
if not fs.is_dir(v2_cccl_include_root)
    warning('CUDA CCCL headers not found at ' + v2_cccl_include_root + ', skipping NIXL-EP V2 build')
    subdir_done()
endif

nvcc = meson.get_compiler('cuda')
have_v2_ucx_device_api = nvcc.compiles('''
        #include <ucp/api/device/ucp_device_impl.h>
        #if !defined(UCT_CUDA_IPC_SUPPORTED) || !UCT_CUDA_IPC_SUPPORTED
        #error "UCX CUDA IPC support is required"
        #endif
        int main() { return 0; }
    ''',
    dependencies: [ucx_dep, cuda_dep],
    args: nvcc_flags + ['-I' + v2_cccl_include_root],
    name: 'NIXL-EP V2 UCX device API with CUDA IPC')
if not have_v2_ucx_device_api
    warning('UCX device API with CUDA IPC is unavailable, skipping NIXL-EP V2 build')
    subdir_done()
endif

if ucx_path != ''
    v2_ucx_include_root = ucx_inc_path
else
    v2_ucx_include_root = ucx_dep.get_variable(pkgconfig: 'includedir', default_value: '')
endif
if v2_ucx_include_root == '' or not fs.is_dir(v2_ucx_include_root)
    warning('Cannot locate the include directory for the linked UCX, skipping NIXL-EP V2 build')
    subdir_done()
endif

torch_dir = run_command(
    py, '-c', 'import os, torch; print(os.path.dirname(torch.__file__))',
    check: true,
).stdout().strip()
torch_include_base = join_paths(torch_dir, 'include')
torch_lib_dir = join_paths(torch_dir, 'lib')
torch_inc_dirs = [
    include_directories(torch_include_base),
    include_directories(join_paths(torch_include_base, 'torch')),
    include_directories(join_paths(torch_include_base, 'torch', 'csrc', 'api', 'include')),
    include_directories(join_paths(torch_include_base, 'ATen')),
]
torch_dep = declare_dependency(
    include_directories: torch_inc_dirs,
    link_args: ['-L' + torch_lib_dir, '-ltorch', '-ltorch_python', '-ltorch_cuda'],
)

nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: nixl_inc_dirs)
nixl_lib_dir = join_paths(meson.project_build_root(), 'src', 'core')

v2_inc_dirs = [
    nixl_inc_dirs,
    nixl_device_api_inc_dirs,
    include_directories('csrc'),
    include_directories('nixl_ep_v2/include'),
    include_directories(v2_cccl_include_root),
    torch_inc_dirs,
]

torch_ver = run_command(
    py, '-c', 'import torch; print("".join(torch.__version__.split(".")[:2]))',
    check: true,
).stdout().strip()
v2_ext_name = 'nixl_ep_v2_cpp_torch' + torch_ver

v2_cpp_args = [
    '-DTORCH_EXTENSION_NAME=' + v2_ext_name,
    '-Wno-deprecated-declarations',
    '-Wno-unused-variable',
    '-Wno-sign-compare',
    '-Wno-reorder',
    '-Wno-attributes',
]
v2_cuda_args = [
    '-DTORCH_EXTENSION_NAME=' + v2_ext_name,
    '--extended-lambda',
    '--expt-relaxed-constexpr',
    '--diag-suppress=128,2417',
    '-Xcompiler', '-Wno-deprecated-declarations',
    '-Xcompiler', '-Wno-unused-variable',
    '-Xcompiler', '-Wno-sign-compare',
    '-Xcompiler', '-Wno-reorder',
    '-Xcompiler', '-Wno-attributes',
]

if nvcc.version().version_compare('>=12.9') and nvcc.version().version_compare('<12.10')
    v2_cpp_args += ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE']
    v2_cuda_args += ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE']
endif

v2_override_options = []
if get_option('buildtype') == 'debug'
    v2_override_options = ['buildtype=release', 'optimization=3', 'debug=false']
    v2_cpp_args += ['-g']
endif

v2_build_rpath = torch_lib_dir
v2_build_rpath += ':' + nixl_lib_dir
v2_build_rpath += ':' + join_paths(nixl_lib_dir, 'core')
v2_build_rpath += ':' + join_paths(nixl_lib_dir, 'plugins')
v2_install_rpath = join_paths(get_option('prefix'), get_option('libdir'))
v2_install_rpath += ':' + join_paths(get_option('prefix'), get_option('libdir'), 'plugins')
v2_install_rpath += ':' + torch_lib_dir
v2_install_dir = join_paths(cuda_wheel_dir, 'runtime', 'ep')

py.extension_module(
    v2_ext_name,
    [
        'csrc/python_api.cpp',
        'csrc/kernels/backend/nixl.cpp',
        'csrc/kernels/backend/nixl_device.cu',
    ],
    subdir: v2_install_dir,
    dependencies: [nixl_dep, pybind_dep, cuda_dep, torch_dep, ucx_dep],
    include_directories: v2_inc_dirs,
    cpp_args: v2_cpp_args,
    cuda_args: v2_cuda_args,
    link_with: [nixl_lib],
    build_rpath: v2_build_rpath,
    install_rpath: v2_install_rpath,
    override_options: v2_override_options,
    install: true,
)

# Compile every JIT kernel include against the same transport headers during
# the normal build, without installing or running the probe.
executable(
    'nixl_ep_v2_indexing_probe',
    'csrc/indexing/main.cu',
    dependencies: [cuda_dep, ucx_dep],
    include_directories: v2_inc_dirs,
    cuda_args: v2_cuda_args + ['-DEP_NIXL_DEVICE_API=1'],
    override_options: v2_override_options,
    build_by_default: true,
    install: false,
)

py.install_sources(
    'nixl_ep_v2/__init__.py',
    subdir: v2_install_dir,
    pure: false,
)
py.install_sources(
    'nixl_ep_v2/buffers/__init__.py',
    'nixl_ep_v2/buffers/elastic.py',
    subdir: join_paths(v2_install_dir, 'buffers'),
    pure: false,
)
py.install_sources(
    'nixl_ep_v2/utils/__init__.py',
    'nixl_ep_v2/utils/envs.py',
    'nixl_ep_v2/utils/event.py',
    'nixl_ep_v2/utils/find_pkgs.py',
    'nixl_ep_v2/utils/gate.py',
    'nixl_ep_v2/utils/math.py',
    'nixl_ep_v2/utils/nixl_bootstrap.py',
    'nixl_ep_v2/utils/nixl_fabric.py',
    'nixl_ep_v2/utils/refs.py',
    'nixl_ep_v2/utils/semantic.py',
    'nixl_ep_v2/utils/testing.py',
    subdir: join_paths(v2_install_dir, 'utils'),
    pure: false,
)

v2_include_install_dir = join_paths(py.get_install_dir(), v2_install_dir, 'include')
install_subdir(
    'nixl_ep_v2/include/nixl_ep_v2',
    install_dir: v2_include_install_dir,
)
install_subdir(
    join_paths(meson.project_source_root(), 'src', 'api', 'device', 'gpu'),
    install_dir: v2_include_install_dir,
    exclude_files: ['meson.build'],
)
install_data(
    join_paths(meson.project_source_root(), 'src', 'api', 'cpp', 'nixl_types.h'),
    install_dir: v2_include_install_dir,
)
install_data(
    join_paths(meson.project_source_root(), 'src', 'utils', 'device', 'proxy', 'proxy_protocol.h'),
    install_dir: join_paths(v2_include_install_dir, 'device', 'proxy'),
)
foreach ucx_header_dir : ['ucp', 'uct', 'ucs']
    install_subdir(
        join_paths(v2_ucx_include_root, ucx_header_dir),
        install_dir: v2_include_install_dir,
    )
endforeach
