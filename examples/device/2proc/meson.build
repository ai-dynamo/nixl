# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NIXL Device API 2-Process Example (Pure C++)
# Demonstrates GPU-initiated RDMA transfers using device API

# Check dependencies
if not cuda_dep.found()
    warning('CUDA not found, skipping device 2proc example')
    subdir_done()
endif

py = import('python').find_installation('python3', pure: false)

# Check PyTorch
torch_check = run_command(py, '-c', 'import torch', check: false)
if torch_check.returncode() != 0
    warning('PyTorch not found, skipping device 2proc example')
    subdir_done()
endif

# Get PyTorch directories for C++ TCPStore API
torch_dir = run_command(py, '-c', 'import torch, os; print(os.path.dirname(torch.__file__))', check: true).stdout().strip()
torch_include_base = join_paths(torch_dir, 'include')
torch_lib_dir = join_paths(torch_dir, 'lib')

# PyTorch include directories
torch_inc_dirs = [
    include_directories(torch_include_base),
    include_directories(join_paths(torch_include_base, 'torch')),
    include_directories(join_paths(torch_include_base, 'torch', 'csrc', 'api', 'include')),
]

# Create PyTorch dependency (minimal - only for TCPStore, not tensor operations)
# Note: libtorch_cpu is needed because c10d::TCPStore is implemented there
torch_dep = declare_dependency(
    include_directories: torch_inc_dirs,
    link_args: ['-L' + torch_lib_dir, '-ltorch', '-ltorch_cpu', '-lc10']
)

# NIXL core dependency
nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: nixl_inc_dirs)

# Build CUDA kernels library
kernels_deps = [cuda_dep]
if ucx_dep.found()
  kernels_deps += [ucx_dep]
endif
doca_inc_dirs = []
if doca_gpunetio_dep.found()
  kernels_deps += [doca_gpunetio_dep]
  doca_inc = doca_gpunetio_dep.get_variable(pkgconfig: 'includedir', default_value: '')
  if doca_inc != ''
    doca_inc_dirs += [include_directories(doca_inc)]
  endif
endif

kernels_lib = static_library(
  'device_2proc_kernels',
  'csrc/kernels.cu',
  dependencies: kernels_deps,
  include_directories: [nixl_inc_dirs, nixl_gpu_inc_dirs] + doca_inc_dirs,
  cuda_args: [
    '-std=c++17',
    '-DHAVE_CUDA',
    '--expt-relaxed-constexpr',
    # Note: No -arch flag here - uses global gencode settings (sm_80, sm_90) for portability
    '--ptxas-options=--register-usage-level=10',
    '-Xcompiler', '-Wno-deprecated-declarations',
    '-Xcompiler', '-Wno-unused-variable',
    '-Xcompiler', '-Wno-sign-compare',
    '-Xcompiler', '-Wno-reorder',
    '-Xcompiler', '-Wno-attributes',
  ] + (meson.get_compiler('cuda').version().version_compare('>=12.9') and
       meson.get_compiler('cuda').version().version_compare('<12.10') ?
       ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE'] : [])
)

# Build C++ executable (pure C++, no Python, uses PyTorch TCPStore)
simple_write_exe = executable(
  'simple_write',
  ['csrc/simple_write.cpp', 'csrc/tcp_store.cpp'],
  dependencies: [nixl_dep, cuda_dep, torch_dep],
  include_directories: [nixl_inc_dirs, utils_inc_dirs, include_directories('csrc')],
  link_with: [nixl_lib, nixl_build_lib, serdes_lib, kernels_lib],
  cpp_args: ['-std=c++17'],
  build_rpath: torch_lib_dir,
  install_rpath: torch_lib_dir,
  install: false,
)

# TODO: Build unified single-file executable (no PyTorch dependency)
# Currently disabled - needs update to Device API V2
# simple_write_unified_exe = executable(
#   'simple_write_unified',
#   'simple_write_unified.cu',
#   dependencies: [nixl_dep, cuda_dep],
#   include_directories: [nixl_inc_dirs, nixl_gpu_inc_dirs, utils_inc_dirs] + doca_inc_dirs,
#   link_with: [nixl_lib, nixl_build_lib, serdes_lib],
#   cuda_args: [
#     '-std=c++17',
#     '-DHAVE_CUDA',
#     '--expt-relaxed-constexpr',
#     '--ptxas-options=--register-usage-level=10',
#     '-Xcompiler', '-Wno-deprecated-declarations',
#     '-Xcompiler', '-Wno-unused-variable',
#     '-Xcompiler', '-Wno-sign-compare',
#     '-Xcompiler', '-Wno-reorder',
#     '-Xcompiler', '-Wno-attributes',
#   ] + (meson.get_compiler('cuda').version().version_compare('>=12.9') and
#        meson.get_compiler('cuda').version().version_compare('<12.10') ?
#        ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE'] : []),
#   install: false,
# )

# Summary
message('NIXL Device 2proc Example (Pure C++)')
message('====================================')
message('simple_write:         ' + simple_write_exe.full_path())
message('')
message('Run: ./simple_write --mode initiator (terminal 1)')
message('     ./simple_write --mode target (terminal 2)')
message('')
message('See CPP_EXAMPLE.md for detailed usage and UCX configuration')
