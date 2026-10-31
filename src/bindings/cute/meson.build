# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if not ucx_gpu_device_api_available
  error('CuTe bindings require UCX GPU device API support')
endif
if ucx_path == '' or cuda_inc_path == ''
  error('CuTe bindings require explicit ucx_path and cudapath_inc')
endif
if selected_cuda_archs.length() != 1
  error('CuTe bindings require a single nixl_cuda_arch_list target')
endif
cute_cuda_path = fs.parent(cuda_inc_path)
if not fs.is_dir(cute_cuda_path / 'nvvm/libdevice')
  error('CuTe bindings require cudapath_inc=<CUDA toolkit>/include')
endif
cute_llvm_path = get_option('cute_llvm_path')
foreach tool : ['clang++', 'opt', 'llvm-as']
  find_program(cute_llvm_path / tool)
endforeach

# Keep the LLVM/libNVVM normalization in Python; Meson owns outputs and dependencies.
custom_target('cute-device-bitcode',
  input: 'device.cu',
  output: 'device.bc',
  depfile: 'device.d',
  command: [py, files('build.py'),
            '--ucx', ucx_path, '--cuda', cute_cuda_path,
            '--arch', selected_cuda_archs[0], '--llvm', cute_llvm_path,
            '--output', '@OUTPUT@', '--depfile', '@DEPFILE@'],
  depend_files: files('build.py'),
  build_by_default: true,
)
configure_file(input: 'nixl_cute.py', output: 'nixl_cute.py', copy: true)
