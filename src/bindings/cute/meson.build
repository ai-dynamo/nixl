# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if not ucx_gpu_device_api_available
  error('CuTe bindings require UCX GPU device API support')
endif
if ucx_path == '' or cuda_inc_path == ''
  error('CuTe bindings require explicit ucx_path and cudapath_inc')
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
cute_bitcodes = {}
foreach arch : selected_cuda_archs
  if arch in cute_bitcodes
    continue
  endif
  bitcode = custom_target('cute-bitcode-' + arch,
    input: 'device.cu',
    output: 'libnixl_device_' + arch + '.bc',
    depfile: 'device_' + arch + '.d',
    command: [py, files('build.py'),
              '--ucx', ucx_path, '--cuda', cute_cuda_path,
              '--arch', arch, '--llvm', cute_llvm_path,
              '--output', '@OUTPUT@', '--depfile', '@DEPFILE@'],
    depend_files: files('build.py'),
    install: true,
    install_dir: get_option('libdir'),
  )
  cute_bitcodes += {arch: bitcode}
endforeach
