# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

nixl_cuda_vmm_args = ['-DHAVE_CUDA']
if cuda_fabric_available
    nixl_cuda_vmm_args += ['-DHAVE_CUDA_FABRIC']
endif

nixl_cuda_vmm_inc = [
    include_directories('.'),
    include_directories('..'),
]

nixl_cuda_vmm_lib = static_library(
    'nixl_cuda_vmm',
    'vmm.cpp',
    dependencies: [cuda_dep],
    include_directories: [nixl_cuda_vmm_inc],
    cpp_args: nixl_cuda_vmm_args,
    install: false,
)

nixl_cuda_vmm_dep = declare_dependency(
    link_with: nixl_cuda_vmm_lib,
    include_directories: [nixl_cuda_vmm_inc],
    dependencies: [cuda_dep],
    compile_args: nixl_cuda_vmm_args,
)
