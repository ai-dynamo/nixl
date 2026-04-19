# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

nixlbench_kernels_dep = []
nixlbench_kernels_lib = []

if build_nixlbench_kernels
  nixlbench_kernels_lib = static_library(
    'nixlbench_kernels',
    'nixlbench_device_launch.cu',
    include_directories: [inc_dir, include_directories('.')],
    dependencies: [cuda_dep],
    cuda_args: ucx_device_cuda_args,
    override_options: ucx_device_override_options,
  )
  nixlbench_kernels_dep = declare_dependency(
    link_with: nixlbench_kernels_lib,
    include_directories: include_directories('.'),
  )
endif
