# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

nixlbench_kernels_dep = []

if cuda_available
  nixlbench_kernels_lib = static_library(
    'nixlbench_kernels',
    'nixlbench_device_launch.cu',
    include_directories: [inc_dir, include_directories('.')],
    dependencies: [cuda_dep],
  )
  nixlbench_kernels_dep = declare_dependency(
    link_with: nixlbench_kernels_lib,
    include_directories: include_directories('.'),
  )
endif
