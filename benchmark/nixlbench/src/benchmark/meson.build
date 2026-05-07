# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

benchmark_framework_sources = [
  'benchmark_executor.cpp',
  'benchmark_executor.h',
  'benchmark_run_components.h',
  'benchmark_runtime_sync.cpp',
  'benchmark_runtime_sync.h',
  'nixl_storage_allocator.cpp',
  'nixl_storage_allocator.h',
  'transfer_descriptor_strategy.cpp',
  'transfer_descriptor_strategy.h',
]

benchmark_framework_deps = [
  nixl_lib,
  tomlplusplus_dep,
]

if cuda_available
  benchmark_framework_deps += [cuda_dep]
endif

benchmark_framework_lib = static_library('benchmark_framework',
  benchmark_framework_sources,
  include_directories: inc_dir,
  dependencies: benchmark_framework_deps,
)

benchmark_framework_dep = declare_dependency(
  link_with: benchmark_framework_lib,
  dependencies: benchmark_framework_deps,
  include_directories: inc_dir,
)
