# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

benchmark_path_sources = [
  'scenario.cpp',
  'scenario.h',
  'allocate_once.cpp',
  'allocate_once.h',
  'allocate_once_worker.cpp',
  'allocate_once_worker.h',
]

benchmark_path_lib = static_library(
  'benchmark_paths',
  benchmark_path_sources,
  dependencies: utils_deps + worker_deps + nixl_worker_deps,
  include_directories: inc_dir,
  link_with: [utils_lib, worker_libs],
)
