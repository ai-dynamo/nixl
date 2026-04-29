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

utils_sources = [
  'cli/benchmark_cli_builder.cpp',
  'cli/benchmark_cli_builder.h',
  'cli/benchmark_command.h',
  'cli/benchmark_requests.h',
  'cli/cli_option.cpp',
  'cli/cli_option.h',
  'cli/plugin_registry.cpp',
  'cli/plugin_registry.h',
  'cli/g3_scenario_command.cpp',
  'cli/g3_scenario_command.h',
  'cli/g4_scenario_command.cpp',
  'cli/g4_scenario_command.h',
  'cli/posix_plugin_command.cpp',
  'cli/posix_plugin_command.h',
  'cli/obj_plugin_command.cpp',
  'cli/obj_plugin_command.h',
  'cli/raw_command.cpp',
  'cli/raw_command.h',
  'cli/raw_execution.cpp',
  'cli/raw_execution.h',
  'neuron.cpp',
  'neuron.h',
  'utils.cpp',
  'utils.h',
  'scope_guard.h'
]

if get_option('with_gds_plugin')
  utils_sources += ['cli/gds_plugin_command.cpp', 'cli/gds_plugin_command.h']
  add_project_arguments('-DWITH_GDS_PLUGIN', language: 'cpp')
endif

if get_option('with_gds_mt_plugin')
  utils_sources += ['cli/gds_mt_plugin_command.cpp', 'cli/gds_mt_plugin_command.h']
  add_project_arguments('-DWITH_GDS_MT_PLUGIN', language: 'cpp')
endif

if get_option('with_gpunetio_plugin')
  utils_sources += ['cli/gpunetio_plugin_command.cpp', 'cli/gpunetio_plugin_command.h']
  add_project_arguments('-DWITH_GPUNETIO_PLUGIN', language: 'cpp')
endif

if get_option('with_azure_blob_plugin')
  utils_sources += ['cli/azure_blob_plugin_command.cpp', 'cli/azure_blob_plugin_command.h']
  add_project_arguments('-DWITH_AZURE_BLOB_PLUGIN', language: 'cpp')
endif

if get_option('with_hf3fs_plugin')
  utils_sources += ['cli/hf3fs_plugin_command.cpp', 'cli/hf3fs_plugin_command.h']
  add_project_arguments('-DWITH_HF3FS_PLUGIN', language: 'cpp')
endif

if get_option('with_gusli_plugin')
  utils_sources += ['cli/gusli_plugin_command.cpp', 'cli/gusli_plugin_command.h']
  add_project_arguments('-DWITH_GUSLI_PLUGIN', language: 'cpp')
endif

utils_deps = [
  cuda_dep,
  openmp_dep,
  gflags_dep,
  cli11_dep,
  tomlplusplus_dep,
  dependency('dl', required: true)
]

utils_lib = static_library('utils',
  utils_sources,
  dependencies: utils_deps,
  include_directories: inc_dir
)
utils_dep = declare_dependency(
  link_with: utils_lib,
  dependencies: utils_deps,
  include_directories: inc_dir
)
