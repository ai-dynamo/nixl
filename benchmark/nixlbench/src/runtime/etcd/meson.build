# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

etcd_rt_sources = [
  'etcd_rt.cpp',
  'etcd_rt.h',
]

etcd_rt_lib = static_library('etcd_rt',
  etcd_rt_sources,
  include_directories: inc_dir,
  dependencies: [etcd_dep],
  install: true,
)

# Python bindings
python = import('python').find_installation('python3', required: false)
pybind11_dep = dependency('pybind11', required: false)

if python.found() and pybind11_dep.found()
  py_mod = python.extension_module('etcd_runtime',
    'python_bindings.cpp',
    include_directories: inc_dir,
    dependencies: [etcd_dep, pybind11_dep],
    link_with: etcd_rt_lib,
    install: true,
    subdir: 'nixlbench'
  )
endif
