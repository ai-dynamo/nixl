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

posix_sources = files(
  'posix_backend.cpp',
  'posix_plugin.cpp',
  'posix_utils.cpp'
)

posix_headers = files(
  'posix_backend.h',
  'posix_utils.h'
)

posix_lib = library('posix',
  posix_sources,
  posix_headers,
  include_directories: inc,
  dependencies: [core_dep],
  install: true
)

posix_dep = declare_dependency(
  link_with: posix_lib,
  include_directories: inc
) 