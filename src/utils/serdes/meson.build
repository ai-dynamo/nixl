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

serdes_lib = library('serdes',
           'serdes.cpp', 'serdes.h',
           include_directories: inc_dir,
           install: true)

if get_option('buildtype') != 'release'

  serdes_test_bin = executable('serdes_test',
             'serdes_test.cpp',
             include_directories: inc_dir,
             link_with: serdes_lib,
             install: true)
endif

serdes_interface = declare_dependency(link_with: serdes_lib)
