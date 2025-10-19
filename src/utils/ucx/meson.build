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

ucx_utils_dep = [ ucx_dep, nixl_common_dep, serdes_interface ]

if cuda_dep.found()
    ucx_utils_dep += [ cuda_dep ]
endif

ucx_utils_inc_dirs = include_directories('.')

ucx_utils_sources = ['ucx_utils.cpp', 'ucx_utils.h', 'config.h', 'config.cpp', 'rkey.cpp', 'rkey.h', 'gpu_xfer_req_h.cpp', 'gpu_xfer_req_h.h']

ucx_utils_lib = library('ucx_utils',
           ucx_utils_sources,
           dependencies: ucx_utils_dep,
           include_directories: [ nixl_inc_dirs, utils_inc_dirs ],
           install: true)
