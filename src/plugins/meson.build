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

ucx_backend_inc_dirs = include_directories('./ucx')

subdir('ucx')
subdir('ucx_mo')

if not get_option('disable_mooncake_backend')
    subdir('mooncake')
endif

liburing_dep = dependency('liburing', required: false)

if liburing_dep.found()
    subdir('posix')
else
    message('liburing dependency not found, will not build POSIX backend')
endif

disable_gds_backend = get_option('disable_gds_backend')
if not disable_gds_backend and cuda_dep.found()
    subdir('cuda_gds')
endif

