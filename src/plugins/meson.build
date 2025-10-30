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

if ucx_dep.found()
    subdir('ucx')
endif

subdir('posix')  # Always try to build POSIX backend, it will handle its own dependencies
subdir('obj')  # Always try to build Obj backend, it will handle its own dependencies

if libfabric_dep.found()
    subdir('libfabric')
endif

disable_gds_backend = get_option('disable_gds_backend')
if not disable_gds_backend and cuda_dep.found()
      subdir('cuda_gds')
endif

if taskflow_proj.found() and cuda_dep.found() and not disable_gds_backend
      subdir('gds_mt')
endif

cc = meson.get_compiler('cpp')
libtransfer_engine = cc.find_library('transfer_engine', required: false)
if libtransfer_engine.found() and not get_option('disable_mooncake_backend')
    subdir('mooncake')
endif

hf3fs_lib_path = '/usr/lib/'
hf3fs_lib_file = 'hf3fs_api_shared'
hf3fs_lib_found = cc.find_library(hf3fs_lib_file, dirs: [hf3fs_lib_path], required: false)
if hf3fs_lib_found.found()
    subdir('hf3fs')
endif

gusli_lib_file = 'gusli_clnt'
gusli_lib_found = cc.find_library(gusli_lib_file, required: false)
if gusli_lib_found.found()
    subdir('gusli')
endif

if cuda_dep.found() and doca_gpunetio_dep.found()
    subdir('gpunetio')
endif
