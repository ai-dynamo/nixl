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

if enabled_plugins.get('UCX')
    if not ucx_dep.found() and is_explicit_enable
        error('UCX plugin requested but UCX dependency not found')
    elif ucx_dep.found()
        subdir('ucx')
    endif
endif

if enabled_plugins.get('POSIX')
    subdir('posix')
endif

if enabled_plugins.get('OBJ')
    subdir('obj')
endif

if enabled_plugins.get('AZURE_BLOB')
    subdir('azure_blob')
endif

if enabled_plugins.get('LIBFABRIC')
    if not libfabric_dep.found() and is_explicit_enable
        error('LIBFABRIC plugin requested but libfabric dependency not found')
    elif libfabric_dep.found()
        subdir('libfabric')
    endif
endif

disable_gds_backend = get_option('disable_gds_backend')
if enabled_plugins.get('GDS')
    if (disable_gds_backend or not cuda_dep.found()) and is_explicit_enable
        if disable_gds_backend
            error('GDS plugin requested but GDS backend is disabled')
        else
            error('GDS plugin requested but CUDA dependency not found')
        endif
    elif not disable_gds_backend and cuda_dep.found()
        subdir('cuda_gds')
    endif
endif

if enabled_plugins.get('GDS_MT')
    if (disable_gds_backend or not cuda_dep.found() or not taskflow_proj.found()) and is_explicit_enable
        if disable_gds_backend
            error('GDS_MT plugin requested but GDS backend is disabled')
        elif not cuda_dep.found()
            error('GDS_MT plugin requested but CUDA dependency not found')
        else
            error('GDS_MT plugin requested but Taskflow dependency not found')
        endif
    elif not disable_gds_backend and cuda_dep.found() and taskflow_proj.found()
        subdir('gds_mt')
    endif
endif

cc = meson.get_compiler('cpp')
libtransfer_engine = cc.find_library('transfer_engine', required: false)
if enabled_plugins.get('MOONCAKE')
    if (get_option('disable_mooncake_backend') or not libtransfer_engine.found()) and is_explicit_enable
        if get_option('disable_mooncake_backend')
            error('MOONCAKE plugin requested but Mooncake backend is disabled')
        else
            error('MOONCAKE plugin requested but transfer_engine library not found')
        endif
    elif not get_option('disable_mooncake_backend') and libtransfer_engine.found()
        subdir('mooncake')
    endif
endif

libuccl_p2p = cc.find_library('uccl_p2p', required: false)
if enabled_plugins.get('UCCL')
    if not libuccl_p2p.found() and is_explicit_enable
        error('UCCL plugin requested but uccl_p2p library not found')
    elif libuccl_p2p.found()
        subdir('uccl')
    endif
endif

hf3fs_lib_path = '/usr/lib/'
hf3fs_lib_file = 'hf3fs_api_shared'
hf3fs_lib_found = cc.find_library(hf3fs_lib_file, dirs: [hf3fs_lib_path], required: false)
if enabled_plugins.get('HF3FS')
    if not hf3fs_lib_found.found() and is_explicit_enable
        error('HF3FS plugin requested but hf3fs_api_shared library not found')
    elif hf3fs_lib_found.found()
        subdir('hf3fs')
    endif
endif

infinia_path = get_option('infinia_path')
infinia_inc_path = infinia_path + '/include'
infinia_lib_path = infinia_path + '/lib'
infinia_lib_file = 'red_async'
infinia_lib_found = cc.find_library(infinia_lib_file, dirs: [infinia_lib_path], required: false)

# Infinia SDK 2.5+ symbols required by this plugin.
# has_header_symbol cannot probe C++ scoped names (preprocessor rejects '::'),
# so compile a snippet that references each symbol.
infinia_symbol_found = cpp.compiles('''
    #include <red/red_async.hpp>
    int main() {
        (void)red_async::RED_ASYNC_DEFAULT_MAX_RETRIES;
        (void)red_async::RED_ASYNC_DEFAULT_BATCH_SIZE;
        (void)red_async::RED_ASYNC_OP_CACHE_HEAD;
        (void)red_async::RED_ASYNC_OP_CACHE_GET;
        (void)red_async::RED_ASYNC_OP_CACHE_PUT;
        return 0;
    }
    ''',
    args: ['-I' + infinia_inc_path],
    name: 'Infinia SDK 2.5 symbols')

disable_infinia_backend = get_option('disable_infinia_backend')
infinia_sdk_ok = infinia_symbol_found and infinia_lib_found.found()
if enabled_plugins.get('INFINIA')
    if (disable_infinia_backend or not infinia_sdk_ok) and is_explicit_enable
        if disable_infinia_backend
            error('INFINIA plugin requested but Infinia backend is disabled')
        elif not infinia_lib_found.found()
            error('INFINIA plugin requested but red_async library not found')
        else
            error('INFINIA plugin requested but Infinia SDK is too old: needs release 2.5.0 or above')
        endif
    elif not disable_infinia_backend and infinia_sdk_ok
        subdir('infinia')
    elif not infinia_symbol_found
        message('INFINIA SDK too old: needs release 2.5.0 or above; disabling INFINIA plugin')
    endif
endif

gusli_lib_file = 'gusli_clnt'
gusli_lib_found = cc.find_library(gusli_lib_file, required: false)
if enabled_plugins.get('GUSLI')
    if not gusli_lib_found.found() and is_explicit_enable
        error('GUSLI plugin requested but gusli_clnt library not found')
    elif gusli_lib_found.found()
        subdir('gusli')
    endif
endif

if enabled_plugins.get('GPUNETIO')
    if (not cuda_dep.found() or not doca_gpunetio_dep.found()) and is_explicit_enable
        if not cuda_dep.found()
            error('GPUNETIO plugin requested but CUDA dependency not found')
        else
            error('GPUNETIO plugin requested but doca-gpunetio dependency not found')
        endif
    elif cuda_dep.found() and doca_gpunetio_dep.found()
        subdir('gpunetio')
    endif
endif

subdir('telemetry')

if nixl_trace_enabled
    subdir('tracing')
endif
