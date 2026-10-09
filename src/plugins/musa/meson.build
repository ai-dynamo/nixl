# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if not cpp.links('''
    #include <ucp/api/ucp.h>
    #include <ucs/memory/memory_type.h>
    int main() {
        ucp_context_attr_t context{};
        context.field_mask = UCP_ATTR_FIELD_MEMORY_TYPES;
        ucp_mem_attr_t memory{};
        memory.field_mask = UCP_MEM_ATTR_FIELD_MEM_TYPE;
        void *packed = nullptr;
        size_t size = 0;
        (void)ucp_context_query(nullptr, &context);
        (void)ucp_mem_query(nullptr, &memory);
        (void)ucp_rkey_pack(nullptr, nullptr, &packed, &size);
        return ucs_memory_type_names[UCS_MEMORY_TYPE_HOST] == nullptr;
    }
    ''', dependencies: ucx_dep, name: 'MUSA_UCX provider API contract')
    error('MUSA_UCX requires UCX memory query, rkey and memory-type name APIs')
endif

musa_runtime_rpath = musa_libdir
if get_option('ucx_path') != ''
    musa_runtime_rpath += ':' + (get_option('ucx_path') / 'lib')
endif
musa_backend_lib = shared_library('MUSA_UCX',
    ['musa_plugin.cpp', 'musa_policy.cpp', 'musa_runtime.cpp'],
    dependencies: [ucx_impl_interface, musa_dep],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    name_prefix: 'libplugin_',
    install: true,
    install_dir: plugin_install_dir,
    build_rpath: musa_runtime_rpath,
    install_rpath: '$ORIGIN/..:' + musa_runtime_rpath)

if get_option('buildtype') == 'debug'
    run_command('sh', '-c',
        'echo "MUSA_UCX=' + musa_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
        check: true)
endif
