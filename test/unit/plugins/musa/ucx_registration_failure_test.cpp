// SPDX-FileCopyrightText: Copyright (c) 2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "test_support.h"
#include "ucx/ucx_backend.h"

#include <array>
#include <new>

namespace {
enum class Fault { none, map, query, pack, pack_exception };
Fault fault = Fault::none;
int unmaps = 0;
} // namespace

extern "C" {
ucs_status_t
__real_ucp_mem_map(ucp_context_h, const ucp_mem_map_params_t *, ucp_mem_h *);
ucs_status_t
__real_ucp_mem_query(ucp_mem_h, ucp_mem_attr_t *);
ucs_status_t
__real_ucp_rkey_pack(ucp_context_h, ucp_mem_h, void **, size_t *);
ucs_status_t __real_ucp_mem_unmap(ucp_context_h, ucp_mem_h);

ucs_status_t
__wrap_ucp_mem_map(ucp_context_h context, const ucp_mem_map_params_t *params, ucp_mem_h *mem) {
    return fault == Fault::map ? UCS_ERR_NO_MEMORY : __real_ucp_mem_map(context, params, mem);
}

ucs_status_t
__wrap_ucp_mem_query(ucp_mem_h mem, ucp_mem_attr_t *attr) {
    return fault == Fault::query ? UCS_ERR_IO_ERROR : __real_ucp_mem_query(mem, attr);
}

ucs_status_t
__wrap_ucp_rkey_pack(ucp_context_h context, ucp_mem_h mem, void **buffer, size_t *size) {
    if (fault == Fault::pack_exception) {
        throw std::bad_alloc();
    }
    return fault == Fault::pack ? UCS_ERR_NO_MEMORY :
                                  __real_ucp_rkey_pack(context, mem, buffer, size);
}

ucs_status_t
__wrap_ucp_mem_unmap(ucp_context_h context, ucp_mem_h mem) {
    ++unmaps;
    return __real_ucp_mem_unmap(context, mem);
}
}

int
main() {
    nixlBackendInitParams init{};
    init.localAgent = "registration-failure-test";
    init.type = "UCX";
    auto engine = nixlUcxEngine::create(init);
    std::array<char, 4096> buffer{};
    nixlBlobDesc desc;
    desc.addr = reinterpret_cast<uintptr_t>(buffer.data());
    desc.len = buffer.size();
    desc.devId = 0;

    for (const auto injected :
         {Fault::map, Fault::query, Fault::none, Fault::pack, Fault::pack_exception}) {
        fault = injected;
        unmaps = 0;
        nixlBackendMD *metadata = reinterpret_cast<nixlBackendMD *>(1);
        const auto type = injected == Fault::query || injected == Fault::none ? VRAM_SEG : DRAM_SEG;
        NIXL_TEST_CHECK(engine->registerMem(desc, type, metadata) == NIXL_ERR_BACKEND);
        NIXL_TEST_CHECK(metadata == nullptr);
        NIXL_TEST_CHECK(unmaps == (injected == Fault::map ? 0 : 1));
    }

    fault = Fault::none;
    unmaps = 0;
    nixlBackendMD *metadata = nullptr;
    NIXL_TEST_CHECK(engine->registerMem(desc, DRAM_SEG, metadata) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(metadata != nullptr && unmaps == 0);
    NIXL_TEST_CHECK(engine->deregisterMem(metadata) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(unmaps == 1);
    std::cout << "production UCX map/query/pack failure cleanup passed\n";
}
