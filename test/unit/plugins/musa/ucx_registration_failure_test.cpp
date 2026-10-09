// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "test_support.h"
#include "ucx/ucx_backend.h"

#include <array>
#include <new>
#include <string_view>

namespace {
enum class Failure { none, map, query, type, pack, pack_exception, unpack_exception };
Failure failure = Failure::none;
int maps = 0, queries = 0, packs = 0, unmaps = 0, unpacks = 0, destroyed_keys = 0;
int network_device_overrides = 0;

void
reset(Failure next) {
    failure = next;
    maps = queries = packs = unmaps = unpacks = destroyed_keys = 0;
}
} // namespace

// Linker wrapping keeps fault injection out of production and forwards successful calls to UCX.
extern "C" {
ucs_status_t
__real_ucp_mem_map(ucp_context_h, const ucp_mem_map_params_t *, ucp_mem_h *);
ucs_status_t
__real_ucp_mem_query(ucp_mem_h, ucp_mem_attr_t *);
ucs_status_t __real_ucp_mem_unmap(ucp_context_h, ucp_mem_h);
ucs_status_t
__real_ucp_rkey_pack(ucp_context_h, ucp_mem_h, void **, size_t *);
ucs_status_t
__real_ucp_ep_rkey_unpack(ucp_ep_h, const void *, ucp_rkey_h *);
void __real_ucp_rkey_destroy(ucp_rkey_h);
ucs_status_t
__real_ucp_config_modify(ucp_config_t *, const char *, const char *);

ucs_status_t
__wrap_ucp_config_modify(ucp_config_t *config, const char *name, const char *value) {
    if (std::string_view(name) == "NET_DEVICES") {
        ++network_device_overrides;
    }
    return __real_ucp_config_modify(config, name, value);
}

ucs_status_t
__wrap_ucp_mem_map(ucp_context_h ctx, const ucp_mem_map_params_t *params, ucp_mem_h *mem) {
    ++maps;
    return failure == Failure::map ? UCS_ERR_NO_MEMORY : __real_ucp_mem_map(ctx, params, mem);
}

ucs_status_t
__wrap_ucp_mem_query(ucp_mem_h mem, ucp_mem_attr_t *attr) {
    ++queries;
    const auto status = __real_ucp_mem_query(mem, attr);
    if (status == UCS_OK && failure == Failure::type) {
        attr->mem_type = UCS_MEMORY_TYPE_HOST;
    }
    return failure == Failure::query ? UCS_ERR_IO_ERROR : status;
}

ucs_status_t
__wrap_ucp_mem_unmap(ucp_context_h ctx, ucp_mem_h mem) {
    ++unmaps;
    return __real_ucp_mem_unmap(ctx, mem);
}

ucs_status_t
__wrap_ucp_rkey_pack(ucp_context_h ctx, ucp_mem_h mem, void **buffer, size_t *size) {
    ++packs;
    if (failure == Failure::pack_exception) {
        throw std::bad_alloc();
    }
    return failure == Failure::pack ? UCS_ERR_IO_ERROR :
                                      __real_ucp_rkey_pack(ctx, mem, buffer, size);
}

ucs_status_t
__wrap_ucp_ep_rkey_unpack(ucp_ep_h ep, const void *buffer, ucp_rkey_h *key) {
    ++unpacks;
    if (failure == Failure::unpack_exception && unpacks == 2) {
        throw std::length_error("injected unpack failure");
    }
    return __real_ucp_ep_rkey_unpack(ep, buffer, key);
}

void
__wrap_ucp_rkey_destroy(ucp_rkey_h key) {
    ++destroyed_keys;
    __real_ucp_rkey_destroy(key);
}
}

int
main() {
    nixlBackendInitParams init;
    nixl_b_params_t params{{"num_workers", "2"}, {"device_list", ""}};
    init.localAgent = "registration-failure-test";
    init.customParams = &params;
    init.type = "UCX";
    auto engine = nixlUcxEngine::create(init);
    NIXL_TEST_CHECK(!engine->getInitErr());
    NIXL_TEST_CHECK(network_device_overrides == 0);
    std::array<char, 4096> buffer{};
    nixlBlobDesc desc;
    desc.addr = reinterpret_cast<uintptr_t>(buffer.data());
    desc.len = buffer.size();
    desc.devId = 0;

    for (auto scenario :
         {Failure::map, Failure::query, Failure::type, Failure::pack, Failure::pack_exception}) {
        reset(scenario);
        auto *metadata = reinterpret_cast<nixlBackendMD *>(1);
        const auto type =
            scenario == Failure::query || scenario == Failure::type ? VRAM_SEG : DRAM_SEG;
        NIXL_TEST_CHECK(engine->registerMem(desc, type, metadata) == NIXL_ERR_BACKEND);
        NIXL_TEST_CHECK(metadata == nullptr);
        NIXL_TEST_CHECK(maps == 1);
        NIXL_TEST_CHECK(unmaps == (scenario == Failure::map ? 0 : 1));
        NIXL_TEST_CHECK(queries == (type == VRAM_SEG ? 1 : 0));
        NIXL_TEST_CHECK(packs ==
                        (scenario == Failure::pack || scenario == Failure::pack_exception ? 1 : 0));
    }

    reset(Failure::none);
    nixlBackendMD *metadata = nullptr;
    NIXL_TEST_CHECK(engine->registerMem(desc, DRAM_SEG, metadata) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(metadata && maps == 1 && packs == 1 && unmaps == 0);
    NIXL_TEST_CHECK(engine->getPublicData(metadata, desc.metaInfo) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(engine->connect(init.localAgent) == NIXL_SUCCESS);
    reset(Failure::unpack_exception);
    auto *remote = metadata;
    NIXL_TEST_CHECK(engine->loadRemoteMD(desc, DRAM_SEG, init.localAgent, remote) ==
                    NIXL_ERR_BACKEND);
    NIXL_TEST_CHECK(remote == nullptr && unpacks == 2 && destroyed_keys == 1);
    NIXL_TEST_CHECK(unmaps == 0);
    reset(Failure::none);
    NIXL_TEST_CHECK(engine->loadRemoteMD(desc, DRAM_SEG, init.localAgent, remote) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(remote && unpacks == 2);
    NIXL_TEST_CHECK(engine->unloadMD(remote) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(destroyed_keys == 2);
    NIXL_TEST_CHECK(engine->deregisterMem(metadata) == NIXL_SUCCESS);
    NIXL_TEST_CHECK(unmaps == 1);
    std::cout << "real UCX map/query/pack/unpack failure cleanup passed\n";
}
