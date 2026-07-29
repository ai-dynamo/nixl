/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <dlfcn.h>

#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "consumer_module.h"
#include "device_proxy/backend_adapter.h"
#include "device_proxy/proxy_runtime.h"

#ifndef NIXL_TEST_LATE_DSO_PATH
#error "NIXL_TEST_LATE_DSO_PATH must point at the deliberately late-loaded consumer"
#endif

namespace {

class StubBackend : public nixlDeviceProxyBackendAdapter {
public:
    nixl_status_t
    init(uint32_t, uint32_t, uint32_t) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    submit(const nixlBackendProxySubmission &, nixlBackendProxyRequest &request) override {
        request = {};
        return NIXL_SUCCESS;
    }

    nixl_status_t
    checkCompletion(const nixlBackendProxyRequest &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    progress() override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    shutdown() override {
        return NIXL_SUCCESS;
    }
};

class DummyBackendMD : public nixlBackendMD {
public:
    DummyBackendMD() : nixlBackendMD(false) {}
};

struct ProxyHandles {
    nixlMemViewH local = nullptr;
    nixlMemViewH remote = nullptr;
};

ProxyHandles
createHandles(nixlProxyRuntime &runtime, uintptr_t base_addr) {
    static DummyBackendMD local_md;
    static DummyBackendMD remote_md;

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(base_addr, 64, 0, &local_md));
    nixlPreparedProxyMemView local;
    EXPECT_EQ(runtime.createLocal(local_dlist, local), NIXL_SUCCESS);

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer0");
    remote_desc.addr = base_addr + 0x1000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    nixlPreparedProxyMemView remote;
    EXPECT_EQ(runtime.createRemote(remote_dlist, remote), NIXL_SUCCESS);

    return {local.handle, remote.handle};
}

DeviceApiModuleResult
callModule(DeviceApiModuleCallFn call, const DeviceApiModuleRequest &request) {
    DeviceApiModuleResult result{};
    EXPECT_EQ(call(&request, &result), cudaSuccess);
    return result;
}

DeviceApiModuleResult
describe(DeviceApiModuleCallFn call, nixlMemViewH handle) {
    DeviceApiModuleRequest request{};
    request.action = DeviceApiModuleAction::DESCRIBE;
    request.src = handle;
    return callModule(call, request);
}

void
expectDescription(DeviceApiModuleCallFn call,
                  nixlMemViewH handle,
                  const void *expected_runtime_identity) {
    const auto result = describe(call, handle);
    EXPECT_EQ(result.status, NIXL_SUCCESS);
    EXPECT_EQ(result.version, NIXL_PROXY_MEM_LIST_VERSION_V1);
    EXPECT_EQ(result.length, 1u);
    EXPECT_EQ(result.runtime_identity, expected_runtime_identity);
}

DeviceApiTransferStatus
submit(DeviceApiModuleCallFn call, const ProxyHandles &handles) {
    DeviceApiModuleRequest request{};
    request.action = DeviceApiModuleAction::PUT;
    request.src = handles.local;
    request.dst = handles.remote;
    const auto result = callModule(call, request);
    EXPECT_EQ(result.status, NIXL_IN_PROG);
    return result.transfer_status;
}

void
expectCompletion(DeviceApiModuleCallFn call, const DeviceApiTransferStatus &transfer_status) {
    DeviceApiModuleRequest request{};
    request.action = DeviceApiModuleAction::POLL;
    request.transfer_status = transfer_status;

    constexpr auto timeout = std::chrono::seconds(2);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    nixl_status_t status = NIXL_IN_PROG;
    while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
        status = callModule(call, request).status;
        if (status == NIXL_IN_PROG) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    EXPECT_EQ(status, NIXL_SUCCESS);
}

class DeviceModuleIntegrationTest : public testing::Test {
protected:
    void
    SetUp() override {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA-capable GPU is available";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    static void
    initRuntime(nixlProxyRuntime &runtime) {
        ASSERT_EQ(runtime.init(std::make_unique<StubBackend>(),
                               /*peer_capacity=*/1,
                               /*channel_count=*/1,
                               /*worker_count=*/1),
                  NIXL_SUCCESS);
        ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    }
};

TEST_F(DeviceModuleIntegrationTest, PassesHandlesAndStatusAcrossPreloadedModules) {
    nixlProxyRuntime runtime;
    ASSERT_NO_FATAL_FAILURE(initRuntime(runtime));
    const auto handles = createHandles(runtime, 0x1000);

    const auto *runtime_identity =
        describe(nixlTestCallExecutableTuA, handles.local).runtime_identity;
    const std::array<DeviceApiModuleCallFn, 4> modules{
        nixlTestCallExecutableTuA,
        nixlTestCallExecutableTuB,
        nixlTestCallEarlyDsoA,
        nixlTestCallEarlyDsoB,
    };
    for (const auto module : modules) {
        expectDescription(module, handles.local, runtime_identity);
    }

    expectCompletion(nixlTestCallEarlyDsoB, submit(nixlTestCallExecutableTuA, handles));
    expectCompletion(nixlTestCallExecutableTuB, submit(nixlTestCallEarlyDsoA, handles));
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

TEST_F(DeviceModuleIntegrationTest, LateLoadedModuleSurvivesUnloadAndReload) {
    nixlProxyRuntime runtime;
    ASSERT_NO_FATAL_FAILURE(initRuntime(runtime));
    const auto handles = createHandles(runtime, 0x3000);
    const auto *runtime_identity =
        describe(nixlTestCallExecutableTuA, handles.local).runtime_identity;

    for (int load = 0; load < 2; ++load) {
        void *dso = dlopen(NIXL_TEST_LATE_DSO_PATH, RTLD_NOW | RTLD_LOCAL);
        ASSERT_NE(dso, nullptr) << dlerror();
        dlerror();
        const auto call =
            reinterpret_cast<DeviceApiModuleCallFn>(dlsym(dso, "nixlTestCallLateDso"));
        const char *symbol_error = dlerror();
        ASSERT_EQ(symbol_error, nullptr) << symbol_error;
        ASSERT_NE(call, nullptr);

        expectDescription(call, handles.remote, runtime_identity);
        expectCompletion(nixlTestCallExecutableTuA, submit(call, handles));
        ASSERT_EQ(dlclose(dso), 0);
    }

    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

TEST_F(DeviceModuleIntegrationTest, MultipleRuntimeContextsRemainHandleLocal) {
    nixlProxyRuntime first;
    nixlProxyRuntime second;
    ASSERT_NO_FATAL_FAILURE(initRuntime(first));
    ASSERT_NO_FATAL_FAILURE(initRuntime(second));
    const auto first_handles = createHandles(first, 0x5000);
    const auto second_handles = createHandles(second, 0x9000);

    const auto *first_identity =
        describe(nixlTestCallExecutableTuA, first_handles.local).runtime_identity;
    const auto *second_identity =
        describe(nixlTestCallExecutableTuB, second_handles.local).runtime_identity;
    ASSERT_NE(first_identity, second_identity);

    expectDescription(nixlTestCallEarlyDsoA, first_handles.local, first_identity);
    expectDescription(nixlTestCallEarlyDsoB, second_handles.remote, second_identity);
    expectCompletion(nixlTestCallExecutableTuB,
                     submit(nixlTestCallExecutableTuA, first_handles));
    expectCompletion(nixlTestCallEarlyDsoA, submit(nixlTestCallEarlyDsoB, second_handles));

    DeviceApiModuleRequest mixed{};
    mixed.action = DeviceApiModuleAction::PUT;
    mixed.src = first_handles.local;
    mixed.dst = second_handles.remote;
    EXPECT_EQ(callModule(nixlTestCallExecutableTuA, mixed).status, NIXL_ERR_INVALID_PARAM);

    ASSERT_EQ(first.shutdown(), NIXL_SUCCESS);
    ASSERT_EQ(second.shutdown(), NIXL_SUCCESS);
}

} // namespace
