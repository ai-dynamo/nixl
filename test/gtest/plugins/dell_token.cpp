/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#endif
#include "s3_accel/dell/client.h"
#include "s3_accel/dell/engine_impl.h"
#include "object/rdma/cuobj_client.h"
#include "obj_engine_registry.h"

namespace {
class fakeDellCuObjApi : public nixl_obj_rdma::cuObjApi {
public:
    int minted = 0;
    int released = 0;
    int unpinned = 0;
    size_t offset = 0;
    cuObjOpType_t operation = CUOBJ_GET;
    bool failMint = false;
    bool failRegister = false;
    std::function<void()> onRegister;

    cuObjErr_t
    cuMemObjGetDescriptor(void *, size_t) override {
        if (onRegister) {
            onRegister();
        }
        return failRegister ? CU_OBJ_FAIL : CU_OBJ_SUCCESS;
    }

    cuObjErr_t
    cuMemObjPutDescriptor(void *) override {
        ++unpinned;
        return CU_OBJ_SUCCESS;
    }

    cuObjErr_t
    cuMemObjGetRDMAToken(void *, size_t, size_t buffer_offset, cuObjOpType_t op, char **out)
        override {
        if (failMint) {
            return CU_OBJ_FAIL;
        }
        offset = buffer_offset;
        operation = op;
        *out = strdup("dell-token");
        ++minted;
        return CU_OBJ_SUCCESS;
    }

    cuObjErr_t
    cuMemObjPutRDMAToken(char *token) override {
        ++released;
        free(token);
        return CU_OBJ_SUCCESS;
    }
};

class recordingDellClient : public awsS3DellObsClient {
public:
    using awsS3DellObsClient::awsS3DellObsClient;
    std::function<void(bool)> pending;
    Aws::S3::Model::PutObjectRequest put;
    Aws::S3::Model::GetObjectRequest get;
    int submitted = 0;
    bool throwSubmit = false;

    void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor>) override {}

    void
    finish(bool success) {
        auto callback = std::move(pending);
        callback(success);
    }

protected:
    void
    submitPut(const Aws::S3::Model::PutObjectRequest &request,
              put_object_callback_t callback) override {
        if (throwSubmit) {
            throw std::runtime_error("submit failed");
        }
        put = request;
        ++submitted;
        pending = std::move(callback);
    }

    void
    submitGet(const Aws::S3::Model::GetObjectRequest &request,
              get_object_callback_t callback) override {
        if (throwSubmit) {
            throw std::runtime_error("submit failed");
        }
        get = request;
        ++submitted;
        pending = std::move(callback);
    }
};

class dellTokenTest : public testing::TestWithParam<nixl_xfer_op_t> {
protected:
    fakeDellCuObjApi *api = new fakeDellCuObjApi;
    nixl_obj_rdma::SharedCuObjClient rdma{std::unique_ptr<nixl_obj_rdma::cuObjApi>(api)};
    nixl_b_params_t params{{"accelerated", "true"},
                           {"type", "dell"},
                           {"bucket", "bucket"},
                           {"access_key", "test"},
                           {"secret_key", "test"}};
    nixlBackendInitParams init;
    std::shared_ptr<recordingDellClient> client =
        std::make_shared<recordingDellClient>(&params, rdma);
    std::unique_ptr<S3DellObsObjEngineImpl> engine;
    std::array<char, 128> buffer{};
    nixlBackendMD *localMd = nullptr;
    nixlBackendMD *remoteMd = nullptr;
    nixlBackendReqH *handle = nullptr;
    nixl_meta_dlist_t local{DRAM_SEG}, remote{OBJ_SEG};

    void
    SetUp() override {
        init.customParams = &params;
        engine = std::make_unique<S3DellObsObjEngineImpl>(&init, client, &rdma);
        ASSERT_EQ(
            engine->registerMem(
                nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer.data()), buffer.size(), 1, ""),
                DRAM_SEG,
                localMd),
            NIXL_SUCCESS);
        ASSERT_EQ(engine->registerMem(nixlBlobDesc(0, 32, 2, "object"), OBJ_SEG, remoteMd),
                  NIXL_SUCCESS);
        local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(buffer.data()) + 16, 32, 1));
        remote.addDesc(nixlMetaDesc(GetParam() == NIXL_READ ? 7 : 0, 32, 2));
    }

    void
    TearDown() override {
        client->pending = {};
        if (handle) {
            engine->releaseReqH(handle);
        }
        engine->deregisterMem(localMd);
        engine->deregisterMem(remoteMd);
    }

    void
    prepare() {
        ASSERT_EQ(engine->prepXfer(GetParam(), local, remote, "agent", "agent", handle, nullptr),
                  NIXL_SUCCESS);
        EXPECT_EQ(api->minted, 0);
    }

    nixl_status_t
    post() {
        return engine->postXfer(GetParam(), local, remote, "agent", handle);
    }
};

TEST_P(dellTokenTest, RequestAndLifetimeThroughCompletion) {
    prepare();
    ASSERT_EQ(post(), NIXL_IN_PROG);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    EXPECT_EQ(api->offset, 16);
    EXPECT_EQ(api->operation, GetParam() == NIXL_WRITE ? CUOBJ_PUT : CUOBJ_GET);
    if (GetParam() == NIXL_WRITE) {
        EXPECT_EQ(client->put.GetAdditionalCustomHeaders().at("x-rdma-info"), "dell-token");
        EXPECT_EQ(client->put.GetContentLength(), 0);
        EXPECT_EQ(client->put.GetBody(), nullptr);
    } else {
        EXPECT_EQ(client->get.GetAdditionalCustomHeaders().at("x-rdma-info"), "dell-token");
        EXPECT_EQ(client->get.GetRange(), "bytes=7-38");
    }
    EXPECT_EQ(api->released, 0);
    engine->deregisterMem(localMd);
    localMd = nullptr;
    EXPECT_EQ(api->unpinned, 0);
    client->finish(true);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(api->released, 1);
    EXPECT_EQ(api->unpinned, 1);
}

TEST_P(dellTokenTest, RepostMintsFreshTokenAndPropagatesFailure) {
    prepare();
    ASSERT_EQ(post(), NIXL_IN_PROG);
    client->finish(true);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
    ASSERT_EQ(post(), NIXL_IN_PROG);
    EXPECT_EQ(api->minted, 2);
    EXPECT_EQ(api->released, 1);
    client->finish(false);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(api->released, 2);
}

TEST_P(dellTokenTest, MintAndSubmissionFailuresCompleteWithError) {
    prepare();
    api->failMint = true;
    ASSERT_EQ(post(), NIXL_IN_PROG);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(client->submitted, 0);
    api->failMint = false;
    client->throwSubmit = true;
    ASSERT_EQ(post(), NIXL_IN_PROG);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(api->minted, api->released);
}

TEST_P(dellTokenTest, UnregisteredAndOutOfBoundsRangesFailAtPost) {
    prepare();
    engine->deregisterMem(localMd);
    localMd = nullptr;
    ASSERT_EQ(post(), NIXL_IN_PROG);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    ASSERT_TRUE(rdma.registerBuffer(buffer.data(), 20));
    ASSERT_EQ(post(), NIXL_IN_PROG);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    rdma.deregisterBuffer(buffer.data());
    EXPECT_EQ(client->submitted, 0);
}

TEST_P(dellTokenTest, InvalidLengthsDoNotMintOrSend) {
    prepare();
    local[0].len = 0;
    ASSERT_EQ(post(), NIXL_IN_PROG);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    local[0].len = 32;
    remote[0].addr = std::numeric_limits<size_t>::max();
    EXPECT_EQ(post(), GetParam() == NIXL_WRITE ? NIXL_ERR_INVALID_PARAM : NIXL_IN_PROG);
    if (GetParam() == NIXL_READ) {
        EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    }
    EXPECT_EQ(api->minted, 0);
    EXPECT_EQ(client->submitted, 0);
}

TEST_P(dellTokenTest, BatchValidationBeforeAnySubmission) {
    prepare();
    remote.addDesc(nixlMetaDesc(0, 32, 2));
    EXPECT_EQ(post(), NIXL_ERR_INVALID_PARAM);
    nixlBackendReqH *other = nullptr;
    EXPECT_EQ(engine->prepXfer(GetParam(), local, remote, "agent", "agent", other, nullptr),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(other, nullptr);
    local.addDesc(local[0]);
    remote[1].devId = 99;
    EXPECT_EQ(post(), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->prepXfer(GetParam(), local, remote, "agent", "agent", other, nullptr),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(other, nullptr);
    EXPECT_EQ(client->submitted, 0);
    EXPECT_EQ(api->minted, 0);
}

TEST_P(dellTokenTest, RegistrationContractsAndVendorDiscovery) {
    EXPECT_TRUE(objAccelEngineRegistry::instance().has("dell"));
    const auto mems = engine->getSupportedMems();
    EXPECT_NE(std::find(mems.begin(), mems.end(), VRAM_SEG), mems.end());
    nixlBackendMD *other = nullptr;
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 0, 3, ""), DRAM_SEG, other), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 1, 3, ""), FILE_SEG, other),
              NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 32, 2, "duplicate"), OBJ_SEG, other),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(other, nullptr);
    prepare();
    ASSERT_EQ(post(), NIXL_IN_PROG);
    client->finish(true);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(ReadAndWrite, dellTokenTest, testing::Values(NIXL_READ, NIXL_WRITE));
#ifdef HAVE_CUDA
// Real CUDA allocation/context handling with an injected cuObject API. No RDMA
// fabric or ObjectScale endpoint is needed; this does not test GPU-direct I/O.
class dellGpuRegistrationTest : public testing::Test {
protected:
    fakeDellCuObjApi *api = new fakeDellCuObjApi;
    nixl_obj_rdma::SharedCuObjClient rdma{std::unique_ptr<nixl_obj_rdma::cuObjApi>(api)};
    nixl_b_params_t params{{"accelerated", "true"},
                           {"type", "dell"},
                           {"bucket", "bucket"},
                           {"access_key", "test"},
                           {"secret_key", "test"}};
    nixlBackendInitParams init;
    std::shared_ptr<recordingDellClient> client;
    std::unique_ptr<S3DellObsObjEngineImpl> engine;
    void *buffer = nullptr;
    static constexpr size_t size = 1 << 20;

    void
    SetUp() override {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0)) {
            GTEST_SKIP() << "CUDA device/driver unavailable";
        }
        ASSERT_EQ(status, cudaSuccess);
        ASSERT_GT(devices, 0);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&buffer, size), cudaSuccess);
        ASSERT_EQ(cudaMemset(buffer, 0x5a, size), cudaSuccess);
        init.customParams = &params;
        client = std::make_shared<recordingDellClient>(&params, rdma);
        engine = std::make_unique<S3DellObsObjEngineImpl>(&init, client, &rdma);
    }

    void
    TearDown() override {
        if (buffer) {
            EXPECT_EQ(cudaFree(buffer), cudaSuccess);
        }
    }

    void
    verifyBuffer() {
        std::vector<unsigned char> bytes(size);
        ASSERT_EQ(cudaMemcpy(bytes.data(), buffer, size, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_TRUE(
            std::all_of(bytes.begin(), bytes.end(), [](unsigned char b) { return b == 0x5a; }));
    }
};

TEST_F(dellGpuRegistrationTest, WorkerThreadRestoresAbsentContextOnSuccessAndFailure) {
    int registrationCalls = 0;
    api->onRegister = [&registrationCalls] {
        ++registrationCalls;
        CUdevice device = -1;
        EXPECT_EQ(cuCtxGetDevice(&device), CUDA_SUCCESS);
        EXPECT_EQ(device, 0);
    };
    std::thread([&] {
        CUcontext before = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&before), CUDA_SUCCESS);
        ASSERT_EQ(before, nullptr);
        for (bool fail : {false, true}) {
            api->failRegister = fail;
            nixlBackendMD *md = nullptr;
            EXPECT_EQ(
                engine->registerMem(
                    nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer), size, 0, ""), VRAM_SEG, md),
                fail ? NIXL_ERR_BACKEND : NIXL_SUCCESS);
            CUcontext after = nullptr;
            EXPECT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
            EXPECT_EQ(after, nullptr);
            if (md) {
                EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
            } else {
                EXPECT_TRUE(fail);
            }
        }
    }).join();
    EXPECT_EQ(registrationCalls, 2);
    EXPECT_EQ(api->unpinned, 1);
    verifyBuffer();
}

TEST_F(dellGpuRegistrationTest, WorkerThreadPreservesExplicitContextOnSuccessAndFailure) {
    int registrationCalls = 0;
    api->onRegister = [&registrationCalls] {
        ++registrationCalls;
        CUdevice device = -1;
        EXPECT_EQ(cuCtxGetDevice(&device), CUDA_SUCCESS);
        EXPECT_EQ(device, 0);
    };
    std::thread([&] {
        CUdevice device;
        ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
        CUcontext saved = nullptr;
#if CUDA_VERSION >= 13000
        ASSERT_EQ(cuCtxCreate(&saved, nullptr, 0, device), CUDA_SUCCESS);
#else
        ASSERT_EQ(cuCtxCreate(&saved, 0, device), CUDA_SUCCESS);
#endif
        for (bool fail : {false, true}) {
            api->failRegister = fail;
            nixlBackendMD *md = nullptr;
            EXPECT_EQ(
                engine->registerMem(
                    nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer), size, 0, ""), VRAM_SEG, md),
                fail ? NIXL_ERR_BACKEND : NIXL_SUCCESS);
            CUcontext after = nullptr;
            EXPECT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
            EXPECT_EQ(after, saved);
            if (md)
                EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
            else
                EXPECT_TRUE(fail);
        }
        EXPECT_EQ(cuCtxDestroy(saved), CUDA_SUCCESS);
    }).join();
    EXPECT_EQ(registrationCalls, 2);
    EXPECT_EQ(api->unpinned, 1);
    verifyBuffer();
}
#endif // HAVE_CUDA

} // namespace
