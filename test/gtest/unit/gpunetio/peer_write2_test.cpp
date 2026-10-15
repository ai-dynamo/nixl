/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nixl.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void
checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void
check(nixl_status_t status) {
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("NIXL status " + std::to_string(status));
    }
}

struct Buffer {
    int owner;
    void *data = nullptr;

    explicit Buffer(int device) : owner(device) {
        checkCuda(cudaSetDevice(owner));
        checkCuda(cudaMalloc(&data, size));
    }

    ~Buffer() {
        cudaSetDevice(owner);
        cudaFree(data);
    }

    static constexpr size_t size = 4096 * 64 + 256;
};

void
runPeerTransfers(const char *nic) {
    std::array<Buffer, 2> buffers{Buffer(1), Buffer(0)};
    std::array<std::unique_ptr<nixlAgent>, 2> agents;
    std::array<nixl_reg_dlist_t, 2> registrations{nixl_reg_dlist_t(VRAM_SEG),
                                                  nixl_reg_dlist_t(VRAM_SEG)};
    std::array<nixl_blob_t, 2> metadata;
    std::array<std::string, 2> peer;
    const char *port = std::getenv("NIXL_TEST_OOB_PORT");
    if (port == nullptr) {
        throw std::runtime_error("Set NIXL_TEST_OOB_PORT to two free consecutive ports");
    }
    for (int engine = 0; engine < 2; ++engine) {
        checkCuda(cudaSetDevice(engine));
        nixlAgentConfig config;
        config.useProgThread = false;
        agents[engine] =
            std::make_unique<nixlAgent>("peer-write2-" + std::to_string(engine), config);
        nixl_mem_list_t types;
        nixl_b_params_t params;
        check(agents[engine]->getPluginParams("GPUNETIO", types, params));
        params["network_devices"] = nic;
        params["gpu_devices"] = std::to_string(engine);
        params["oob_port"] = std::to_string(std::stoul(port) + engine);
        if (const char *iface = std::getenv("NIXL_TEST_OOB_INTERFACE")) {
            params["oob_interface"] = iface;
        }
        if (const char *gid = std::getenv("NIXL_TEST_GID")) {
            params["gid_index"] = gid;
        }
        nixlBackendH *backend = nullptr;
        check(agents[engine]->createBackend("GPUNETIO", params, backend));
        const auto address = reinterpret_cast<uintptr_t>(buffers[engine].data);
        // Two source MRs exercise a pair whose local keys differ.
        const size_t first_size = engine == 0 ? Buffer::size / 2 - 64 : Buffer::size;
        registrations[engine].addDesc(nixlBlobDesc(address, first_size, buffers[engine].owner));
        if (first_size != Buffer::size) {
            registrations[engine].addDesc(nixlBlobDesc(
                address + first_size, Buffer::size - first_size, buffers[engine].owner));
        }
        check(agents[engine]->registerMem(registrations[engine]));
        int current;
        checkCuda(cudaGetDevice(&current));
        EXPECT_EQ(current, engine);
        check(agents[engine]->getLocalMD(metadata[engine]));
    }
    for (int engine = 0; engine < 2; ++engine) {
        checkCuda(cudaSetDevice(engine));
        check(agents[engine]->loadRemoteMD(metadata[1 - engine], peer[engine]));
    }
    // Cross ring wrap while alternating paired and fragmented target layouts.
    for (int layout = 0; layout < 2; ++layout) {
        const size_t remote_stride = layout == 0 ? 32 : 64;
        for (const uint32_t count : {2U, 513U, 1024U, 4096U}) {
            const size_t rows = count - 1;
            nixl_xfer_dlist_t local(VRAM_SEG), remote(VRAM_SEG);
            for (size_t row = 0; row < rows; ++row) {
                local.addDesc(
                    nixlBasicDesc(reinterpret_cast<uintptr_t>(buffers[0].data) + row * 64, 32, 1));
                remote.addDesc(nixlBasicDesc(
                    reinterpret_cast<uintptr_t>(buffers[1].data) + row * remote_stride, 32, 0));
            }
            local.addDesc(
                nixlBasicDesc(reinterpret_cast<uintptr_t>(buffers[0].data) + rows * 64, 8, 1));
            remote.addDesc(nixlBasicDesc(
                reinterpret_cast<uintptr_t>(buffers[1].data) + rows * remote_stride, 8, 0));
            nixl_opt_args_t attached;
            cudaStream_t stream = nullptr;
            attached.customParam.assign(reinterpret_cast<const char *>(&stream), sizeof(stream));
            nixlXferReqH *unsupported = nullptr;
            EXPECT_EQ(agents[0]->createXferReq(
                          NIXL_WRITE, local, remote, peer[0], unsupported, &attached),
                      NIXL_ERR_NOT_SUPPORTED);
            EXPECT_EQ(unsupported, nullptr);
            for (const auto op : {NIXL_WRITE, NIXL_READ}) {
                nixlXferReqH *request = nullptr;
                check(agents[0]->createXferReq(op, local, remote, peer[0], request));
                for (uint64_t epoch = 1; epoch <= 3; ++epoch) {
                    const int src = op == NIXL_WRITE ? 0 : 1;
                    const int dst = 1 - src;
                    const size_t src_stride = src == 0 ? 64 : remote_stride;
                    const size_t dst_stride = dst == 0 ? 64 : remote_stride;
                    std::vector<unsigned char> input(Buffer::size, 0xa5),
                        expected(Buffer::size, 0xa5);
                    for (size_t row = 0; row < rows; ++row) {
                        const uint64_t identity = (epoch << 32) | row;
                        for (size_t byte = 0; byte < 32; ++byte) {
                            const auto value =
                                reinterpret_cast<const unsigned char *>(&identity)[byte % 8];
                            input[row * src_stride + byte] = value;
                            expected[row * dst_stride + byte] = value;
                        }
                    }
                    std::memcpy(input.data() + rows * src_stride, &epoch, sizeof(epoch));
                    std::memcpy(expected.data() + rows * dst_stride, &epoch, sizeof(epoch));
                    checkCuda(cudaSetDevice(buffers[src].owner));
                    checkCuda(cudaMemcpy(
                        buffers[src].data, input.data(), Buffer::size, cudaMemcpyHostToDevice));
                    checkCuda(cudaStreamSynchronize(nullptr));
                    checkCuda(cudaSetDevice(buffers[dst].owner));
                    checkCuda(cudaMemset(buffers[dst].data, 0xa5, Buffer::size));
                    checkCuda(cudaStreamSynchronize(nullptr));
                    checkCuda(cudaSetDevice(1)); // Posting must select engine0 and restore caller1.
                    auto status = agents[0]->postXferReq(request);
                    const auto deadline =
                        std::chrono::steady_clock::now() + std::chrono::seconds(10);
                    while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
                        status = agents[0]->getXferStatus(request);
                        std::this_thread::yield();
                    }
                    check(status);
                    int current;
                    checkCuda(cudaGetDevice(&current));
                    EXPECT_EQ(current, 1);
                    std::vector<unsigned char> actual(Buffer::size);
                    checkCuda(cudaSetDevice(buffers[dst].owner));
                    checkCuda(cudaMemcpy(
                        actual.data(), buffers[dst].data, Buffer::size, cudaMemcpyDeviceToHost));
                    EXPECT_EQ(actual, expected)
                        << "op=" << op << " descriptors=" << count << " epoch=" << epoch;
                }
                check(agents[0]->releaseXferReq(request));
            }
        }
    }
    for (int engine = 0; engine < 2; ++engine) {
        check(agents[engine]->deregisterMem(registrations[engine]));
    }
}

TEST(GpunetioPeerWrite2, CrossOwnerReadWriteAndRepost) {
    const char *nic = std::getenv("NIXL_TEST_RDMA_DEVICE");
    if (nic == nullptr) {
        GTEST_SKIP() << "Set NIXL_TEST_RDMA_DEVICE and NIXL_TEST_OOB_PORT for the hardware test";
    }
    ASSERT_NO_THROW(runPeerTransfers(nic));
}
} // namespace
