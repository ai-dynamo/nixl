/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal single-process GPU-initiated RDMA example.
 * Both initiator and target run as threads with in-memory metadata exchange.
 */

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include "nixl.h"
#include "nixl_device.cuh"
#include "serdes/serdes.h"

constexpr uint8_t FILL_VALUE = 42;
constexpr int TARGET_GPU = 0;
constexpr int INITIATOR_GPU = 1;
constexpr size_t DATA_SIZE = 1024 * 1024;

#define CHECK(cmd, msg)                    \
    do {                                   \
        if ((cmd) != NIXL_SUCCESS) {       \
            std::cerr << msg << std::endl; \
            exit(1);                       \
        }                                  \
    } while (0)
#define CUDA_CHECK(cmd)                                      \
    do {                                                     \
        cudaError_t e = cmd;                                 \
        if (e != cudaSuccess) {                              \
            std::cerr << cudaGetErrorString(e) << std::endl; \
            exit(1);                                         \
        }                                                    \
    } while (0)

struct MetadataExchange {
    std::mutex mutex;
    std::condition_variable cv;
    bool initiator_ready = false, target_ready = false;
    std::string initiator_md, target_md, target_data_desc, target_signal_desc;
};

__global__ void
write_and_signal_kernel(nixlMemDesc src_desc,
                        nixlMemDesc dst_desc,
                        nixlMemDesc signal_desc,
                        size_t size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nixlPut<nixl_gpu_level_t::THREAD>(src_desc,
                                          dst_desc,
                                          size,
                                          0,
                                          static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY),
                                          nullptr);
        nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
            1, signal_desc, 0, static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr);
    }
}

__global__ void
wait_for_signal_kernel(const void *signal_ptr, uint64_t expected) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (nixlGpuReadSignal<nixl_gpu_level_t::THREAD>(signal_ptr) < expected) {}
    }
}

void
setup_agent(int gpu,
            const char *name,
            std::unique_ptr<nixlAgent> &agent,
            nixlBackendH *&backend,
            nixl_opt_args_t &extra_params,
            void *&data_ptr,
            void *&signal_ptr,
            size_t &signal_size) {
    CUDA_CHECK(cudaSetDevice(gpu));
    nixlAgentConfig cfg(true);
    agent = std::make_unique<nixlAgent>(name, cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t init_params;
    CHECK(agent->getPluginParams("UCX", mems, init_params), "getPluginParams");
    init_params["num_workers"] = "1";
    init_params["ucx_error_handling_mode"] = "none";
    CHECK(agent->createBackend("UCX", init_params, backend), "createBackend");
    extra_params.backends.push_back(backend);

    CUDA_CHECK(cudaMalloc(&data_ptr, DATA_SIZE));
    CUDA_CHECK(cudaMemset(data_ptr, gpu == TARGET_GPU ? 0 : FILL_VALUE, DATA_SIZE));
    CHECK(agent->getGpuSignalSize(signal_size, &extra_params), "getGpuSignalSize");
    CUDA_CHECK(cudaMalloc(&signal_ptr, signal_size));
    CUDA_CHECK(cudaMemset(signal_ptr, 0, signal_size));
}

void
run_target(MetadataExchange &ex) {
    std::cout << "[target] Starting on GPU " << TARGET_GPU << std::endl;

    std::unique_ptr<nixlAgent> agent;
    nixlBackendH *backend = nullptr;
    nixl_opt_args_t extra_params;
    void *data_ptr = nullptr, *signal_ptr = nullptr;
    size_t signal_size = 0;

    setup_agent(
        TARGET_GPU, "target", agent, backend, extra_params, data_ptr, signal_ptr, signal_size);

    nixl_reg_dlist_t data_reg(VRAM_SEG), signal_reg(VRAM_SEG);
    data_reg.addDesc(nixlBlobDesc((uintptr_t)data_ptr, DATA_SIZE, TARGET_GPU, "data"));
    signal_reg.addDesc(nixlBlobDesc((uintptr_t)signal_ptr, signal_size, TARGET_GPU, "signal"));
    CHECK(agent->registerMem(data_reg, &extra_params), "registerMem");
    CHECK(agent->registerMem(signal_reg, &extra_params), "registerMem");
    CHECK(agent->prepGpuSignal(signal_reg, &extra_params), "prepGpuSignal");

    std::string local_md;
    CHECK(agent->getLocalMD(local_md), "getLocalMD");
    {
        std::lock_guard<std::mutex> lock(ex.mutex);
        ex.target_md = local_md;
        ex.target_ready = true;
    }
    ex.cv.notify_all();

    std::string remote_md;
    {
        std::unique_lock<std::mutex> lock(ex.mutex);
        ex.cv.wait(lock, [&ex] { return ex.initiator_ready; });
        remote_md = ex.initiator_md;
    }

    std::string remote_name;
    CHECK(agent->loadRemoteMD(remote_md, remote_name), "loadRemoteMD");

    nixl_xfer_dlist_t empty(VRAM_SEG);
    while (agent->checkRemoteMD(remote_name, empty) != NIXL_SUCCESS)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::cout << "[target] Connected to initiator" << std::endl;

    nixl_xfer_dlist_t data_xfer(VRAM_SEG), signal_xfer(VRAM_SEG);
    data_xfer.addDesc(nixlBasicDesc((uintptr_t)data_ptr, DATA_SIZE, TARGET_GPU));
    signal_xfer.addDesc(nixlBasicDesc((uintptr_t)signal_ptr, signal_size, TARGET_GPU));

    nixlSerDes data_ser, signal_ser;
    CHECK(data_xfer.serialize(&data_ser), "serialize");
    CHECK(signal_xfer.serialize(&signal_ser), "serialize");

    {
        std::lock_guard<std::mutex> lock(ex.mutex);
        ex.target_data_desc = data_ser.exportStr();
        ex.target_signal_desc = signal_ser.exportStr();
    }
    ex.cv.notify_all();

    std::cout << "[target] Waiting for data..." << std::endl;
    wait_for_signal_kernel<<<1, 1>>>(signal_ptr, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify data transfer (check first and last byte)
    uint8_t check[2];
    CUDA_CHECK(cudaMemcpy(&check[0], data_ptr, 1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(&check[1], (uint8_t *)data_ptr + DATA_SIZE - 1, 1, cudaMemcpyDeviceToHost));
    std::cout << "[target] " << (check[0] == FILL_VALUE && check[1] == FILL_VALUE ? "✓" : "✗")
              << " Data verified (first=" << (int)check[0] << ", last=" << (int)check[1] << ")"
              << std::endl;

    CHECK(agent->deregisterMem(data_reg, &extra_params), "deregisterMem");
    CHECK(agent->deregisterMem(signal_reg, &extra_params), "deregisterMem");
    CUDA_CHECK(cudaFree(data_ptr));
    CUDA_CHECK(cudaFree(signal_ptr));
}

void
run_initiator(MetadataExchange &ex) {
    std::cout << "[initiator] Starting on GPU " << INITIATOR_GPU << std::endl;

    std::unique_ptr<nixlAgent> agent;
    nixlBackendH *backend = nullptr;
    nixl_opt_args_t extra_params;
    void *data_ptr = nullptr, *signal_ptr = nullptr;
    size_t signal_size = 0;

    setup_agent(INITIATOR_GPU,
                "initiator",
                agent,
                backend,
                extra_params,
                data_ptr,
                signal_ptr,
                signal_size);

    nixl_reg_dlist_t data_reg(VRAM_SEG), signal_reg(VRAM_SEG);
    data_reg.addDesc(nixlBlobDesc((uintptr_t)data_ptr, DATA_SIZE, INITIATOR_GPU, "data"));
    signal_reg.addDesc(nixlBlobDesc((uintptr_t)signal_ptr, signal_size, INITIATOR_GPU, "signal"));
    CHECK(agent->registerMem(data_reg, &extra_params), "registerMem");
    CHECK(agent->registerMem(signal_reg, &extra_params), "registerMem");

    std::string local_md;
    CHECK(agent->getLocalMD(local_md), "getLocalMD");
    {
        std::lock_guard<std::mutex> lock(ex.mutex);
        ex.initiator_md = local_md;
        ex.initiator_ready = true;
    }
    ex.cv.notify_all();

    std::string remote_md;
    {
        std::unique_lock<std::mutex> lock(ex.mutex);
        ex.cv.wait(lock, [&ex] { return ex.target_ready; });
        remote_md = ex.target_md;
    }

    std::string target_name;
    CHECK(agent->loadRemoteMD(remote_md, target_name), "loadRemoteMD");

    nixl_xfer_dlist_t empty(VRAM_SEG);
    while (agent->checkRemoteMD(target_name, empty) != NIXL_SUCCESS)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::string target_data_desc, target_signal_desc;
    {
        std::unique_lock<std::mutex> lock(ex.mutex);
        ex.cv.wait(
            lock, [&ex] { return !ex.target_data_desc.empty() && !ex.target_signal_desc.empty(); });
        target_data_desc = ex.target_data_desc;
        target_signal_desc = ex.target_signal_desc;
    }

    // Prepare local descriptors
    nixl_xfer_dlist_t local_data(VRAM_SEG);
    local_data.addDesc(nixlBasicDesc((uintptr_t)data_ptr, DATA_SIZE, INITIATOR_GPU));

    // Deserialize and convert remote descriptors to nixlRemoteDesc
    nixlSerDes remote_data_ser, remote_signal_ser;
    remote_data_ser.importStr(target_data_desc);
    remote_signal_ser.importStr(target_signal_desc);
    nixl_xfer_dlist_t remote_data_basic(&remote_data_ser), remote_signal_basic(&remote_signal_ser);

    nixl_remote_dlist_t remote_data(VRAM_SEG), remote_signal(VRAM_SEG);
    for (const auto &desc : remote_data_basic) {
        remote_data.addDesc(nixlRemoteDesc(desc, target_name));
    }
    for (const auto &desc : remote_signal_basic) {
        remote_signal.addDesc(nixlRemoteDesc(desc, target_name));
    }

    // Prepare memory views using Device API V2
    nixlMemoryViewH local_data_mvh = nullptr, remote_data_mvh = nullptr,
                    remote_signal_mvh = nullptr;
    CHECK(agent->prepMemoryView(local_data, local_data_mvh, &extra_params), "prepMemoryView local");
    CHECK(agent->prepMemoryView(remote_data, remote_data_mvh, &extra_params),
          "prepMemoryView remote data");
    CHECK(agent->prepMemoryView(remote_signal, remote_signal_mvh, &extra_params),
          "prepMemoryView remote signal");

    // Create memory descriptors for GPU
    nixlMemDesc src_desc{local_data_mvh, 0, 0};
    nixlMemDesc dst_desc{remote_data_mvh, 0, 0};
    nixlMemDesc signal_desc{remote_signal_mvh, 0, 0};

    std::cout << "[initiator] Transferring " << DATA_SIZE << " bytes..." << std::endl;
    write_and_signal_kernel<<<1, 1>>>(src_desc, dst_desc, signal_desc, DATA_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[initiator] Complete!" << std::endl;

    // Cleanup memory views
    agent->releaseMemoryView(local_data_mvh);
    agent->releaseMemoryView(remote_data_mvh);
    agent->releaseMemoryView(remote_signal_mvh);
    CHECK(agent->deregisterMem(data_reg, &extra_params), "deregisterMem");
    CHECK(agent->deregisterMem(signal_reg, &extra_params), "deregisterMem");
    CUDA_CHECK(cudaFree(data_ptr));
    CUDA_CHECK(cudaFree(signal_ptr));
}

int
main() {
    std::cout << "Starting unified example with " << DATA_SIZE << " bytes" << std::endl;

    MetadataExchange ex;
    std::thread target_thread([&ex]() { run_target(ex); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::thread initiator_thread([&ex]() { run_initiator(ex); });

    target_thread.join();
    initiator_thread.join();

    std::cout << "\n=== Example completed successfully ===" << std::endl;
    return 0;
}
