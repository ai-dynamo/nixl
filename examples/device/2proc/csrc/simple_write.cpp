/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <getopt.h>

#include <cuda_runtime.h>

#include "nixl.h"
#include "serdes/serdes.h"
#include "tcp_store.h"
#include "kernels.h"

// Configuration constants
constexpr uint8_t INITIATOR_FILL_VALUE = 42;  // Initiator writes this pattern
constexpr int INITIATOR_DEVICE = 0;            // Initiator always uses GPU 0
constexpr int TARGET_DEVICE = 1;               // Target always uses GPU 1

#define CUDA_CHECK(cmd) do {                         \
    cudaError_t e = cmd;                             \
    if (e != cudaSuccess) {                          \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                          \
    }                                                \
} while(0)

#define NIXL_CHECK(cmd, msg) do {                    \
    if ((cmd) != NIXL_SUCCESS) {                     \
        std::cerr << "NIXL error: " << msg           \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                          \
    }                                                \
} while(0)

struct Config {
    std::string mode = "initiator";  // "initiator" or "target"
    size_t size = 1024 * 1024;       // 1 MB default
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <initiator|target>  Run as initiator or target (default: initiator)\n"
              << "  --size <bytes>             Buffer size in bytes (default: 1048576)\n"
              << "  --help                     Show this help\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;

    static struct option long_options[] = {
        {"mode", required_argument, 0, 'm'},
        {"size", required_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:s:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                cfg.mode = optarg;
                break;
            case 's':
                cfg.size = std::stoull(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }

    return cfg;
}

void run_target(const Config& cfg) {
    std::cout << "[target] Starting..." << std::endl;

    // Target always uses GPU 1 (for demo simplicity)
    CUDA_CHECK(cudaSetDevice(TARGET_DEVICE));

    // Create NIXL agent (pattern from nixl_example.cpp)
    auto agent = std::make_unique<nixlAgent>("target", nixlAgentConfig(true));

    // Get UCX backend parameters
    nixl_mem_list_t mems;
    nixl_b_params_t init_params;
    NIXL_CHECK(agent->getPluginParams("UCX", mems, init_params), "getPluginParams");

    // UCX configuration for device API (from EP framework):
    // - num_workers=1: Single worker sufficient for 2proc pattern
    // - ucx_error_handling_mode=none: Reduces overhead, optimal for device API
    init_params["num_workers"] = "1";
    init_params["ucx_error_handling_mode"] = "none";

    nixlBackendH* backend = nullptr;
    NIXL_CHECK(agent->createBackend("UCX", init_params, backend), "createBackend");

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend);

    // Allocate GPU memory (initialize to 0 - will receive INITIATOR_FILL_VALUE)
    void* data_ptr = nullptr;
    void* signal_ptr = nullptr;

    CUDA_CHECK(cudaMalloc(&data_ptr, cfg.size));
    CUDA_CHECK(cudaMemset(data_ptr, 0, cfg.size));

    // Get signal size - ensures both sides use same architecture/signal format
    size_t signal_size = 0;
    NIXL_CHECK(agent->getGpuSignalSize(signal_size, &extra_params), "getGpuSignalSize");

    CUDA_CHECK(cudaMalloc(&signal_ptr, signal_size));
    CUDA_CHECK(cudaMemset(signal_ptr, 0, signal_size));

    // Register memory with NIXL
    // nixlBlobDesc: For registration (includes name field for debugging)
    // nixlBasicDesc: For transfers (minimal - just ptr, size, dev_id)
    nixl_reg_dlist_t data_reg(VRAM_SEG);
    data_reg.addDesc(nixlBlobDesc((uintptr_t)data_ptr, cfg.size, TARGET_DEVICE, "target_data"));
    NIXL_CHECK(agent->registerMem(data_reg, &extra_params), "registerMem data");

    nixl_reg_dlist_t signal_reg(VRAM_SEG);
    signal_reg.addDesc(nixlBlobDesc((uintptr_t)signal_ptr, signal_size, TARGET_DEVICE, "target_signal"));
    NIXL_CHECK(agent->registerMem(signal_reg, &extra_params), "registerMem signal");
    NIXL_CHECK(agent->prepGpuSignal(signal_reg, &extra_params), "prepGpuSignal");

    // Create TCPStore (target is not master, waits for initiator's server)
    tcp_store::TCPStore store("127.0.0.1", 9998, false);

    // Get local metadata
    std::string local_meta;
    NIXL_CHECK(agent->getLocalMD(local_meta), "getLocalMD");

    // Publish metadata using TCPStore (following EP pattern)
    store.set("NIXL_2PROC/target_meta", local_meta);

    // Serialize and publish transfer descriptors
    nixl_xfer_dlist_t data_xfer(VRAM_SEG);
    data_xfer.addDesc(nixlBasicDesc((uintptr_t)data_ptr, cfg.size, TARGET_DEVICE));
    nixlSerDes data_serdes;
    NIXL_CHECK(data_xfer.serialize(&data_serdes), "serialize data descs");
    store.set("NIXL_2PROC/target_data_descs", data_serdes.exportStr());

    nixl_xfer_dlist_t signal_xfer(VRAM_SEG);
    signal_xfer.addDesc(nixlBasicDesc((uintptr_t)signal_ptr, signal_size, TARGET_DEVICE));
    nixlSerDes signal_serdes;
    NIXL_CHECK(signal_xfer.serialize(&signal_serdes), "serialize signal descs");
    store.set("NIXL_2PROC/target_signal_descs", signal_serdes.exportStr());

    std::cout << "[target] Published metadata and descriptors" << std::endl;

    // Wait for initiator metadata and establish connection
    if (!store.wait("NIXL_2PROC/initiator_meta", 30000)) {
        throw std::runtime_error("Timeout waiting for initiator metadata");
    }
    std::string remote_meta = store.get("NIXL_2PROC/initiator_meta");

    std::string remote_name;
    NIXL_CHECK(agent->loadRemoteMD(remote_meta, remote_name), "loadRemoteMD");

    // Wait for remote agent to be ready
    nixl_xfer_dlist_t empty_descs(VRAM_SEG);
    while (agent->checkRemoteMD(remote_name, empty_descs) != NIXL_SUCCESS) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "[target] Connected to initiator" << std::endl;

    // Wait for GPU signal from initiator
    std::cout << "[target] Waiting for data..." << std::endl;
    launch_wait_for_signal((uintptr_t)signal_ptr, 1, 1, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[target] Signal received!" << std::endl;

    // Verify data (should be INITIATOR_FILL_VALUE, not 0)
    std::vector<uint8_t> host_data(cfg.size);
    CUDA_CHECK(cudaMemcpy(host_data.data(), data_ptr, cfg.size, cudaMemcpyDeviceToHost));

    uint64_t expected_checksum = cfg.size * INITIATOR_FILL_VALUE;
    uint64_t actual_checksum = 0;
    for (uint8_t val : host_data) {
        actual_checksum += val;
    }

    if (actual_checksum == expected_checksum) {
        std::cout << "[target] ✓ Checksum OK: " << actual_checksum << std::endl;
    } else {
        std::cerr << "[target] ✗ Checksum MISMATCH: got=" << actual_checksum
                  << " expected=" << expected_checksum << std::endl;
    }

    // Cleanup
    NIXL_CHECK(agent->deregisterMem(data_reg, &extra_params), "deregisterMem data");
    NIXL_CHECK(agent->deregisterMem(signal_reg, &extra_params), "deregisterMem signal");
    CUDA_CHECK(cudaFree(data_ptr));
    CUDA_CHECK(cudaFree(signal_ptr));

    // Clean up TCPStore keys (following EP pattern)
    store.delete_key("NIXL_2PROC/target_meta");
    store.delete_key("NIXL_2PROC/target_data_descs");
    store.delete_key("NIXL_2PROC/target_signal_descs");

    std::cout << "[target] Complete!" << std::endl;
}

void run_initiator(const Config& cfg) {
    std::cout << "[initiator] Starting..." << std::endl;

    // Create TCPStore (initiator is master, starts the server)
    tcp_store::TCPStore store("127.0.0.1", 9998, true);

    // Initiator always uses GPU 0 (for demo simplicity)
    CUDA_CHECK(cudaSetDevice(INITIATOR_DEVICE));

    // Create NIXL agent (pattern from nixl_example.cpp and EP framework)
    auto agent = std::make_unique<nixlAgent>("initiator", nixlAgentConfig(true));

    // Get UCX backend parameters
    nixl_mem_list_t mems;
    nixl_b_params_t init_params;
    NIXL_CHECK(agent->getPluginParams("UCX", mems, init_params), "getPluginParams");

    // UCX configuration for device API
    init_params["num_workers"] = "1";
    init_params["ucx_error_handling_mode"] = "none";

    nixlBackendH* backend = nullptr;
    NIXL_CHECK(agent->createBackend("UCX", init_params, backend), "createBackend");

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend);

    // Allocate GPU memory (fill with test pattern)
    void* data_ptr = nullptr;
    void* signal_ptr = nullptr;

    CUDA_CHECK(cudaMalloc(&data_ptr, cfg.size));
    CUDA_CHECK(cudaMemset(data_ptr, INITIATOR_FILL_VALUE, cfg.size));

    size_t signal_size = 0;
    NIXL_CHECK(agent->getGpuSignalSize(signal_size, &extra_params), "getGpuSignalSize");

    CUDA_CHECK(cudaMalloc(&signal_ptr, signal_size));
    CUDA_CHECK(cudaMemset(signal_ptr, 0, signal_size));

    uint64_t checksum = cfg.size * INITIATOR_FILL_VALUE;
    std::cout << "[initiator] Data checksum: " << checksum << std::endl;

    // Register memory with NIXL
    nixl_reg_dlist_t data_reg(VRAM_SEG);
    data_reg.addDesc(nixlBlobDesc((uintptr_t)data_ptr, cfg.size, INITIATOR_DEVICE, "initiator_data"));
    NIXL_CHECK(agent->registerMem(data_reg, &extra_params), "registerMem data");

    nixl_reg_dlist_t signal_reg(VRAM_SEG);
    signal_reg.addDesc(nixlBlobDesc((uintptr_t)signal_ptr, signal_size, INITIATOR_DEVICE, "initiator_signal"));
    NIXL_CHECK(agent->registerMem(signal_reg, &extra_params), "registerMem signal");

    // Get local metadata
    std::string local_meta;
    NIXL_CHECK(agent->getLocalMD(local_meta), "getLocalMD");

    // Publish metadata using TCPStore
    store.set("NIXL_2PROC/initiator_meta", local_meta);

    std::cout << "[initiator] Published metadata, waiting for target..." << std::endl;

    // Wait for target metadata and establish connection
    if (!store.wait("NIXL_2PROC/target_meta", 30000)) {
        throw std::runtime_error("Timeout waiting for target metadata");
    }
    std::string remote_meta = store.get("NIXL_2PROC/target_meta");

    std::string target_name;
    NIXL_CHECK(agent->loadRemoteMD(remote_meta, target_name), "loadRemoteMD");

    // Wait for remote agent to be ready
    nixl_xfer_dlist_t empty_descs(VRAM_SEG);
    while (agent->checkRemoteMD(target_name, empty_descs) != NIXL_SUCCESS) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "[initiator] Connected to target" << std::endl;

    // Wait for target descriptors
    if (!store.wait("NIXL_2PROC/target_data_descs", 30000) ||
        !store.wait("NIXL_2PROC/target_signal_descs", 30000)) {
        throw std::runtime_error("Timeout waiting for target descriptors");
    }
    std::string target_data_descs = store.get("NIXL_2PROC/target_data_descs");
    std::string target_signal_descs = store.get("NIXL_2PROC/target_signal_descs");

    // Create write requests
    nixl_xfer_dlist_t local_data(VRAM_SEG);
    local_data.addDesc(nixlBasicDesc((uintptr_t)data_ptr, cfg.size, INITIATOR_DEVICE));

    nixlSerDes remote_data_serdes;
    remote_data_serdes.importStr(target_data_descs);
    nixl_xfer_dlist_t remote_data(&remote_data_serdes);

    nixlXferReqH* data_xfer_req = nullptr;
    NIXL_CHECK(agent->createXferReq(NIXL_WRITE, local_data, remote_data, target_name,
                                    data_xfer_req, &extra_params), "createXferReq data");

    nixlGpuXferReqH data_gpu_req = nullptr;
    NIXL_CHECK(agent->createGpuXferReq(*data_xfer_req, data_gpu_req), "createGpuXferReq data");

    nixl_xfer_dlist_t local_signal(VRAM_SEG);
    local_signal.addDesc(nixlBasicDesc((uintptr_t)signal_ptr, signal_size, INITIATOR_DEVICE));

    nixlSerDes remote_signal_serdes;
    remote_signal_serdes.importStr(target_signal_descs);
    nixl_xfer_dlist_t remote_signal(&remote_signal_serdes);

    nixlXferReqH* signal_xfer_req = nullptr;
    NIXL_CHECK(agent->createXferReq(NIXL_WRITE, local_signal, remote_signal, target_name,
                                    signal_xfer_req, &extra_params), "createXferReq signal");

    nixlGpuXferReqH signal_gpu_req = nullptr;
    NIXL_CHECK(agent->createGpuXferReq(*signal_xfer_req, signal_gpu_req), "createGpuXferReq signal");

    // Prepare GPU request handles array (kernel needs device memory, not host stack pointer)
    uintptr_t* data_req_handles = nullptr;
    CUDA_CHECK(cudaMalloc(&data_req_handles, sizeof(uintptr_t)));
    uintptr_t data_gpu_req_val = reinterpret_cast<uintptr_t>(data_gpu_req);
    CUDA_CHECK(cudaMemcpy(data_req_handles, &data_gpu_req_val, sizeof(uintptr_t),
                          cudaMemcpyHostToDevice));

    // Launch GPU kernel to post write + signal
    std::cout << "[initiator] Transferring data via GPU kernel..." << std::endl;
    launch_post_write_and_signal(
        (uintptr_t)data_req_handles,                     // GPU handles array
        1,                                                // Number of handles
        reinterpret_cast<uintptr_t>(signal_gpu_req),    // Signal handle
        (uintptr_t)signal_ptr,                           // Signal memory
        cfg.size,                                         // Transfer size
        0,                                                // THREAD level cooperation
        1,                                                // Single thread
        1,                                                // One write per signal
        0                                                 // Default stream
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[initiator] Transfer complete!" << std::endl;

    // Cleanup
    agent->releaseGpuXferReq(data_gpu_req);
    agent->releaseGpuXferReq(signal_gpu_req);
    NIXL_CHECK(agent->releaseXferReq(data_xfer_req), "releaseXferReq data");
    NIXL_CHECK(agent->releaseXferReq(signal_xfer_req), "releaseXferReq signal");
    NIXL_CHECK(agent->deregisterMem(data_reg, &extra_params), "deregisterMem data");
    NIXL_CHECK(agent->deregisterMem(signal_reg, &extra_params), "deregisterMem signal");
    NIXL_CHECK(agent->invalidateRemoteMD(target_name), "invalidateRemoteMD");

    CUDA_CHECK(cudaFree(data_req_handles));
    CUDA_CHECK(cudaFree(data_ptr));
    CUDA_CHECK(cudaFree(signal_ptr));

    // Clean up TCPStore keys (following EP pattern)
    store.delete_key("NIXL_2PROC/initiator_meta");

    std::cout << "[initiator] Complete!" << std::endl;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    if (cfg.mode == "target") {
        run_target(cfg);
    } else if (cfg.mode == "initiator") {
        run_initiator(cfg);
    } else {
        std::cerr << "Invalid mode: " << cfg.mode << std::endl;
        return 1;
    }

    return 0;
}
