/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Simplified single-file GPU-initiated RDMA example.
 * Demonstrates NIXL device API with write-and-signal pattern in ~300 lines.
 *
 * Usage:
 *   # Terminal 1 (target - receives data)
 *   ./simple_write_unified --mode target --size 1048576
 *
 *   # Terminal 2 (initiator - sends data)
 *   ./simple_write_unified --mode initiator --size 1048576
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <getopt.h>

#include <cuda_runtime.h>
#include "nixl.h"
#include "nixl_device.cuh"
#include "serdes/serdes.h"

// Configuration
constexpr uint8_t FILL_VALUE = 42;
constexpr int GPU_DEVICE = 0;
constexpr int STORE_PORT = 9998;

#define CUDA_CHECK(cmd)                                                                       \
    do {                                                                                      \
        cudaError_t e = cmd;                                                                  \
        if (e != cudaSuccess) {                                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                               \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

#define NIXL_CHECK(cmd, msg)                                                            \
    do {                                                                                \
        if ((cmd) != NIXL_SUCCESS) {                                                    \
            std::cerr << "NIXL error: " << msg << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                     \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

//=============================================================================
// CUDA Kernels (Device Code)
//=============================================================================

// GPU kernel: Write data and signal completion
__global__ void
write_and_signal_kernel(uintptr_t data_req_handle,
                        uintptr_t signal_req_handle,
                        size_t size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        auto data_req = reinterpret_cast<nixlGpuXferReqH>(data_req_handle);
        auto signal_req = reinterpret_cast<nixlGpuXferReqH>(signal_req_handle);

        // Post RDMA write
        nixlGpuPostSingleWriteXferReq<nixl_gpu_level_t::THREAD>(
            data_req, 0, 0, 0, size, 0, true);

        // Signal completion (increment remote signal by 1)
        nixlGpuPostSignalXferReq<nixl_gpu_level_t::THREAD>(
            signal_req, 0, 1, 0, 0, true);
    }
}

// GPU kernel: Wait for signal
__global__ void
wait_for_signal_kernel(const void *signal_ptr, uint64_t expected_value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (nixlGpuReadSignal<nixl_gpu_level_t::THREAD>(signal_ptr) < expected_value) {
            // Busy-wait for signal
        }
    }
}

//=============================================================================
// Host Code
//=============================================================================

struct Config {
    std::string mode = "initiator";
    size_t size = 1024 * 1024;
    std::string peer_ip = "127.0.0.1";
};

void
print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --mode <initiator|target>   Role (default: initiator)\n"
              << "  --size <bytes>              Buffer size (default: 1MB)\n"
              << "  --peer-ip <ip>              Peer IP address (default: 127.0.0.1)\n"
              << "  --help                      Show help\n";
}

Config
parse_args(int argc, char **argv) {
    Config cfg;
    static struct option opts[] = {{"mode", required_argument, 0, 'm'},
                                   {"size", required_argument, 0, 's'},
                                   {"peer-ip", required_argument, 0, 'p'},
                                   {"help", no_argument, 0, 'h'},
                                   {0, 0, 0, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "m:s:p:h", opts, nullptr)) != -1) {
        switch (opt) {
        case 'm': cfg.mode = optarg; break;
        case 's': cfg.size = std::stoull(optarg); break;
        case 'p': cfg.peer_ip = optarg; break;
        case 'h': print_usage(argv[0]); exit(0);
        default: print_usage(argv[0]); exit(1);
        }
    }
    return cfg;
}

// Simple key-value store using notifications
class SimpleStore {
public:
    SimpleStore(nixlAgent *agent, const std::string &peer)
        : agent_(agent), peer_(peer) {}

    void set(const std::string &key, const std::string &value) {
        std::string msg = "SET:" + key + ":" + value;
        agent_->sendNotif(peer_, msg.c_str(), msg.size());
    }

    std::string get(const std::string &key, int timeout_ms = 30000) {
        auto start = std::chrono::steady_clock::now();
        std::string prefix = "SET:" + key + ":";

        while (true) {
            std::list<std::pair<std::string, std::string>> notifs;
            agent_->getNewNotifs(notifs);

            for (const auto &notif : notifs) {
                if (notif.second.substr(0, prefix.length()) == prefix) {
                    return notif.second.substr(prefix.length());
                }
            }

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeout_ms) {
                throw std::runtime_error("Timeout waiting for key: " + key);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    nixlAgent *agent_;
    std::string peer_;
};

void
run_target(const Config &cfg) {
    std::cout << "[target] Starting..." << std::endl;
    CUDA_CHECK(cudaSetDevice(GPU_DEVICE));

    // Create NIXL agent with listening enabled
    nixlAgentConfig agent_cfg(true); // enable_device_api
    agent_cfg.listen_for_peers = true;
    agent_cfg.listen_port = STORE_PORT;
    auto agent = std::make_unique<nixlAgent>("target", agent_cfg);

    // Setup UCX backend
    nixl_mem_list_t mems;
    nixl_b_params_t init_params;
    NIXL_CHECK(agent->getPluginParams("UCX", mems, init_params), "getPluginParams");
    init_params["num_workers"] = "1";
    init_params["ucx_error_handling_mode"] = "none";

    nixlBackendH *backend = nullptr;
    NIXL_CHECK(agent->createBackend("UCX", init_params, backend), "createBackend");

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend);

    // Allocate GPU memory
    void *data_ptr = nullptr;
    void *signal_ptr = nullptr;
    size_t signal_size = 0;

    CUDA_CHECK(cudaMalloc(&data_ptr, cfg.size));
    CUDA_CHECK(cudaMemset(data_ptr, 0, cfg.size));
    NIXL_CHECK(agent->getGpuSignalSize(signal_size, &extra_params), "getGpuSignalSize");
    CUDA_CHECK(cudaMalloc(&signal_ptr, signal_size));
    CUDA_CHECK(cudaMemset(signal_ptr, 0, signal_size));

    // Register memory
    nixl_reg_dlist_t data_reg(VRAM_SEG);
    data_reg.addDesc(nixlBlobDesc((uintptr_t)data_ptr, cfg.size, GPU_DEVICE, "data"));
    NIXL_CHECK(agent->registerMem(data_reg, &extra_params), "registerMem data");

    nixl_reg_dlist_t signal_reg(VRAM_SEG);
    signal_reg.addDesc(nixlBlobDesc((uintptr_t)signal_ptr, signal_size, GPU_DEVICE, "signal"));
    NIXL_CHECK(agent->registerMem(signal_reg, &extra_params), "registerMem signal");
    NIXL_CHECK(agent->prepGpuSignal(signal_reg, &extra_params), "prepGpuSignal");

    std::cout << "[target] Waiting for initiator to connect..." << std::endl;

    // Wait for initiator to connect
    std::string initiator_name;
    while (true) {
        std::list<std::string> peers;
        agent->getConnectedPeers(peers);
        if (!peers.empty()) {
            initiator_name = peers.front();
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "[target] Connected to: " << initiator_name << std::endl;

    // Exchange descriptors via notifications
    SimpleStore store(agent.get(), initiator_name);

    nixl_xfer_dlist_t data_xfer(VRAM_SEG);
    data_xfer.addDesc(nixlBasicDesc((uintptr_t)data_ptr, cfg.size, GPU_DEVICE));
    nixlSerDes data_serdes;
    NIXL_CHECK(data_xfer.serialize(&data_serdes), "serialize data");
    store.set("target_data", data_serdes.exportStr());

    nixl_xfer_dlist_t signal_xfer(VRAM_SEG);
    signal_xfer.addDesc(nixlBasicDesc((uintptr_t)signal_ptr, signal_size, GPU_DEVICE));
    nixlSerDes signal_serdes;
    NIXL_CHECK(signal_xfer.serialize(&signal_serdes), "serialize signal");
    store.set("target_signal", signal_serdes.exportStr());

    std::cout << "[target] Waiting for data..." << std::endl;

    // Wait for GPU signal
    wait_for_signal_kernel<<<1, 1>>>(signal_ptr, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[target] Data received! Verifying..." << std::endl;

    // Verify data
    std::vector<uint8_t> host_data(cfg.size);
    CUDA_CHECK(cudaMemcpy(host_data.data(), data_ptr, cfg.size, cudaMemcpyDeviceToHost));

    uint64_t checksum = 0;
    for (uint8_t val : host_data) {
        checksum += val;
    }

    uint64_t expected = cfg.size * FILL_VALUE;
    if (checksum == expected) {
        std::cout << "[target] ✓ Checksum OK: " << checksum << std::endl;
    } else {
        std::cerr << "[target] ✗ Checksum FAIL: " << checksum << " (expected " << expected << ")" << std::endl;
    }

    // Cleanup
    NIXL_CHECK(agent->deregisterMem(data_reg, &extra_params), "deregisterMem data");
    NIXL_CHECK(agent->deregisterMem(signal_reg, &extra_params), "deregisterMem signal");
    CUDA_CHECK(cudaFree(data_ptr));
    CUDA_CHECK(cudaFree(signal_ptr));

    std::cout << "[target] Complete!" << std::endl;
}

void
run_initiator(const Config &cfg) {
    std::cout << "[initiator] Starting..." << std::endl;
    CUDA_CHECK(cudaSetDevice(GPU_DEVICE));

    // Create NIXL agent (no listening, will connect to target)
    nixlAgentConfig agent_cfg(true);
    agent_cfg.listen_for_peers = false;
    auto agent = std::make_unique<nixlAgent>("initiator", agent_cfg);

    // Setup UCX backend
    nixl_mem_list_t mems;
    nixl_b_params_t init_params;
    NIXL_CHECK(agent->getPluginParams("UCX", mems, init_params), "getPluginParams");
    init_params["num_workers"] = "1";
    init_params["ucx_error_handling_mode"] = "none";

    nixlBackendH *backend = nullptr;
    NIXL_CHECK(agent->createBackend("UCX", init_params, backend), "createBackend");

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend);

    // Connect to target
    std::cout << "[initiator] Connecting to target at " << cfg.peer_ip << ":" << STORE_PORT << std::endl;
    std::string target_name;
    NIXL_CHECK(agent->connectToPeer(cfg.peer_ip, STORE_PORT, "target", target_name),
               "connectToPeer");

    std::cout << "[initiator] Connected to: " << target_name << std::endl;

    // Allocate GPU memory and fill with pattern
    void *data_ptr = nullptr;
    size_t signal_size = 0;

    CUDA_CHECK(cudaMalloc(&data_ptr, cfg.size));
    CUDA_CHECK(cudaMemset(data_ptr, FILL_VALUE, cfg.size));
    NIXL_CHECK(agent->getGpuSignalSize(signal_size, &extra_params), "getGpuSignalSize");

    // Register local memory
    nixl_reg_dlist_t data_reg(VRAM_SEG);
    data_reg.addDesc(nixlBlobDesc((uintptr_t)data_ptr, cfg.size, GPU_DEVICE, "data"));
    NIXL_CHECK(agent->registerMem(data_reg, &extra_params), "registerMem data");

    // Get remote descriptors via notifications
    SimpleStore store(agent.get(), target_name);
    std::string remote_data_str = store.get("target_data");
    std::string remote_signal_str = store.get("target_signal");

    nixl_xfer_dlist_t remote_data(VRAM_SEG);
    nixlSerDes data_serdes(remote_data_str);
    NIXL_CHECK(remote_data.deserialize(&data_serdes), "deserialize data");

    nixl_xfer_dlist_t remote_signal(VRAM_SEG);
    nixlSerDes signal_serdes(remote_signal_str);
    NIXL_CHECK(remote_signal.deserialize(&signal_serdes), "deserialize signal");

    // Create transfer requests
    nixlXferReqH *data_req = nullptr;
    NIXL_CHECK(agent->createXferReq(NIXL_WRITE, data_reg, remote_data, target_name,
                                     data_req, &extra_params), "createXferReq data");

    nixlXferReqH *signal_req = nullptr;
    NIXL_CHECK(agent->createXferReq(NIXL_WRITE, remote_signal, remote_signal, target_name,
                                     signal_req, &extra_params), "createXferReq signal");

    // Create GPU handles
    nixlGpuXferReqH data_gpu_req = nullptr;
    NIXL_CHECK(agent->createGpuXferReq(*data_req, data_gpu_req), "createGpuXferReq data");

    nixlGpuXferReqH signal_gpu_req = nullptr;
    NIXL_CHECK(agent->createGpuXferReq(*signal_req, signal_gpu_req), "createGpuXferReq signal");

    std::cout << "[initiator] Launching GPU kernel to write " << cfg.size << " bytes..." << std::endl;

    // Launch GPU kernel to perform RDMA write + signal
    write_and_signal_kernel<<<1, 1>>>(
        reinterpret_cast<uintptr_t>(data_gpu_req),
        reinterpret_cast<uintptr_t>(signal_gpu_req),
        cfg.size);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[initiator] Transfer complete!" << std::endl;

    // Cleanup
    NIXL_CHECK(agent->deregisterMem(data_reg, &extra_params), "deregisterMem");
    CUDA_CHECK(cudaFree(data_ptr));

    std::cout << "[initiator] Complete!" << std::endl;
}

int
main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    try {
        if (cfg.mode == "target") {
            run_target(cfg);
        } else if (cfg.mode == "initiator") {
            run_initiator(cfg);
        } else {
            std::cerr << "Invalid mode: " << cfg.mode << std::endl;
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
