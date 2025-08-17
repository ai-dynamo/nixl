/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Standalone test application - no gtest dependencies
#include <iostream>
#include <string>
#include <algorithm>
#include <thread>
#include <chrono>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include <cassert>
#include "stream/metadata_stream.h"
#include "serdes/serdes.h"
#include <mutex>
#include <vector>

#include <cuda_runtime.h>
#include "ucx/nixl_gdaki_device.cuh"

//#define NUM_TRANSFERS 32
#define NUM_THREADS 1
//#define SIZE 16 * 1024
#define MEM_VAL 0xBB
//#define DEV_ID 0
#define N_ITERS 10000

uint32_t DEV_ID = 0;
uint32_t SIZE = 16 * 1024;
uint32_t NUM_TRANSFERS = 32;
/**
 * This test does p2p from using PUT.
 * intitator -> target so the metadata and
 * desc list needs to move from
 * target to initiator
 */

struct SharedNotificationState {
    std::mutex mtx;
    std::vector<nixlSerDes> remote_serdes;
};

static const std::string target("target");
static const std::string initiator("initiator");

#define UCS_NSEC_PER_SEC   1000000000ul
#define NS_TO_SEC(ns)      ((ns)*1.0 / (UCS_NSEC_PER_SEC))
__device__ inline unsigned long long GdakiGetTimeNs()
{
    unsigned long long globaltimer;
    // 64-bit GPU global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

template<nixl_gpu_xfer_coordination_level_t level>
__global__ void
GdakiPostGpuXferReqLaunchKernel(nixlGpuXferReqH *req_hdnl, size_t num_iters,
                                uint64_t signal_inc,
                                unsigned long long *start_time_ptr,
                                unsigned long long *end_time_ptr) {
    __shared__ nixlGpuXferStatusH xfer_status;
    nixl_status_t status;

    if (threadIdx.x == 0) {
        unsigned long long start_time = GdakiGetTimeNs();
        *start_time_ptr = start_time;
    }

    for (size_t i = 0; i < num_iters; ++i) {
        status = nixlPostGpuXferReq<level>(req_hdnl, signal_inc, &xfer_status);
        if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
            printf("Failed to execute batch %d\n", status);
            return;
        }

        status = nixlGetGpuXferReqStatus<level>(&xfer_status);
        if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
            printf("Failed to progress request %d\n", status);
            return;
        }
    }

    if (threadIdx.x == 0) {
        unsigned long long end_time = GdakiGetTimeNs();
        *end_time_ptr = end_time;
    }
}

template<nixl_gpu_xfer_coordination_level_t level>
nixl_status_t
GdakiPostGpuXferReqwKernel(unsigned num_threads, nixlGpuXferReqH *req_hdnl,
                           size_t num_iters, uint64_t signal_inc,
                           unsigned long long *start_time_ptr,
                           unsigned long long *end_time_ptr) {
    nixl_status_t ret = NIXL_SUCCESS;
    cudaError_t err;

    GdakiPostGpuXferReqLaunchKernel<level><<<1, num_threads>>>(req_hdnl, num_iters, signal_inc, start_time_ptr, end_time_ptr);

    err = cudaDeviceSynchronize(); // Wait until kernel prints are visible
    if (err != cudaSuccess) {
        printf("Failed to synchronize: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    return ret;
}

class MemBuffer : std::shared_ptr<void> {
public:
    MemBuffer(size_t size, nixl_mem_t mem_type) :
        std::shared_ptr<void>(allocate(size, mem_type),
                              [mem_type](void *ptr) {
                                  release(ptr, mem_type);
                              }),
        size(size)
    {
    }

    operator uintptr_t() const
    {
        return reinterpret_cast<uintptr_t>(get());
    }

    size_t getSize() const
    {
        return size;
    }

private:
    static void *allocate(size_t size, nixl_mem_t mem_type)
    {
        void *ptr;
        return cudaSuccess == cudaMalloc(&ptr, size)? ptr : nullptr;
    }

    static void release(void *ptr, nixl_mem_t mem_type)
    {
        cudaFree(ptr);
    }

    const size_t size;
};

// Timing helper functions
void initTiming(unsigned long long** start_time_ptr, unsigned long long** end_time_ptr) {
    cudaMalloc(start_time_ptr, sizeof(unsigned long long));
    cudaMalloc(end_time_ptr, sizeof(unsigned long long));
    cudaMemset(*start_time_ptr, 0, sizeof(unsigned long long));
    cudaMemset(*end_time_ptr, 0, sizeof(unsigned long long));
}

void getTiming(unsigned long long* start_time_ptr, unsigned long long* end_time_ptr,
               unsigned long long& start_time_cpu, unsigned long long& end_time_cpu) {
    cudaMemcpy(&start_time_cpu, start_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end_time_cpu, end_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
}

void logResults(size_t size, size_t count, size_t num_iters,
               unsigned long long start_time_cpu, unsigned long long end_time_cpu) {
    auto total_time = NS_TO_SEC(end_time_cpu - start_time_cpu);
    double total_size = size * count * num_iters;
    auto bandwidth = total_size / total_time / (1024 * 1024);
    std::cout << "[ INFO     ] Average Results: " << size << "x" << count
             << "x" << num_iters << "=" << total_size << " bytes in "
             << total_time << " seconds " << "(" << bandwidth << " MB/s)" << std::endl;
}

static std::vector<MemBuffer> initMem(nixlAgent &agent,
                                                       nixl_reg_dlist_t &vram,
                                                       nixl_opt_args_t *extra_params,
                                                       uint8_t val) {
    std::vector<MemBuffer> addrs;

    //one extra for the signal
    for (uint32_t i = 0; i < NUM_TRANSFERS; i++) {

        addrs.emplace_back(SIZE, VRAM_SEG);
        cudaMemset(reinterpret_cast<void*>(static_cast<uintptr_t>(addrs[i])), val, SIZE);
        vram.addDesc(nixlBlobDesc((uintptr_t)(addrs[i]), SIZE, 0, ""));
    }

    //handle signal differently
    addrs.emplace_back(sizeof(uint64_t), VRAM_SEG);
    cudaMemset(reinterpret_cast<void*>(static_cast<uintptr_t>(addrs.back())), 0, sizeof(uint64_t));
    vram.addDesc(nixlBlobDesc((uintptr_t)(addrs.back()), sizeof(uint64_t), 0, ""));

    agent.registerMem(vram, extra_params);

    return addrs;
}

template<nixl_gpu_xfer_coordination_level_t level>
void doTransfer(nixlAgent &from,
                const std::string &to_name, size_t size,
                size_t count, size_t num_threads,
                nixl_xfer_dlist_t src_buffers,
                nixl_xfer_dlist_t dst_buffers,
                uint64_t signal_addr, nixlBackendH *ucx)
{
    nixl_opt_args_t extra_params = {};
    extra_params.hasNotif = true;
    extra_params.notifMsg = "notification";
    unsigned long long start_time_cpu = 0;
    unsigned long long end_time_cpu = 0;
    unsigned long long *start_time_ptr = nullptr;
    unsigned long long *end_time_ptr = nullptr;
    nixl_status_t status;
    uint64_t signal_inc;

    std::cout << "Setting signal buffer address " << static_cast<uintptr_t>(signal_addr) << " and device id " << DEV_ID;
    extra_params.signal_addr = signal_addr;
    extra_params.signal_dev_id = 0;
    signal_inc = 1;

    extra_params.backends.push_back(ucx);

    nixlXferReqH *xfer_req = nullptr;
    std::cout << "Creating xfer request with source and destination buffers";

    status = from.createXferReq(
                NIXL_WRITE,
                src_buffers,
                dst_buffers, to_name,
                xfer_req, &extra_params);

    assert(status == NIXL_SUCCESS);
    assert(xfer_req != nullptr);
    std::cout << "[ INFO     ] Prep xfer request" << std::endl;

    nixlGpuXferReqH *gpu_req_hndl = from.exportXferReqtoGPU(xfer_req);
    assert(gpu_req_hndl != nullptr);
    std::cout << "[ INFO     ] Exported gpu request: " << static_cast<void*>(gpu_req_hndl) << std::endl;

    // Set default batch execution parameters
    const size_t num_iters = N_ITERS;

    initTiming(&start_time_ptr, &end_time_ptr);

    status = GdakiPostGpuXferReqwKernel<level>(num_threads, gpu_req_hndl, num_iters, signal_inc, start_time_ptr, end_time_ptr);
    assert(status == NIXL_SUCCESS);

    getTiming(start_time_ptr, end_time_ptr, start_time_cpu, end_time_cpu);
    logResults(size, count, num_iters, start_time_cpu, end_time_cpu);

    cudaFree(start_time_ptr);
    cudaFree(end_time_ptr);

    status = from.releaseXferReqtoGPU(xfer_req);
    assert(status == NIXL_SUCCESS);

    status = from.releaseXferReq(xfer_req);
    assert(status == NIXL_SUCCESS);
}

static void targetThread(nixlAgent &agent, nixl_opt_args_t *extra_params, int thread_id) {
    nixl_reg_dlist_t vram_for_ucx(VRAM_SEG);
    auto addrs = initMem(agent, vram_for_ucx, extra_params, 0);

    nixl_blob_t tgt_metadata;
    agent.getLocalMD(tgt_metadata);

    std::cout << "Thread " << thread_id << " Start Control Path metadata exchanges\n";

    std::cout << "Thread " << thread_id << " Desc List from Target to Initiator\n";
    vram_for_ucx.print();

    /** Only send desc list */
    nixlSerDes serdes;
    assert(vram_for_ucx.trim().serialize(&serdes) == NIXL_SUCCESS);

    std::cout << "Thread " << thread_id << " Wait for initiator and then send xfer descs\n";
    std::string message = serdes.exportStr();
    while (agent.genNotif(initiator, message, extra_params) != NIXL_SUCCESS);
    std::cout << "Thread " << thread_id << " End Control Path metadata exchanges\n";

    std::cout << "Thread " << thread_id << " Start Data Path Exchanges\n";
    std::cout << "Thread " << thread_id << " Waiting to receive Data from Initiator\n";

    //Remove signal from end of list now that metadata has been sent
    void* signal_addr = reinterpret_cast<void*>(static_cast<uintptr_t>(addrs.back()));
    addrs.pop_back();

    std::cout << "Thread " << thread_id << " Waiting for all iterations to finish...\n";
    uint64_t count = 0;
    while (count < N_ITERS) {
	cudaMemcpy(&count, signal_addr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    bool rc = false;
    for (int n_tries = 0; !rc && n_tries < 500; n_tries++) {
        //Only works with progress thread now, as backend is protected
        /** Sanity Check */
        rc = std::all_of(addrs.begin(), addrs.end(), [](auto &addr) {
    	    uint8_t chk_buffer[SIZE];
	    cudaMemcpy(chk_buffer, reinterpret_cast<void*>(static_cast<uintptr_t>(addr)), SIZE, cudaMemcpyDeviceToHost);
            return std::all_of(chk_buffer, chk_buffer + SIZE, [](int x) {
                return x == MEM_VAL;
            });
        });
        if (!rc)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!rc)
        std::cerr << "Thread " << thread_id << " UCX Transfer failed, buffers are different\n";
    else
        std::cout << "Thread " << thread_id << " Transfer completed and Buffers match with Initiator\n"
                  << "Thread " << thread_id << " UCX Transfer Success!!!\n";

    //std::cout << "Thread " << thread_id << " waiting for iterations...\n";
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    std::cout << "Thread " << thread_id << " Cleanup..\n";
    agent.deregisterMem(vram_for_ucx, extra_params);
}

static void initiatorThread(nixlAgent &agent, nixl_opt_args_t *extra_params,
                          const std::string &target_ip, int target_port, int thread_id,
                          SharedNotificationState &shared_state) {
    nixl_reg_dlist_t vram_for_ucx(VRAM_SEG);
    auto addrs = initMem(agent, vram_for_ucx, extra_params, MEM_VAL);

    std::cout << "Thread " << thread_id << " Start Control Path metadata exchanges\n";
    std::cout << "Thread " << thread_id << " Exchange metadata with Target\n";

    nixl_opt_args_t md_extra_params;
    md_extra_params.ipAddr = target_ip;
    md_extra_params.port = target_port;

    agent.fetchRemoteMD(target, &md_extra_params);

    agent.sendLocalMD(&md_extra_params);

    // Wait for notifications and populate shared state
    while (true) {
        {
            std::lock_guard<std::mutex> lock(shared_state.mtx);
            if (shared_state.remote_serdes.size() >= NUM_THREADS) {
                break;
            }
        }

        nixl_notifs_t notifs;
        nixl_status_t ret = agent.getNotifs(notifs, extra_params);
        assert(ret >= 0);

        if (notifs.size() > 0) {
            std::lock_guard<std::mutex> lock(shared_state.mtx);
            for (const auto &notif : notifs[target]) {
                nixlSerDes serdes;
                serdes.importStr(notif);
                shared_state.remote_serdes.push_back(serdes);
            }
        }
    }

    // Get our thread's serdes instance
    nixlSerDes remote_serdes;
    {
        std::lock_guard<std::mutex> lock(shared_state.mtx);
        remote_serdes = shared_state.remote_serdes[thread_id];
    }

    std::cout << "Thread " << thread_id << " Verify Deserialized Target's Desc List at Initiator\n";
    nixl_xfer_dlist_t vram_target_ucx(&remote_serdes);
    nixl_xfer_dlist_t vram_initiator_ucx = vram_for_ucx.trim();
    vram_target_ucx.print();

    std::cout << "Thread " << thread_id << " End Control Path metadata exchanges\n";
    std::cout << "Thread " << thread_id << " Start Data Path Exchanges\n\n";
    std::cout << "Thread " << thread_id << " Create transfer request with UCX backend\n";

    // Need to wait to createXfer
    // UCX AM with desc list is faster than listener thread can recv/load MD with sockets
    // Will be deprecated with ETCD or callbacks
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    //nixlXferReqH *treq;
    //nixl_status_t ret = NIXL_SUCCESS;
    //do {
        //this should fail until metadata has been received by listener thread
    //    ret = agent.createXferReq(NIXL_WRITE, vram_initiator_ucx, vram_target_ucx,
    //                              target, treq, extra_params);
    //} while (ret == NIXL_ERR_NOT_FOUND);

    //last entry is signal, get and then delete
    uint64_t signal_addr = vram_target_ucx[NUM_TRANSFERS].addr;
    vram_target_ucx.remDesc(NUM_TRANSFERS);
    //local signal can be discarded
    vram_initiator_ucx.remDesc(NUM_TRANSFERS);

    doTransfer<NIXL_GPU_XFER_COORDINATION_BLOCK>(
            agent, target, SIZE, NUM_TRANSFERS, 32,
            vram_initiator_ucx, vram_target_ucx, signal_addr, extra_params->backends[0]);

    std::cout << "Thread " << thread_id << " Post the request with UCX backend\n";
    //ret = agent.postXferReq(treq);
    std::cout << "Thread " << thread_id << " Initiator posted Data Path transfer\n";
    std::cout << "Thread " << thread_id << " Waiting for completion\n";

    //while (ret != NIXL_SUCCESS) {
    //    ret = agent.getXferStatus(treq);
    //    assert(ret >= 0);
    //}
    std::cout << "Thread " << thread_id << " Completed Sending Data using UCX backend\n";
    //agent.releaseXferReq(treq);
    agent.invalidateLocalMD(&md_extra_params);

    std::cout << "Thread " << thread_id << " Cleanup..\n";
    agent.deregisterMem(vram_for_ucx, extra_params);
}

static void runTarget(const std::string &ip, int port, nixl_thread_sync_t sync_mode) {
    nixlAgentConfig cfg(true, true, port, sync_mode);

    std::cout << "Starting Agent for target\n";
    nixlAgent agent(target, cfg);

    nixl_b_params_t params = {
        { "num_workers", "4" },
    };
    nixlBackendH *ucx;
    agent.createBackend("UCX", params, ucx);

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
        threads.emplace_back(targetThread, std::ref(agent), &extra_params, i);

    for (auto &thread : threads)
        thread.join();
}

static void runInitiator(const std::string &target_ip, int target_port, nixl_thread_sync_t sync_mode) {
    nixlAgentConfig cfg(true, true, 0, sync_mode);

    std::cout << "Starting Agent for initiator\n";
    nixlAgent agent(initiator, cfg);

    nixl_b_params_t params = {
        { "num_workers", "4" },
    };
    nixlBackendH *ucx;
    agent.createBackend("UCX", params, ucx);

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);

    SharedNotificationState shared_state;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
        threads.emplace_back(initiatorThread, std::ref(agent), &extra_params,
                             target_ip, target_port, i, std::ref(shared_state));

    for (auto &thread : threads)
        thread.join();
}

int main(int argc, char *argv[]) {
    /** Argument Parsing */
    if (argc < 4) {
        std::cout <<"Enter the required arguments\n" << std::endl;
        std::cout <<"<Role> " <<"<Target IP> <Target Port>"
                  << std::endl;
        exit(-1);
    }

    std::string role = std::string(argv[1]);
    const char  *target_ip   = argv[2];
    int         target_port = std::stoi(argv[3]);

    std::transform(role.begin(), role.end(), role.begin(), ::tolower);

    if (!role.compare(initiator) && !role.compare(target)) {
            std::cerr << "Invalid role. Use 'initiator' or 'target'."
                      << "Currently "<< role <<std::endl;
            return 1;
    }

    if (argc >= 5) DEV_ID = atoi(argv[4]);
    if (argc >= 6) NUM_TRANSFERS = atoi(argv[5]);
    if (argc >= 7) SIZE = atoi(argv[6]);
    
    auto sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    if (argc == 8) {
        std::string sync_mode_str{argv[7]};
        std::transform(sync_mode_str.begin(), sync_mode_str.end(), sync_mode_str.begin(), ::tolower);
        if (sync_mode_str == "rw") {
            sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
            std::cout << "Using RW sync mode" << std::endl;
        } else if (sync_mode_str == "strict") {
            sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
            std::cout << "Using Strict sync mode" << std::endl;
        } else {
            std::cerr << "Invalid sync mode. Use 'rw' or 'strict'." << std::endl;
            return 1;
        }
    }


    /*** End - Argument Parsing */

    if (role == target)
        runTarget(target_ip, target_port, sync_mode);
    else
        runInitiator(target_ip, target_port, sync_mode);

    return 0;
}
