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
#include <iostream>
#include <string>
#include <algorithm>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cerrno>
#include <cstring>

#define NUM_TRANSFERS 1
#define SIZE 1024

static void checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    // nixl_status_t           ret = NIXL_SUCCESS;
    
    std::string             role;
    // int                     status = 0;
    // int                     i;
    // int                     fd[NUM_TRANSFERS];
    void                    *addr_local[NUM_TRANSFERS];
    void                    *addr_remote[NUM_TRANSFERS];
    nixlAgentConfig         cfg(true);
    nixl_b_params_t         params;
    nixlBlobDesc            buf_local[NUM_TRANSFERS];
    nixlBlobDesc            buf_remote[NUM_TRANSFERS];
    // nixlBlobDesc            ftrans[NUM_TRANSFERS];
    nixlBackendH            *doca;
    nixlBlobDesc desc;
    nixlXferReqH            *treq;
    std::string             name = "DOCA";
    nixl_opt_args_t extra_params;
    nixl_reg_dlist_t dram_for_doca_local(DRAM_SEG);
    nixl_reg_dlist_t dram_for_doca_remote(DRAM_SEG);
    cudaStream_t stream;
    nixl_status_t status;
    std::string target_name;
    std::string target_metadata;
    std::string tgt_metadata_init;

    std::cout << "Starting Agent for " << "DOCA Test Agent" << "\n";

    nixlAgent agent("DOCAUnitTest", cfg);

    params["network_devices"] = "mlx5_0";
    params["gpu_devices"] = "8A:00.0";

    // To also test the decision making of createXferReq
    agent.createBackend(name, params, doca);
    if (doca == nullptr) {
        std::cerr <<"Error creating a new backend\n";
        exit(-1);
    }

    std::cout << "DOCA Backend created\n";

    checkCudaError(cudaSetDevice(0), "Failed to set device");
    cudaFree(0);
    //Loop for all transfers

    extra_params.backends.push_back(doca);

    for (int i = 0; i < NUM_TRANSFERS; i++) {
        checkCudaError(cudaMalloc(&addr_local[i], SIZE), "Failed to allocate CUDA buffer 0");
        checkCudaError(cudaMemset(addr_local[i], 0, SIZE), "Failed to memset CUDA buffer 0");
        buf_local[i].addr  = (uintptr_t)(addr_local[i]);
        buf_local[i].len   = SIZE;
        buf_local[i].devId = 0;
        dram_for_doca_local.addDesc(buf_local[i]);
        std::cout << "GPU alloc buffer " << i << "\n";
    }
    std::cout << "DOCA registerMem local\n";
    agent.registerMem(dram_for_doca_local, &extra_params);

    /* As this is a unit test single peer, fake the remote memory with different local memory */
    for (int i = 0; i < NUM_TRANSFERS; i++) {
        checkCudaError(cudaMalloc(&addr_remote[i], SIZE), "Failed to allocate CUDA buffer 0");
        checkCudaError(cudaMemset(addr_remote[i], 0, SIZE), "Failed to memset CUDA buffer 0");
        buf_remote[i].addr  = (uintptr_t)(addr_remote[i]);
        buf_remote[i].len   = SIZE;
        buf_remote[i].devId = 0;
        dram_for_doca_remote.addDesc(buf_remote[i]);
        std::cout << "GPU alloc buffer " << i << "\n";
    }

    std::cout << "DOCA registerMem remote\n";
    agent.registerMem(dram_for_doca_remote, &extra_params);
    agent.getLocalMD(target_metadata);
    agent.loadRemoteMD(tgt_metadata_init, target_name);

    checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Failed to create CUDA stream");

    nixl_xfer_dlist_t dram_initiator_doca = dram_for_doca_local.trim();
    nixl_xfer_dlist_t dram_target_doca = dram_for_doca_remote.trim();
    extra_params.customParam = (uintptr_t)stream;

    status = agent.createXferReq(NIXL_WRITE, dram_initiator_doca, dram_target_doca,
        "DOCAUnitTest", treq, &extra_params);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Error creating transfer request\n";
        exit(-1);
    }
    std::cout << " Post the request with DOCA backend\n ";
    status = agent.postXferReq(treq);
    std::cout << " Waiting for completion\n";

    while (status != NIXL_SUCCESS) {
        status = agent.getXferStatus(treq);
        assert(status >= 0);
    }
    std::cout <<" Completed writing data using DOCA backend\n";
    agent.releaseXferReq(treq);


    std::cout << "Memory cleanup.. \n";
    agent.deregisterMem(dram_for_doca_local, &extra_params);
    agent.deregisterMem(dram_for_doca_remote, &extra_params);
    
    std::cout << "Closing.. \n";

    cudaStreamDestroy(stream);

    /** Argument Parsing */
    // if (argc < 2) {
    //     std::cout <<"Enter the required arguments\n" << std::endl;
    //     std::cout <<"Directory path " << std::endl;
    //     exit(-1);
    // }

        return 0;
}
