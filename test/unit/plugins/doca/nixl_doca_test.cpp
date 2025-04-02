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
    void                    *addr[NUM_TRANSFERS];
    nixlAgentConfig         cfg(true);
    nixl_b_params_t         params;
    nixlBlobDesc            buf[NUM_TRANSFERS];
    // nixlBlobDesc            ftrans[NUM_TRANSFERS];
    nixlBackendH            *doca;
    nixlBlobDesc desc;
    // nixlXferReqH            *treq;
    std::string             name = "DOCA";
    nixl_opt_args_t extra_params;
    nixl_reg_dlist_t dram_for_doca(DRAM_SEG);

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
        checkCudaError(cudaMalloc(&addr[i], SIZE), "Failed to allocate CUDA buffer 0");
        checkCudaError(cudaMemset(addr[i], 0, SIZE), "Failed to memset CUDA buffer 0");
        buf[i].addr  = (uintptr_t)(addr[i]);
        buf[i].len   = SIZE;
        buf[i].devId = 0;
        dram_for_doca.addDesc(buf[i]);
        std::cout << "GPU alloc buffer " << i << "\n";
    }

    std::cout << "DOCA registerMem " << "\n";
    agent.registerMem(dram_for_doca, &extra_params);

    std::cout << "Memory cleanup.. \n";
    agent.deregisterMem(dram_for_doca, &extra_params);
    
    std::cout << "Closing.. \n";

    // agent.getLocalMD(tgt_metadata);

        /** Argument Parsing */
        // if (argc < 2) {
        //     std::cout <<"Enter the required arguments\n" << std::endl;
        //     std::cout <<"Directory path " << std::endl;
        //     exit(-1);
        // }
#if 0
        for (i = 0; i < NUM_TRANSFERS; i++) {
        }

        agent.registerMem(file_for_gds);
        agent.registerMem(vram_for_gds);

        nixl_xfer_dlist_t vram_for_gds_list = vram_for_gds.trim();
        nixl_xfer_dlist_t file_for_gds_list = file_for_gds.trim();

        ret = agent.createXferReq(NIXL_WRITE, vram_for_gds_list, file_for_gds_list,
                                  "GDSTester", treq);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Error creating transfer request\n" << ret;
            exit(-1);
        }

        std::cout << " Post the request with GDS backend\n ";
        status = agent.postXferReq(treq);
        std::cout << " GDS File IO has been posted\n";
        std::cout << " Waiting for completion\n";

        while (status != NIXL_SUCCESS) {
            status = agent.getXferStatus(treq);
            assert(status >= 0);
        }
        std::cout <<" Completed writing data using GDS backend\n";
        agent.releaseXferReq(treq);

        std::cout <<"Cleanup.. \n";
        agent.deregisterMem(vram_for_gds);
        agent.deregisterMem(file_for_gds);
#endif
        return 0;
}
