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
#include "stream/metadata_stream.h"
#include "serdes/serdes.h"
#define NUM_TRANSFERS 32
#define SIZE 1024
#define INITIATOR_VALUE 0xbb
#define VOLATILE(x) (*(volatile typeof(x) *)&(x))
#define INITIATOR_THRESHOLD_NS 50000 //50us
#define USE_NVTX 1

#if USE_NVTX
#include <nvtx3/nvToolsExt.h>

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define DEVICE_GET_TIME(globaltimer) asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer))

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

static void checkCudaError(cudaError_t result, const char *message) {
	if (result != cudaSuccess) {
		std::cerr << message << " (Error code: " << result << " - "
				   << cudaGetErrorString(result) << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

__global__ void target_kernel(uintptr_t addr, size_t size)
{
    uint8_t ok = 1;

    printf(">>>>>>> CUDA target waiting on addr %p size %d\n", (void*)addr, (uint32_t)size);
    while(VOLATILE(((uint8_t*)addr)[0]) == 0);
    for (int i = 0; i < (int)size; i++) {
        if (((uint8_t*)addr)[i] != INITIATOR_VALUE) {
            printf(">>>>>>> CUDA target byte %x is wrong\n", i);
            ok = 1;
        }
    }
    if (ok == 1)
        printf(">>>>>>> CUDA target, all bytes received!\n");
    else
        printf(">>>>>>> CUDA target, not all received bytes are ok!\n");
}

int launch_target_wait_kernel(cudaStream_t stream, uintptr_t addr, size_t size)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    target_kernel<<<1, 1, 0, stream>>>(addr, size);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

__global__ void initiator_kernel(uintptr_t addr, size_t size)
{
    unsigned long long start, end;

    ((uint8_t*)addr)[threadIdx.x] = INITIATOR_VALUE;

    __syncthreads();

    /* Simulate a longer CUDA kernel to process initiator data */
    DEVICE_GET_TIME(start);
    do {
        DEVICE_GET_TIME(end);
    } while (end - start < INITIATOR_THRESHOLD_NS);
}

int launch_initiator_send_kernel(cudaStream_t stream, uintptr_t addr, size_t size)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    initiator_kernel<<<1, SIZE, 0, stream>>>(addr, size);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

/**
 * This test does p2p from using PUT.
 * intitator -> target so the metadata and
 * desc list needs to move from
 * target to initiator
 */

bool allBytesAre(void* buffer, size_t size, uint8_t value) {
    uint8_t* byte_buffer = static_cast<uint8_t*>(buffer); // Cast void* to uint8_t*
    // Iterate over each byte in the buffer
    for (size_t i = 0; i < size; ++i) {
        if (byte_buffer[i] != value) {
            return false; // Return false if any byte doesn't match the value
        }
    }
    return true; // All bytes match the value
}

std::string recvFromTarget(int port) {
    nixlMDStreamListener listener(port);
    listener.startListenerForClient();
    return listener.recvFromClient();
}

void sendToInitiator(const char *ip, int port, std::string data) {
    nixlMDStreamClient client(ip, port);
    client.connectListener();
    client.sendData(data);
}

int main(int argc, char *argv[]) {
    int                     initiator_port;
    nixl_status_t           ret = NIXL_SUCCESS;
    void                    *addr[NUM_TRANSFERS];
    std::string             role;
    const char              *initiator_ip;
    nixl_blob_t             remote_desc;
    nixl_blob_t             metadata;
    nixl_blob_t             remote_metadata;
    int                     status = 0;

    /** NIXL declarations */
    /** Agent and backend creation parameters */
    nixlAgentConfig cfg(true);
    nixl_b_params_t params;
    nixlBlobDesc    buf[NUM_TRANSFERS];
    nixlBackendH    *doca;
    cudaStream_t    stream;
    /** Serialization/Deserialization object to create a blob */
    nixlSerDes *serdes        = new nixlSerDes();
    nixlSerDes *remote_serdes = new nixlSerDes();
    std::string target_name;

    /** Descriptors and Transfer Request */
    nixl_reg_dlist_t  dram_for_doca(DRAM_SEG);
    nixlXferReqH      *treq;

    /** Argument Parsing */
    if (argc < 4) {
        std::cout <<"Enter the required arguments\n" << std::endl;
        std::cout <<"<Role> " <<"Peer IP> <Peer Port>"
                  << std::endl;
        exit(-1);
    }

    role = std::string(argv[1]);
    initiator_ip   = argv[2];
    initiator_port = std::stoi(argv[3]);
    std::transform(role.begin(), role.end(), role.begin(), ::tolower);

    if (!role.compare("initiator") && !role.compare("target")) {
            std::cerr << "Invalid role. Use 'initiator' or 'target'."
                      << "Currently "<< role <<std::endl;
            return 1;
    }
    /*** End - Argument Parsing */

    checkCudaError(cudaSetDevice(0), "Failed to set device");
	cudaFree(0);
	checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Failed to create CUDA stream");

    /** Common to both Initiator and Target */
    std::cout << "Starting Agent for "<< role << "\n";
    nixlAgent     agent(role, cfg);
    params["network_devices"] = "mlx5_0";
	params["gpu_devices"] = "0";
    PUSH_RANGE("createBackend", 0)
    agent.createBackend("DOCA", params, doca);
    POP_RANGE

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(doca);

    for (int i = 0; i < NUM_TRANSFERS; i++) {
        checkCudaError(cudaMalloc(&addr[i], SIZE), "Failed to allocate CUDA buffer 0");
        checkCudaError(cudaMemset(addr[i], 0, SIZE), "Failed to memset CUDA buffer 0");
        if (role != "target") {
            std::cout << "Allocating for initiator : "
                      << addr[i] << ", "
                      << std::endl;
        } else {
            std::cout << "Allocating for target : "
                      << addr[i] << ", "
                      << std::endl;
        }
        buf[i].addr  = (uintptr_t)(addr[i]);
        buf[i].len   = SIZE;
        buf[i].devId = 0;
        dram_for_doca.addDesc(buf[i]);
    }

    /** Register memory in both initiator and target */
    agent.registerMem(dram_for_doca, &extra_params);
    agent.getLocalMD(metadata);

    std::cout << " Start Control Path metadata exchanges \n";
    if (role == "target") {
        std::cout << " Desc List from Target to Initiator\n";
        dram_for_doca.print();

        /** Sending both metadata strings together */
        assert(serdes->addStr("AgentMD", metadata) == NIXL_SUCCESS);
        assert(dram_for_doca.trim().serialize(serdes) == NIXL_SUCCESS);

        std::cout << " Serialize Metadata to string and Send to Initiator\n";
        std::cout << " \t -- To be handled by runtime - currently sent via a TCP Stream\n";
        sendToInitiator(initiator_ip, initiator_port, serdes->exportStr());
        std::cout << " End Control Path metadata exchanges \n";

        std::cout << " Start Data Path Exchanges \n";
        std::cout << " Waiting to receive Data from Initiator\n";

        /** Sanity Check , assume NUM_TRANSFERS == 1 */
        for (int i = 0; i < NUM_TRANSFERS; i++)
            launch_target_wait_kernel(stream, (uintptr_t)addr[i], SIZE);

        cudaStreamSynchronize(stream);
        std::cout << " DOCA Transfer completed!\n";
    } else {
        std::cout << " Receive metadata from Target \n";
        std::cout << " \t -- To be handled by runtime - currently received via a TCP Stream\n";
        std::string rrstr = recvFromTarget(initiator_port);

        remote_serdes->importStr(rrstr);
        remote_metadata = remote_serdes->getStr("AgentMD");
        assert (remote_metadata != "");
        agent.loadRemoteMD(remote_metadata, target_name);

        std::cout << " Verify Deserialized Target's Desc List at Initiator\n";
        nixl_xfer_dlist_t dram_target_doca(remote_serdes);
        nixl_xfer_dlist_t dram_initiator_doca = dram_for_doca.trim();
        dram_target_doca.print();
        std::cout << " Got metadata from " << target_name << " \n";

        std::cout << " Create transfer request with DOCA backend\n ";
        extra_params.customParam = (uintptr_t)stream;
        PUSH_RANGE("createXferReq", 1)
        ret = agent.createXferReq(NIXL_WRITE, dram_initiator_doca, dram_target_doca,
                                  "target", treq, &extra_params);
        POP_RANGE
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Error creating transfer request\n";
            exit(-1);
        }

        std::cout << "Launch initiator send kernel on stream\n";
        /* Synthetic simulation of GPU processing data before sending */
        PUSH_RANGE("InitKernels", 2)
        for (int i = 0; i < NUM_TRANSFERS; i++)
            launch_initiator_send_kernel(stream, buf[i].addr, buf[i].len);
        POP_RANGE

        std::cout << " Post the request with DOCA backend\n ";
        PUSH_RANGE("postXferReq", 3)
        status = agent.postXferReq(treq);
        POP_RANGE
        std::cout << " Initiator posted Data Path transfer\n";
        std::cout << " Waiting for completion\n";

        PUSH_RANGE("getXferStatus", 4)
        while (status != NIXL_SUCCESS) {
            status = agent.getXferStatus(treq);
            assert(status >= 0);
        }
        POP_RANGE
        std::cout << " Completed Sending " << NUM_TRANSFERS << " transfers using DOCA backend\n";
        agent.releaseXferReq(treq);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::cout <<"Cleanup.. \n";
    agent.deregisterMem(dram_for_doca, &extra_params);
    for (int i = 0; i < NUM_TRANSFERS; i++) {
        cudaFree(addr[i]);
    }
    delete serdes;
    delete remote_serdes;

    return 0;
}
