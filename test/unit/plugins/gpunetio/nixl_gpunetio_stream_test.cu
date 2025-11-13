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
#include "test_utils.h"
#include "stream/metadata_stream.h"
#include "serdes/serdes.h"

#define CUDA_THREADS 512
#define TRANSFER_NUM_BUFFER 32
#define SIZE 1024
#define INITIATOR_VALUE 0xbb
#define VOLATILE(x) (*(volatile typeof (x) *)&(x))
#define INITIATOR_THRESHOLD_NS 50000 // 50us
#define ALIGN_SIZE(size, align) size = ((size + (align) - 1) / (align)) * (align);
#define DEVICE_GET_TIME(globaltimer) asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer))

static void
checkCudaError (cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - " << cudaGetErrorString (result)
                  << ")" << std::endl;
        exit (EXIT_FAILURE);
    }
}

__global__ void
target_kernel (uintptr_t addr, uint8_t val) {
    uint8_t ok = 1;
    uintptr_t buffer_addr = addr + (threadIdx.x * SIZE);

    printf (">>>>>>> CUDA target kernel, waiting on buffer %d addr %lx size %d\n",
            threadIdx.x,
            buffer_addr,
            (uint32_t)SIZE);

    while (VOLATILE (((uint8_t *)buffer_addr)[0]) == 0)
        ;

    for (int i = 0; i < (int)SIZE; i++) {
        if (((uint8_t *)buffer_addr)[i] != val) {
            printf (">>>>>>> CUDA target kernel, buffer %d byte %x is wrong\n", threadIdx.x, i);
            ok = 1;
        }
    }
    if (ok == 1)
        printf (">>>>>>> CUDA target kernel, buffer %d, all bytes received! val=%d\n",
                threadIdx.x,
                val);
    else
        printf (">>>>>>> CUDA target kernel, buffer %d, not all received bytes are ok!\n",
                threadIdx.x);
}

int
launch_target_wait_kernel (cudaStream_t stream, uintptr_t addr, uint8_t val) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
        return -1;
    }

    target_kernel<<<1, TRANSFER_NUM_BUFFER, 0, stream>>> (addr, val);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
        return -1;
    }

    return 0;
}

__global__ void
initiator_kernel (uintptr_t addr, uint8_t val) {
    unsigned long long start, end;
    // Each block updates a buffer in this transfer
    uintptr_t block_address = (addr + (blockIdx.x * SIZE));

    /* Simulate a longer CUDA kernel to process initiator data */
    DEVICE_GET_TIME (start);

    for (int i = threadIdx.x; i < SIZE; i += blockDim.x)
        ((uint8_t *)block_address)[i] = val;

    __syncthreads();

    do {
        DEVICE_GET_TIME (end);
    } while (end - start < INITIATOR_THRESHOLD_NS);
}

int
launch_initiator_send_kernel (cudaStream_t stream, uintptr_t addr, uint8_t val) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
        return -1;
    }

    // Block = # buffers x transfer
    initiator_kernel<<<TRANSFER_NUM_BUFFER, CUDA_THREADS, 0, stream>>> (addr, val);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
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

bool
allBytesAre (void *buffer, size_t size, uint8_t value) {
    uint8_t *byte_buffer = static_cast<uint8_t *> (buffer); // Cast void* to uint8_t*
    // Iterate over each byte in the buffer
    for (size_t i = 0; i < size; ++i) {
        if (byte_buffer[i] != value) {
            return false; // Return false if any byte doesn't match the value
        }
    }
    return true; // All bytes match the value
}

int
main (int argc, char *argv[]) {
    int peer_port;
    nixl_status_t ret = NIXL_SUCCESS;
    uint8_t *data_address;
    std::string role;
    std::string stream_mode;
    const char *peer_ip;
    nixl_blob_t remote_desc;
    nixl_blob_t metadata;
    nixl_blob_t remote_metadata;
    int status = 0;
    static std::string target ("target");
    static std::string initiator ("initiator");

    /** NIXL declarations */
    nixl_b_params_t params;
    nixlBackendH *gpunetio;
    cudaStream_t stream;
    /** Serialization/Deserialization object to create a blob */
    nixlSerDes *serdes = new nixlSerDes();
    nixlSerDes *remote_serdes = new nixlSerDes();

    /** Descriptors and Transfer Request */
    nixl_xfer_dlist_t local_vram (VRAM_SEG);
    nixl_reg_dlist_t local_vram_rdlist (VRAM_SEG);
    nixlBlobDesc temp;
    nixlXferReqH *treq;
    nixl_notifs_t notifs;
    size_t buf_size = SIZE;
    uint32_t buf_num = TRANSFER_NUM_BUFFER;
    uintptr_t data_address_ptr = 0;

    /** Argument Parsing */
    if (argc < 5) {
        std::cout << "Enter the required arguments\n" << std::endl;
        std::cout << "<Role> <Peer IP> <Peer Port> <stream mode: attached or pool>" << std::endl;
        exit (-1);
    }

    role = std::string (argv[1]);
    std::transform (role.begin(), role.end(), role.begin(), ::tolower);
    if (!role.compare (initiator) && !role.compare (target)) {
        std::cerr << "Invalid role. Use 'initiator' or 'target'."
                  << "Currently " << role << std::endl;
        return 1;
    }

    peer_ip = argv[2];
    peer_port = std::stoi (argv[3]);
    stream_mode = std::string (argv[4]);
    std::transform (stream_mode.begin(), stream_mode.end(), stream_mode.begin(), ::tolower);
    // rename stream_mode mode as both sides should have a GPU anyway.
    // stream_attaching or round robin stream

    if (stream_mode.compare ("pool") != 0 && stream_mode.compare ("attached") != 0) {
        std::cerr << "Invalid type of stream_mode. Use 'attached' or 'pool'. "
                  << "Input provided: " << stream_mode << std::endl;
        return 1;
    }

    /*** End - Argument Parsing */
    checkCudaError (cudaSetDevice (0), "Failed to set device");
    cudaFree (0);

    /** Common to both Initiator and Target */
    std::cout << "Starting Agent for " << role << " with stream mode " << stream_mode << "\n";
    /** Agent and backend creation parameters */

    nixlAgentConfig cfg (true, true, peer_port, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);
    nixlAgent agent (role, cfg);

    if (stream_mode.compare ("pool") == 0) params["cuda_streams"] = "2";

    params["network_devices"] = "mlx5_0";
    params["gpu_devices"] = "0";
    ret = agent.createBackend ("GPUNETIO", params, gpunetio);
    // check return

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back (gpunetio);

    static size_t host_page_size = sysconf(_SC_PAGESIZE);
    size_t aligned_size = SIZE * TRANSFER_NUM_BUFFER;
    ALIGN_SIZE(aligned_size, host_page_size);

    // Note: cudaMalloc may not report an host system page aligned address.
    // Note: doca_gpu_mem_alloc is capable to return an aligned GPU memory address.
    checkCudaError (cudaMalloc (&data_address, aligned_size),
                    "Failed to allocate CUDA buffer 0");
    checkCudaError (cudaMemset ((void *)data_address, 0, aligned_size),
                    "Failed to memset CUDA buffer 0");

    if (role != target) {
        std::cout << "Allocating for initiator : " << TRANSFER_NUM_BUFFER << " buffers " << SIZE
                  << " Bytes each " << (void *)data_address << " address " << std::endl;
    } else {
        std::cout << "Allocating for target : " << TRANSFER_NUM_BUFFER << " buffers " << SIZE
                  << " Bytes each " << (void *)data_address << " address " << std::endl;
    }

    for (uint32_t i = 0; i < TRANSFER_NUM_BUFFER; i++) {
        temp.addr = (uintptr_t)(data_address + (i * SIZE));
        temp.len = SIZE;
        temp.devId = std::stoi (params["gpu_devices"]);
        local_vram_rdlist.addDesc (temp);
    }

    /** Register memory in both initiator and target */
    ret = agent.registerMem (local_vram_rdlist, &extra_params);
    nixl_exit_on_failure(ret, "Failed to register memory", role);
    local_vram = local_vram_rdlist.trim();
    ret = agent.getLocalMD(metadata);
    nixl_exit_on_failure(ret, "Failed to get local MD", role);

    std::cout << " Start Control Path metadata exchanges \n";
    if (role == target) {
        bool found = false;
        // Not used
        nixl_xfer_dlist_t descs (DRAM_SEG);
        std::cout << target << " waiting checkRemoteMD from " << initiator << std::endl;
        while (agent.checkRemoteMD (initiator, descs) != NIXL_SUCCESS)
            ;
        std::cout << " Received checkRemoteMD from " << initiator << std::endl;

        data_address_ptr = (uintptr_t)data_address;
        nixl_exit_on_failure(serdes->addBuf("BaseAddress", &data_address_ptr, sizeof(uintptr_t)),
                             "Failed to add BaseAddress",
                             role);
        nixl_exit_on_failure(serdes->addBuf("BufferSize", &buf_size, sizeof(size_t)),
                             "Failed to add BufferSize",
                             role);
        nixl_exit_on_failure(serdes->addBuf("BufferTransfer", &buf_num, sizeof(uint32_t)),
                             "Failed to add BufferTransfer",
                             role);
        nixl_exit_on_failure(serdes->addStr("AgentMD", metadata), "Failed to add AgentMD", role);
        std::string message = serdes->exportStr();
        while (agent.genNotif (initiator, message, &extra_params) != NIXL_SUCCESS)
            ;

        std::cout << " Sending serdes to " << initiator << " to enable notifications" << std::endl;
        std::cout << " data_address_ptr " << data_address_ptr << " data_address "
                  << (uintptr_t)data_address << " buf_size " << buf_size << " buf_num " << buf_num
                  << std::endl;

        // First recv notif: initiator ack it connected correctly
        do {
            nixl_status_t ret = agent.getNotifs (notifs);
        } while (notifs.size() == 0);

        for (const auto &n : notifs) {
            for (size_t idx = 0; idx < n.second.size(); idx++) {
                std::cout << "Received message from " << n.first << " msg: " << n.second[idx]
                          << " at " << idx << std::endl;

                if (n.first == initiator && n.second[idx] == "connected") {
                    std::cout << "Received correct message from " << n.first
                              << " msg: " << n.second[idx] << " at " << idx << std::endl;
                    break;
                }
            }
        }

        std::cout << " Start Data Path Exchanges \n";
        std::cout << " Waiting for first 'sent' notif from " << initiator << std::endl;

        checkCudaError (cudaStreamCreateWithFlags (&stream, cudaStreamNonBlocking),
                        "Failed to create CUDA stream");

        // Second recv notif: initiator ack data has been sent
        do {
            for (const auto &n : notifs) {
                for (size_t idx = 0; idx < n.second.size(); idx++) {
                    if (n.first == initiator && n.second[idx] == "sent") {
                        std::cout << "Received correct message from " << n.first
                                  << " msg: " << n.second[idx] << " at " << idx << std::endl;
                        launch_target_wait_kernel (
                                stream, (uintptr_t)(data_address), INITIATOR_VALUE);
                        cudaStreamSynchronize (stream);
                        std::cout << " GPUNETIO Transfer completed -- first!\n";
                        found = true;
                        break;
                    }
                }
            }
            nixl_status_t ret = agent.getNotifs (notifs);
        } while (found == false);

        // Cleanup all previous notifications
        notifs.clear();

        // First send notif: target processed previously sent data
        std::string msg = "processed";
        ret = agent.genNotif (initiator, msg, &extra_params);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Target genNotif error " << ret << "\n";
        }
        found = false;

        std::cout << " Waiting for second 'sent' notif from " << initiator << std::endl;
        // Third recv notif: sent
        do {
            for (const auto &n : notifs) {
                for (size_t idx = 0; idx < n.second.size(); idx++) {
                    if (n.first == initiator && n.second[idx] == "sent") {
                        std::cout << "Received correct message from " << n.first
                                  << " msg: " << n.second[idx] << " at " << idx << std::endl;
                        launch_target_wait_kernel (
                                stream, (uintptr_t)(data_address), INITIATOR_VALUE + 1);
                        cudaStreamSynchronize (stream);
                        std::cout << " GPUNETIO Transfer completed -- second!\n";
                        found = true;
                        break;
                    }
                }
            }
            nixl_status_t ret = agent.getNotifs (notifs);
        } while (found == false);

        cudaStreamDestroy (stream);
    } else {
        std::cout << "Exchange metadata with " << target << std::endl;
        std::cout << "\t -- To be handled by runtime - currently received via a TCP Stream\n";

        nixl_opt_args_t md_extra_params;
        md_extra_params.ipAddr = peer_ip;
        md_extra_params.port = peer_port;

        ret = agent.fetchRemoteMD (target, &md_extra_params);
        nixl_exit_on_failure(ret, "Failed to fetch remote MD", role);
        ret = agent.sendLocalMD (&md_extra_params);
        nixl_exit_on_failure(ret, "Failed to send local MD", role);
        // Not used
        nixl_xfer_dlist_t descs (DRAM_SEG);
        std::cout << initiator << " waiting checkRemoteMD from " << target << std::endl;
        while (agent.checkRemoteMD (target, descs) != NIXL_SUCCESS)
            ;
        std::cout << initiator << " received checkRemoteMD from " << target << std::endl;

        do {
            nixl_status_t ret = agent.getNotifs (notifs);
        } while (notifs.size() == 0);

        for (const auto &notif : notifs[target]) {
            remote_serdes->importStr (notif);
            nixl_exit_on_failure(
                remote_serdes->getBuf("BaseAddress", &data_address_ptr, sizeof(uintptr_t)),
                "Failed to get BaseAddress",
                role);
            nixl_exit_on_failure(remote_serdes->getBuf("BufferSize", &buf_size, sizeof(size_t)),
                                 "Failed to get BufferSize",
                                 role);
            nixl_exit_on_failure(
                remote_serdes->getBuf("BufferTransfer", &buf_num, sizeof(uint32_t)),
                "Failed to get BufferTransfer",
                role);
            remote_metadata = remote_serdes->getStr ("AgentMD");
            nixl_exit_on_failure((remote_metadata != ""), "Failed to get AgentMD", role);
            agent.loadRemoteMD (remote_metadata, target);
        }
        notifs.clear();

        // First send notif: connected
        std::string msg = "connected";
        std::cout << "Sending " << msg << " to " << target << std::endl;
        ret = agent.genNotif (target, msg);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Target genNotif error " << ret << "\n";
        }

        std::cout << "Verify Deserialized Target's Desc List at Initiator\n";
        nixl_xfer_dlist_t remote_vram_list (VRAM_SEG);

        for (uint32_t i = 0; i < buf_num; i++) {
            temp.addr = (uintptr_t)(data_address_ptr + (i * buf_size));
            temp.len = buf_size;
            temp.devId = std::stoi (params["gpu_devices"]);
            remote_vram_list.addDesc (temp);
        }

        remote_vram_list.print();
        std::cout << "Got metadata from " << target << " \n";
        std::cout << "Create transfer request with GPUNETIO backend\n ";

        // Create Xfer request with notification
        if (stream_mode.compare ("attached") == 0)
            checkCudaError (cudaStreamCreateWithFlags (&stream, cudaStreamNonBlocking),
                            "Failed to create CUDA stream");

        if (stream_mode.compare ("attached") == 0) {
            extra_params.customParam.resize (sizeof (uintptr_t));
            *((uintptr_t *)extra_params.customParam.data()) = (uintptr_t)stream;
        }
        extra_params.notifMsg = "sent";
        extra_params.hasNotif = true;
        ret = agent.createXferReq (
                NIXL_WRITE, local_vram, remote_vram_list, target, treq, &extra_params);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Error creating transfer request\n";
            exit (-1);
        }

        std::cout << "Launch initiator send kernel on stream\n";

        /* Synthetic simulation of GPU data processing with stream attached mode before sending */
        if (stream_mode.compare ("attached") == 0) {
            std::cout << "First xfer, prepare data, GPU mode, transfer 1" << std::endl;
            launch_initiator_send_kernel (stream, (uintptr_t)(data_address), INITIATOR_VALUE);

            std::cout << "Post the request with GPUNETIO backend transfer 1" << std::endl;
            status = agent.postXferReq (treq);
            nixl_exit_on_failure((status < 0), "Failed to post Xfer Req", role);


            std::cout << "Waiting for completion to re-use buffers\n";
            while (status != NIXL_SUCCESS) {
                status = agent.getXferStatus (treq);
                nixl_exit_on_failure(!(status == NIXL_SUCCESS && status == NIXL_IN_PROG),
                                     "Failed to get Xfer Status",
                                     role);
            }
            // No need for cudaStreamSyncronize as CUDA kernel and Xfer are on the same stream
            std::cout << "Second xfer, prepare data, GPU mode, transfer 2" << std::endl;
            launch_initiator_send_kernel (stream, (uintptr_t)(data_address), INITIATOR_VALUE + 1);

            // First recv notif: target processed previously sent data
            std::cout << "Waiting from 'processed' ack from" << target << std::endl;
            do {
                nixl_status_t ret = agent.getNotifs (notifs);
            } while (notifs.size() == 0);

            for (const auto &n : notifs) {
                for (size_t idx = 0; idx < n.second.size(); idx++) {
                    if (n.first == target && n.second[idx] == "processed") {
                        std::cout << "Received correct message from " << n.first
                                  << " msg: " << n.second[idx] << " at " << idx << std::endl;
                        break;
                    }
                }
            }

            // Repost same treq with different data in buffers
            std::cout << "Post the request with GPUNETIO backend transfer 2" << std::endl;
            status = agent.postXferReq (treq);
            nixl_exit_on_failure(status, "Failed to post Xfer Req", role);

            std::cout << "Waiting for completion\n";
            while (status != NIXL_SUCCESS) {
                status = agent.getXferStatus (treq);
                nixl_exit_on_failure(!(status == NIXL_SUCCESS && status == NIXL_IN_PROG),
                                     "Failed to get Xfer Status",
                                     role);
            }
        } else {
            /* Synthetic simulation of CPU data processing with stream pool mode before sending */
            std::cout << "First xfer, prepare data, CPU mode, transfer 1" << std::endl;
            cudaMemset ((void *)data_address, INITIATOR_VALUE, TRANSFER_NUM_BUFFER * SIZE);

            std::cout << "Post the request with GPUNETIO backend transfer 1" << std::endl;
            status = agent.postXferReq (treq);
            nixl_exit_on_failure(status, "Failed to post Xfer Req", role);

            std::cout << "Waiting for completion\n";
            while (status != NIXL_SUCCESS) {
                status = agent.getXferStatus (treq);
                nixl_exit_on_failure(!(status == NIXL_SUCCESS && status == NIXL_IN_PROG),
                                     "Failed to get Xfer Status",
                                     role);
            }

            std::cout << "Second xfer, prepare data, CPU mode, transfer 2" << std::endl;
            cudaMemset ((void *)data_address, INITIATOR_VALUE + 1, TRANSFER_NUM_BUFFER * SIZE);

            // First recv notif: target processed previously sent data
            std::cout << "Waiting from 'processed' ack from" << target << std::endl;
            do {
                nixl_status_t ret = agent.getNotifs (notifs);
            } while (notifs.size() == 0);

            for (const auto &n : notifs) {
                for (size_t idx = 0; idx < n.second.size(); idx++) {
                    if (n.first == target && n.second[idx] == "processed") {
                        std::cout << "Received correct message from " << n.first
                                  << " msg: " << n.second[idx] << " at " << idx << std::endl;
                        break;
                    }
                }
            }

            // Repost same treq with different data in buffers
            std::cout << "Post the request with GPUNETIO backend transfer 2" << std::endl;
            status = agent.postXferReq (treq);
            nixl_exit_on_failure(status, "Failed to post Xfer Req", role);

            std::cout << "Waiting for completion\n";
            while (status != NIXL_SUCCESS) {
                status = agent.getXferStatus (treq);
                nixl_exit_on_failure(!(status == NIXL_SUCCESS && status == NIXL_IN_PROG),
                                     "Failed to get Xfer Status",
                                     role);
            }
        }

        std::cout << "Releasing request " << std::endl;
        agent.releaseXferReq (treq);

        if (stream_mode.compare ("attached") == 0) {
            cudaStreamSynchronize (stream);
            cudaStreamDestroy (stream);
        }
    }

    std::cout << "Cleanup.. \n";

    agent.deregisterMem (local_vram_rdlist, &extra_params);
    // cudaFree(data_address);

    if (role == "target")
        delete serdes;
    else
        delete remote_serdes;

    std::cout << "Exit.. \n";

    return 0;
}
