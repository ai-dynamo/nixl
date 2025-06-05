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

#include "doca_backend.h"
#include "serdes/serdes.h"
#include <arpa/inet.h>
#include <cassert>
#include <stdexcept>
#include <unistd.h>
#include "common/nixl_log.h"

/****************************************
 * Constructor/Destructor
 *****************************************/

nixlDocaEngine::nixlDocaEngine (const nixlBackendInitParams *init_params) :
        nixlBackendEngine (init_params) {
    std::vector<std::string> ndevs, tmp_gdevs; /* Empty vector */
    doca_error_t result;
    nixl_b_params_t *custom_params = init_params->customParams;

    result = doca_log_backend_create_standard();
    if (result != DOCA_SUCCESS) throw std::invalid_argument ("Can't initialize doca log");

    result = doca_log_backend_create_with_file_sdk (stderr, &sdk_log);
    if (result != DOCA_SUCCESS) throw std::invalid_argument ("Can't initialize doca log");

    result = doca_log_backend_set_sdk_level (sdk_log, DOCA_LOG_LEVEL_ERROR);
    if (result != DOCA_SUCCESS) throw std::invalid_argument ("Can't initialize doca log");

    std::cout << "DOCA network devices ";
    // Temporary: will extend to more GPUs in a dedicated PR
    if (custom_params->count("network_devices") > 1)
        throw std::invalid_argument ("Only 1 network device is allowed");

    if (custom_params->count("network_devices") == 0 || (*custom_params)["network_devices"] == "" || (*custom_params)["network_devices"] == "all") {
        ndevs.push_back("mlx5_0");
        std::cout << "Using default network device mlx5_0";
    } else {
        ndevs = str_split((*custom_params)["network_devices"], " ");
        std::cout << ndevs[0];
    }
    std::cout << std::endl;

    std::cout << "DOCA GPU devices: ";
    // Temporary: will extend to more GPUs in a dedicated PR
    if (custom_params->count("gpu_devices") > 1)
        throw std::invalid_argument ("Only 1 GPU device is allowed");

    if (custom_params->count("gpu_devices") == 0 || (*custom_params)["gpu_devices"] == "" || (*custom_params)["gpu_devices"] == "all") {
        gdevs.push_back (std::pair ((uint32_t)0, nullptr));
        std::cout << "Using default CUDA device ID 0";
    } else {
        tmp_gdevs = str_split ((*custom_params)["gpu_devices"], " ");
        for (auto &cuda_id : tmp_gdevs) {
            gdevs.push_back (std::pair ((uint32_t)std::stoi (cuda_id), nullptr));
            std::cout << "cuda_id " << cuda_id;
        }
    }
    std::cout << std::endl;

    nstreams = 0;
    if (custom_params->count ("cuda_streams") != 0 && (*custom_params)["cuda_streams"] != "")
        nstreams = std::stoi ((*custom_params)["cuda_streams"]);
    if (nstreams == 0) nstreams = DOCA_POST_STREAM_NUM;

    std::cout << "CUDA streams used for pool mode: " << nstreams << std::endl;

    /* Open DOCA device */
    result = open_doca_device_with_ibdev_name (
            (const uint8_t *)(ndevs[0].c_str()), ndevs[0].size(), &(ddev));
    if (result != DOCA_SUCCESS) {
        throw std::invalid_argument ("Failed to open DOCA device");
    }

    char pciBusId[DOCA_DEVINFO_IBDEV_NAME_SIZE];
    for (auto &item : gdevs) {
        cudaDeviceGetPCIBusId (pciBusId, DOCA_DEVINFO_IBDEV_NAME_SIZE, item.first);
        result = doca_gpu_create (pciBusId, &item.second);
        if (result != DOCA_SUCCESS)
            NIXL_ERROR << "Failed to create DOCA GPU device " << doca_error_get_descr (result);
    }

    doca_devinfo_get_ipv4_addr (
            doca_dev_as_devinfo (ddev), (uint8_t *)ipv4_addr, DOCA_DEVINFO_IPV4_ADDR_SIZE);

    // DOCA_GPU_MEM_TYPE_GPU_CPU == GDRCopy
    result = doca_gpu_mem_alloc (gdevs[0].second,
                                 sizeof (struct docaXferReqGpu) * DOCA_XFER_REQ_MAX,
                                 4096,
                                 DOCA_GPU_MEM_TYPE_GPU_CPU,
                                 (void **)&xferReqRingGpu,
                                 (void **)&xferReqRingCpu);
    if (result != DOCA_SUCCESS || xferReqRingGpu == NULL || xferReqRingCpu == NULL) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc with DOCA_GPU_MEM_TYPE_GPU_CPU returned "
                   << doca_error_get_descr (result);
        NIXL_ERROR << "Allocating memory with DOCA_GPU_MEM_TYPE_CPU_GPU";
        result = doca_gpu_mem_alloc (gdevs[0].second,
                                     sizeof (struct docaXferReqGpu) * DOCA_XFER_REQ_MAX,
                                     4096,
                                     DOCA_GPU_MEM_TYPE_CPU_GPU,
                                     (void **)&xferReqRingGpu,
                                     (void **)&xferReqRingCpu);
        if (result != DOCA_SUCCESS || xferReqRingGpu == NULL || xferReqRingCpu == NULL) {
            NIXL_ERROR << "Function doca_gpu_mem_alloc with DOCA_GPU_MEM_TYPE_CPU_GPU returned "
                       << doca_error_get_descr (result);
            throw std::invalid_argument ("Can't allocate memory");
        }
    }

    nixlDocaEngineCheckCudaError (
            cudaMemset (xferReqRingGpu, 0, sizeof (struct docaXferReqGpu) * DOCA_XFER_REQ_MAX),
            "Failed to memset GPU memory");

    result = doca_gpu_mem_alloc (gdevs[0].second,
                                 sizeof (uint64_t) * 1,
                                 4096,
                                 DOCA_GPU_MEM_TYPE_GPU,
                                 (void **)&last_rsvd_flags,
                                 nullptr);
    if (result != DOCA_SUCCESS || last_rsvd_flags == NULL || last_rsvd_flags == NULL) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr (result);
    }

    nixlDocaEngineCheckCudaError (cudaMemset (last_rsvd_flags, 0, sizeof (uint64_t) * 1),
                                  "Failed to memset GPU memory");

    result = doca_gpu_mem_alloc (gdevs[0].second,
                                 sizeof (uint64_t) * 1,
                                 4096,
                                 DOCA_GPU_MEM_TYPE_GPU,
                                 (void **)&last_posted_flags,
                                 nullptr);
    if (result != DOCA_SUCCESS || last_posted_flags == NULL || last_posted_flags == NULL) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr (result);
    }

    nixlDocaEngineCheckCudaError (cudaMemset (last_posted_flags, 0, sizeof (uint64_t) * 1),
                                  "Failed to memset GPU memory");


    nixlDocaEngineCheckCudaError (cudaStreamCreateWithFlags (&wait_stream, cudaStreamNonBlocking),
                                  "Failed to create CUDA stream");
    for (int i = 0; i < nstreams; i++)
        nixlDocaEngineCheckCudaError (
                cudaStreamCreateWithFlags (&post_stream[i], cudaStreamNonBlocking),
                "Failed to create CUDA stream");
    xferStream = 0;

    result = doca_gpu_mem_alloc (gdevs[0].second,
                                 sizeof (struct docaXferCompletion) * DOCA_MAX_COMPLETION_INFLIGHT,
                                 4096,
                                 DOCA_GPU_MEM_TYPE_CPU_GPU,
                                 (void **)&completion_list_gpu,
                                 (void **)&completion_list_cpu);
    if (result != DOCA_SUCCESS || completion_list_gpu == NULL || completion_list_cpu == NULL) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr (result);
    }

    memset (completion_list_cpu,
            0,
            sizeof (struct docaXferCompletion) * DOCA_MAX_COMPLETION_INFLIGHT);

    // DOCA_GPU_MEM_TYPE_GPU_CPU == GDRCopy
    result = doca_gpu_mem_alloc (gdevs[0].second,
                                 sizeof (uint32_t),
                                 4096,
                                 DOCA_GPU_MEM_TYPE_GPU_CPU,
                                 (void **)&wait_exit_gpu,
                                 (void **)&wait_exit_cpu);
    if (result != DOCA_SUCCESS || wait_exit_gpu == NULL || wait_exit_cpu == NULL) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc with DOCA_GPU_MEM_TYPE_GPU_CPU returned "
                   << doca_error_get_descr (result);
        NIXL_ERROR << "Allocating memory with DOCA_GPU_MEM_TYPE_CPU_GPU";
        result = doca_gpu_mem_alloc (gdevs[0].second,
                                     sizeof (uint32_t),
                                     4096,
                                     DOCA_GPU_MEM_TYPE_CPU_GPU,
                                     (void **)&wait_exit_gpu,
                                     (void **)&wait_exit_cpu);
        if (result != DOCA_SUCCESS || wait_exit_gpu == NULL || wait_exit_cpu == NULL) {
            NIXL_ERROR << "Function doca_gpu_mem_alloc with DOCA_GPU_MEM_TYPE_CPU_GPU returned "
                       << doca_error_get_descr (result);
            throw std::invalid_argument ("Can't allocate memory");
        }
    }

    ((volatile uint8_t *)wait_exit_cpu)[0] = 0;

    result = doca_gpu_mem_alloc (gdevs[0].second,
                                 sizeof (struct docaNotifRecv) * 1,
                                 4096,
                                 DOCA_GPU_MEM_TYPE_CPU_GPU,
                                 (void **)&notif_fill_gpu,
                                 (void **)&notif_fill_cpu);
    if (result != DOCA_SUCCESS || notif_fill_gpu == NULL || notif_fill_cpu == NULL) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr (result);
    }

    result = doca_gpu_mem_alloc (gdevs[0].second,
                                 sizeof (struct docaNotifRecv) * 1,
                                 4096,
                                 DOCA_GPU_MEM_TYPE_CPU_GPU,
                                 (void **)&notif_progress_gpu,
                                 (void **)&notif_progress_cpu);
    if (result != DOCA_SUCCESS || notif_progress_gpu == NULL || notif_progress_cpu == NULL) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr (result);
    }

    memset (notif_progress_cpu, 0, sizeof (struct docaNotifRecv) * 1);

    result = doca_gpu_mem_alloc (gdevs[0].second,
                                 sizeof (struct docaNotifSend) * 1,
                                 4096,
                                 DOCA_GPU_MEM_TYPE_CPU_GPU,
                                 (void **)&notif_send_gpu,
                                 (void **)&notif_send_cpu);
    if (result != DOCA_SUCCESS || notif_send_gpu == NULL || notif_send_cpu == NULL) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr (result);
    }

    memset (notif_send_cpu, 0, sizeof (struct docaNotifSend) * 1);

    cuCtxGetCurrent (&main_cuda_ctx);
    // Warmup
    doca_kernel_progress (wait_stream,
                          nullptr,
                          notif_fill_gpu,
                          notif_progress_gpu,
                          notif_send_gpu,
                          wait_exit_gpu);
    cudaStreamSynchronize (wait_stream);
    doca_kernel_progress (wait_stream,
                          completion_list_gpu,
                          notif_fill_gpu,
                          notif_progress_gpu,
                          notif_send_gpu,
                          wait_exit_gpu);

    // We may need a GPU warmup with relevant DOCA engine kernels
    doca_kernel_write (0, nullptr, nullptr, 0);
    doca_kernel_read (0, nullptr, nullptr, 0);

    lastPostedReq = 0;
    xferRingPos = 0;

    progressThreadStart();
}

nixl_mem_list_t
nixlDocaEngine::getSupportedMems() const {
    nixl_mem_list_t mems;
    mems.push_back (DRAM_SEG);
    mems.push_back (VRAM_SEG);
    return mems;
}

nixlDocaEngine::~nixlDocaEngine() {
    doca_error_t result;

    // per registered memory deregisters it, which removes the corresponding
    // metadata too parent destructor takes care of the desc list For remote
    // metadata, they should be removed here
    if (this->initErr) {
        // Nothing to do
        return;
    }

    // Cause accept in thread to fail and thus exit
    oob_connection_server_close (oob_sock_server);
    oob_connection_client_close (oob_sock_client);
    progressThreadStop();

    ((volatile uint8_t *)wait_exit_cpu)[0] = 1;
    cudaStreamSynchronize (wait_stream);
    cudaStreamDestroy (wait_stream);
    doca_gpu_mem_free (gdevs[0].second, wait_exit_gpu);
    doca_gpu_mem_free (gdevs[0].second, xferReqRingGpu);
    doca_gpu_mem_free (gdevs[0].second, last_rsvd_flags);
    doca_gpu_mem_free (gdevs[0].second, last_posted_flags);

    for (int i = 0; i < nstreams; i++) {
        cudaStreamSynchronize (post_stream[i]);
        cudaStreamDestroy (post_stream[i]);
    }

    for (auto notif : notifMap)
        nixlDocaDestroyNotif (gdevs[0].second, notif.second);

    doca_gpu_mem_free (gdevs[0].second, notif_fill_gpu);
    doca_gpu_mem_free (gdevs[0].second, notif_progress_gpu);
    doca_gpu_mem_free (gdevs[0].second, notif_send_gpu);
    doca_gpu_mem_free (gdevs[0].second, completion_list_gpu);

    for (const auto &rdma_qp : qpMap) {
        result = doca_ctx_stop (rdma_qp.second->rdma_ctx_data);
        if (result != DOCA_SUCCESS)
            NIXL_ERROR << "Failed to stop RDMA context " << doca_error_get_descr (result);

        result = doca_rdma_destroy (rdma_qp.second->rdma_data);
        if (result != DOCA_SUCCESS)
            NIXL_ERROR << "Failed to destroy DOCA RDMA " << doca_error_get_descr (result);

        result = doca_ctx_stop (rdma_qp.second->rdma_ctx_notif);
        if (result != DOCA_SUCCESS)
            NIXL_ERROR << "Failed to stop RDMA context " << doca_error_get_descr (result);

        result = doca_rdma_destroy (rdma_qp.second->rdma_notif);
        if (result != DOCA_SUCCESS)
            NIXL_ERROR << "Failed to destroy DOCA RDMA " << doca_error_get_descr (result);
    }

    result = doca_dev_close (ddev);
    if (result != DOCA_SUCCESS)
        NIXL_ERROR << "Failed to close DOCA device " << doca_error_get_descr (result);

    result = doca_gpu_destroy (gdevs[0].second);
    if (result != DOCA_SUCCESS)
        NIXL_ERROR << "Failed to close DOCA GPU device " << doca_error_get_descr (result);
}

/****************************************
 * DOCA request management
 *****************************************/

nixl_status_t
nixlDocaEngine::nixlDocaInitNotif (const std::string &remote_agent,
                                   struct doca_dev *dev,
                                   struct doca_gpu *gpu) {
    doca_error_t result;
    struct nixlDocaNotif *notif;
    bool exist = false;

    // Same peer can be server or client
    if (notifMap.find (remote_agent) != notifMap.end()) {
        return NIXL_SUCCESS;
    }

    notif = new struct nixlDocaNotif;

    notif->elems_num = DOCA_MAX_NOTIF_INFLIGHT;
    notif->elems_size = DOCA_MAX_NOTIF_MESSAGE_SIZE;
    notif->send_addr = (uint8_t *)calloc (notif->elems_size * notif->elems_num, sizeof (uint8_t));
    if (notif->send_addr == nullptr) {
        NIXL_ERROR << "Can't alloc memory for send notif";
        return NIXL_ERR_BACKEND;
    }
    memset (notif->send_addr, 0, notif->elems_size * notif->elems_num);

    result = create_doca_mmap (
            notif->send_addr, notif->elems_num, notif->elems_size, ddev, &notif->send_mmap);
    if (result != DOCA_SUCCESS) goto error;

    result = create_doca_buf_arr (notif->send_mmap,
                                  notif->elems_num,
                                  (size_t)notif->elems_size,
                                  gpu,
                                  &(notif->send_barr),
                                  &(notif->send_barr_gpu));
    if (result != DOCA_SUCCESS) goto error;

    notif->recv_addr = (uint8_t *)calloc (notif->elems_size * notif->elems_num, sizeof (uint8_t));
    if (notif->recv_addr == nullptr) {
        NIXL_ERROR << "Can't alloc memory for send notif";
        return NIXL_ERR_BACKEND;
    }
    memset (notif->recv_addr, 0, notif->elems_size * notif->elems_num);

    result = create_doca_mmap (
            notif->recv_addr, notif->elems_num, notif->elems_size, ddev, &notif->recv_mmap);
    if (result != DOCA_SUCCESS) goto error;

    result = create_doca_buf_arr (notif->recv_mmap,
                                  notif->elems_num,
                                  (size_t)notif->elems_size,
                                  gpu,
                                  &(notif->recv_barr),
                                  &(notif->recv_barr_gpu));
    if (result != DOCA_SUCCESS) goto error;

    notif->send_pi = 0;
    notif->recv_pi = 0;

    // Ensure notif list is not added twice for the same peer
    notifLock.lock();
    if (notifMap.find (remote_agent) != notifMap.end()) {
        exist = true;
        notifLock.unlock();
    } else {
        notifMap[remote_agent] = notif;
        ((volatile struct docaNotifRecv *)notif_fill_cpu)->barr_gpu = notif->recv_barr_gpu;
        std::atomic_thread_fence (std::memory_order_release);
        ((volatile struct docaNotifRecv *)notif_fill_cpu)->rdma_qp =
                qpMap[remote_agent]->rdma_gpu_notif;
        while (((volatile struct docaNotifRecv *)notif_fill_cpu)->rdma_qp != nullptr)
            ;
        notifLock.unlock();
    }

    if (exist == true) goto error;

    return NIXL_SUCCESS;

error:
    destroy_doca_buf_arr (notif->send_barr);
    destroy_doca_buf_arr (notif->recv_barr);

    destroy_doca_mmap (notif->send_mmap);
    destroy_doca_mmap (notif->recv_mmap);

    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlDocaEngine::nixlDocaDestroyNotif (struct doca_gpu *gpu, struct nixlDocaNotif *notif) {

    if (notif->send_barr) doca_buf_arr_destroy (notif->send_barr);
    if (notif->recv_barr) doca_buf_arr_destroy (notif->recv_barr);

    if (notif->send_mmap) doca_mmap_destroy (notif->send_mmap);
    if (notif->recv_mmap) doca_mmap_destroy (notif->recv_mmap);

    return NIXL_SUCCESS;
}

void
nixlDocaEngine::_requestInit (void *request) {
    /* Initialize request in-place (aka "placement new")*/
    new (request) nixlDocaBckndReq;
}

void
nixlDocaEngine::_requestFini (void *request) {
    /* Finalize request */
    nixlDocaBckndReq *req = (nixlDocaBckndReq *)request;
    req->~nixlDocaBckndReq();
}

// For now just connection setup, not used for xfers to be a complete progThread, so supportsProgTh
// is false
nixl_status_t
nixlDocaEngine::progressThreadStart() {
    struct sockaddr_in server_addr = {0};
    int enable = 1;
    int result;
    pthrStop = pthrActive = 0;
    noSyncIters = 32;

    /* Create socket */

    oob_sock_server = socket (AF_INET, SOCK_STREAM, 0);
    if (oob_sock_server < 0) {
        NIXL_ERROR << "Error while creating socket " << oob_sock_server;
        return NIXL_ERR_NOT_SUPPORTED;
    }
    NIXL_INFO << "DOCA Server socket created successfully";

    if (setsockopt (oob_sock_server, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof (enable))) {
        NIXL_ERROR << "Error setting socket options";
        close (oob_sock_server);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (setsockopt (oob_sock_server, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof (enable))) {
        NIXL_ERROR << "Error setting socket options";
        close (oob_sock_server);
        return NIXL_ERR_NOT_SUPPORTED;
    }
    /* Set port and IP: */
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons (DOCA_RDMA_CM_LOCAL_PORT_SERVER);
    server_addr.sin_addr.s_addr = INADDR_ANY; /* listen on any interface */

    /* Bind to the set port and IP: */
    if (bind (oob_sock_server, (struct sockaddr *)&server_addr, sizeof (server_addr)) < 0) {
        NIXL_ERROR << "Couldn't bind to the port " << DOCA_RDMA_CM_LOCAL_PORT_SERVER;
        close (oob_sock_server);
        return NIXL_ERR_NOT_SUPPORTED;
    }
    NIXL_DEBUG << "Done with binding";

    /* Listen for clients: */
    if (listen (oob_sock_server, 1) < 0) {
        NIXL_ERROR << "Error while listening";
        close (oob_sock_server);
        return NIXL_ERR_NOT_SUPPORTED;
    }
    NIXL_INFO << "Listening for incoming connections";

    // Start the thread
    // TODO [Relaxed mem] mem barrier to ensure pthr_x updates are complete
    // new (&pthr) std::thread(&nixlDocaEngine::threadProgressFunc, this);

    result = pthread_create (&server_thread_id, NULL, threadProgressFunc, (void *)this);
    if (result != 0) {
        NIXL_ERROR << "Failed to create threadProgressFunc thread";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

void
nixlDocaEngine::progressThreadStop() {
    ACCESS_ONCE (pthrStop) = 1;
    // pthr.join();
    pthread_join (server_thread_id, NULL);
}

uint32_t
nixlDocaEngine::getGpuCudaId() {
    return gdevs[0].first;
}

nixl_status_t
nixlDocaEngine::addRdmaQp (const std::string &remote_agent) {
    doca_error_t result;
    struct nixlDocaRdmaQp *rdma_qp;

    // Same peer can be server or client
    if (qpMap.find (remote_agent) != qpMap.end()) {
        return NIXL_IN_PROG;
    }

    NIXL_DEBUG << "DOCA addRdmaQp for remote " << remote_agent;

    rdma_qp = new struct nixlDocaRdmaQp;

    /* DATA QP */

    /* Create DOCA RDMA instance */
    result = doca_rdma_create (ddev, &(rdma_qp->rdma_data));
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create DOCA RDMA " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Convert DOCA RDMA to general DOCA context */
    rdma_qp->rdma_ctx_data = doca_rdma_as_ctx (rdma_qp->rdma_data);
    if (rdma_qp->rdma_ctx_data == NULL) {
        result = DOCA_ERROR_UNEXPECTED;
        NIXL_ERROR << "Failed to convert DOCA RDMA to DOCA context "
                   << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Set permissions to DOCA RDMA */
    result = doca_rdma_set_permissions (
            rdma_qp->rdma_data, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set permissions to DOCA RDMA " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    // /* Set gid_index to DOCA RDMA if it's provided */
    // if (cfg->is_gid_index_set) {
    // 	/* Set gid_index to DOCA RDMA */
    // 	result = doca_rdma_set_gid_index(rdma, cfg->gid_index);
    // 	if (result != DOCA_SUCCESS) {
    // 		NIXL_ERROR << "Failed to set gid_index to DOCA RDMA " << // doca_error_get_descr(result);
    // return NIXL_ERR_BACKEND;
    // 	}
    // }

    /* Set send queue size to DOCA RDMA */
    result = doca_rdma_set_send_queue_size (rdma_qp->rdma_data, RDMA_SEND_QUEUE_SIZE);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set send queue size to DOCA RDMA "
                   << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Setup datapath of RDMA CTX on GPU */
    result = doca_ctx_set_datapath_on_gpu (rdma_qp->rdma_ctx_data, gdevs[0].second);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set datapath on GPU " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Set receive queue size to DOCA RDMA */
    result = doca_rdma_set_recv_queue_size (rdma_qp->rdma_data, RDMA_RECV_QUEUE_SIZE);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set receive queue size to DOCA RDMA "
                   << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Set GRH to DOCA RDMA */
    result = doca_rdma_set_grh_enabled (rdma_qp->rdma_data, true);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set GRH to DOCA RDMA " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    result = doca_ctx_start (rdma_qp->rdma_ctx_data);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to start RDMA context data " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    result = doca_rdma_get_gpu_handle (rdma_qp->rdma_data, &(rdma_qp->rdma_gpu_data));
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to get RDMA GPU handler " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    result = doca_rdma_export (rdma_qp->rdma_data,
                               &(rdma_qp->connection_details_data),
                               &(rdma_qp->conn_det_len_data),
                               &rdma_qp->connection_data);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to export RDMA handler " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* NOTIF QP */

    /* Create DOCA RDMA instance */
    result = doca_rdma_create (ddev, &(rdma_qp->rdma_notif));
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create DOCA RDMA " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Convert DOCA RDMA to general DOCA context */
    rdma_qp->rdma_ctx_notif = doca_rdma_as_ctx (rdma_qp->rdma_notif);
    if (rdma_qp->rdma_ctx_notif == NULL) {
        result = DOCA_ERROR_UNEXPECTED;
        NIXL_ERROR << "Failed to convert DOCA RDMA to DOCA context "
                   << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Set permissions to DOCA RDMA */
    result = doca_rdma_set_permissions (
            rdma_qp->rdma_notif, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set permissions to DOCA RDMA " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    // /* Set gid_index to DOCA RDMA if it's provided */
    // if (cfg->is_gid_index_set) {
    // 	/* Set gid_index to DOCA RDMA */
    // 	result = doca_rdma_set_gid_index(rdma, cfg->gid_index);
    // 	if (result != DOCA_SUCCESS) {
    // 		NIXL_ERROR << "Failed to set gid_index to DOCA RDMA " << // doca_error_get_descr(result);
    // return NIXL_ERR_BACKEND;
    // 	}
    // }

    /* Set send queue size to DOCA RDMA */
    result = doca_rdma_set_send_queue_size (rdma_qp->rdma_notif, RDMA_SEND_QUEUE_SIZE);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set send queue size to DOCA RDMA "
                   << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Setup notifpath of RDMA CTX on GPU */
    result = doca_ctx_set_datapath_on_gpu (rdma_qp->rdma_ctx_notif, gdevs[0].second);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set notifpath on GPU " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Set receive queue size to DOCA RDMA */
    result = doca_rdma_set_recv_queue_size (rdma_qp->rdma_notif, RDMA_RECV_QUEUE_SIZE);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set receive queue size to DOCA RDMA "
                   << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Set GRH to DOCA RDMA */
    result = doca_rdma_set_grh_enabled (rdma_qp->rdma_notif, true);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set GRH to DOCA RDMA " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    result = doca_ctx_start (rdma_qp->rdma_ctx_notif);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to start RDMA context notif " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    result = doca_rdma_get_gpu_handle (rdma_qp->rdma_notif, &(rdma_qp->rdma_gpu_notif));
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to get RDMA GPU handler " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    result = doca_rdma_export (rdma_qp->rdma_notif,
                               &(rdma_qp->connection_details_notif),
                               &(rdma_qp->conn_det_len_notif),
                               &rdma_qp->connection_notif);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to export RDMA handler " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    qpMap[remote_agent] = rdma_qp;

    NIXL_DEBUG << "DOCA addRdmaQp new QP added for " << remote_agent;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::connectClientRdmaQp (int oob_sock_client, const std::string &remote_agent) {
    doca_error_t result;
    void *remote_conn_details_data = NULL;
    void *remote_conn_details_notif = NULL;
    size_t remote_conn_details_len_data = 0;
    size_t remote_conn_details_len_notif = 0;
    struct nixlDocaRdmaQp *rdma_qp = qpMap[remote_agent]; // validate

    // Data QP
    if (send (oob_sock_client, &rdma_qp->conn_det_len_data, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (send (oob_sock_client, rdma_qp->connection_details_data, rdma_qp->conn_det_len_data, 0) <
        0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    if (send (oob_sock_client, &rdma_qp->conn_det_len_notif, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (send (oob_sock_client, rdma_qp->connection_details_notif, rdma_qp->conn_det_len_notif, 0) <
        0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    NIXL_ERROR << "Receive client remote data qp connection details";
    if (recv (oob_sock_client, &remote_conn_details_len_data, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (remote_conn_details_len_data <= 0 || remote_conn_details_len_data >= (size_t)-1) {
        NIXL_ERROR << "Received wrong remote connection details, client, data "
                   << remote_conn_details_len_data;
        result = DOCA_ERROR_NO_MEMORY;
        return NIXL_ERR_BACKEND;
    }

    remote_conn_details_data = calloc (1, remote_conn_details_len_data);
    if (remote_conn_details_data == NULL) {
        NIXL_ERROR << "Failed to allocate memory for remote connection details";
        result = DOCA_ERROR_NO_MEMORY;
        return NIXL_ERR_BACKEND;
    }

    if (recv (oob_sock_client, remote_conn_details_data, remote_conn_details_len_data, 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << "Receive remote notif qp connection details";
    if (recv (oob_sock_client, &remote_conn_details_len_notif, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (remote_conn_details_len_notif <= 0 || remote_conn_details_len_notif >= (size_t)-1) {
        NIXL_ERROR << "Received wrong remote connection details, client, notif";
        result = DOCA_ERROR_NO_MEMORY;
        return NIXL_ERR_BACKEND;
    }

    remote_conn_details_notif = calloc (1, remote_conn_details_len_notif);
    if (remote_conn_details_notif == NULL) {
        NIXL_ERROR << "Failed to allocate memory for remote connection details";
        result = DOCA_ERROR_NO_MEMORY;
        return NIXL_ERR_BACKEND;
    }

    if (recv (oob_sock_client, remote_conn_details_notif, remote_conn_details_len_notif, 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_INFO << "Connect DOCA RDMA to remote RDMA -- data";
    result = doca_rdma_connect (rdma_qp->rdma_data,
                                remote_conn_details_data,
                                remote_conn_details_len_data,
                                rdma_qp->connection_data);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function doca_rdma_connect failed " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    free (remote_conn_details_data);
    remote_conn_details_data = NULL;

    /* Connect local rdma to the remote rdma */
    NIXL_INFO << "Connect DOCA RDMA to remote RDMA -- notif";
    result = doca_rdma_connect (rdma_qp->rdma_notif,
                                remote_conn_details_notif,
                                remote_conn_details_len_notif,
                                rdma_qp->connection_notif);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function doca_rdma_connect failed " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    free (remote_conn_details_notif);
    remote_conn_details_notif = NULL;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::recvLocalAgent (int oob_sock_client, std::string &remote_agent) {
    size_t msg_size;
    char *remote_agent_;

    // Msg
    if (recv (oob_sock_client, &msg_size, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to recv msg details";
        close (oob_sock_client);
        return NIXL_ERR_BACKEND;
    }

    remote_agent_ = (char *)calloc (msg_size, sizeof (char));
    if (remote_agent_ == nullptr) {
        NIXL_ERROR << "Failed to alloc msg memory";
        close (oob_sock_client);
        return NIXL_ERR_BACKEND;
    }

    if (recv (oob_sock_client, remote_agent_, msg_size, 0) < 0) {
        NIXL_ERROR << "Failed to recv msg details";
        close (oob_sock_client);
        return NIXL_ERR_BACKEND;
    }

    remote_agent = remote_agent_;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::sendLocalAgent (int oob_sock_client, const std::string &remote_agent) {
    size_t agent_size = localAgent.size();

    if (send (oob_sock_client, &agent_size, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        return NIXL_ERR_BACKEND;
    }

    if (send (oob_sock_client, localAgent.c_str(), localAgent.size(), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::connectServerRdmaQp (int oob_sock_client, const std::string &remote_agent) {
    doca_error_t result;
    void *remote_conn_details_data = NULL;
    size_t remote_conn_details_data_len = 0;
    void *remote_conn_details_notif = NULL;
    size_t remote_conn_details_notif_len = 0;

    struct nixlDocaRdmaQp *rdma_qp = qpMap[remote_agent]; // validate

    NIXL_DEBUG << "DOCA connectServerRdmaQp for agent %s", remote_agent.c_str();

    if (recv (oob_sock_client, &remote_conn_details_data_len, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (remote_conn_details_data_len <= 0 || remote_conn_details_data_len >= (size_t)-1) {
        NIXL_ERROR << "Received wrong remote connection details, server, data";
        result = DOCA_ERROR_NO_MEMORY;
        return NIXL_ERR_BACKEND;
    }

    remote_conn_details_data = calloc (1, remote_conn_details_data_len);
    if (remote_conn_details_data == NULL) {
        NIXL_ERROR << "Failed to allocate memory for remote data details"
                   << remote_conn_details_data_len;
        result = DOCA_ERROR_NO_MEMORY;
        return NIXL_ERR_BACKEND;
    }

    if (recv (oob_sock_client, remote_conn_details_data, remote_conn_details_data_len, 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (recv (oob_sock_client, &remote_conn_details_notif_len, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (remote_conn_details_notif_len <= 0 || remote_conn_details_notif_len >= (size_t)-1) {
        NIXL_ERROR << "Received wrong remote connection details, server, notif";
        result = DOCA_ERROR_NO_MEMORY;
        return NIXL_ERR_BACKEND;
    }

    remote_conn_details_notif = calloc (1, remote_conn_details_notif_len);
    if (remote_conn_details_notif == NULL) {
        NIXL_ERROR << "Failed to allocate memory for remote notif details "
                   << remote_conn_details_notif_len;
        result = DOCA_ERROR_NO_MEMORY;
        return NIXL_ERR_BACKEND;
    }

    if (recv (oob_sock_client, remote_conn_details_notif, remote_conn_details_notif_len, 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Data QP
    if (send (oob_sock_client, &rdma_qp->conn_det_len_data, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (send (oob_sock_client, rdma_qp->connection_details_data, rdma_qp->conn_det_len_data, 0) <
        0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    if (send (oob_sock_client, &rdma_qp->conn_det_len_notif, sizeof (size_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (send (oob_sock_client, rdma_qp->connection_details_notif, rdma_qp->conn_det_len_notif, 0) <
        0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_INFO << "Connect DOCA RDMA to remote RDMA -- data";
    result = doca_rdma_connect (rdma_qp->rdma_data,
                                remote_conn_details_data,
                                remote_conn_details_data_len,
                                rdma_qp->connection_data);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function doca_rdma_connect failed " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    free (remote_conn_details_data);
    remote_conn_details_data = NULL;

    /* Connect local rdma to the remote rdma */
    NIXL_INFO << "Connect DOCA RDMA to remote RDMA -- notif";
    result = doca_rdma_connect (rdma_qp->rdma_notif,
                                remote_conn_details_notif,
                                remote_conn_details_notif_len,
                                rdma_qp->connection_notif);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function doca_rdma_connect failed " << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    free (remote_conn_details_notif);
    remote_conn_details_notif = NULL;

    return NIXL_SUCCESS;
}

/****************************************
 * Connection management
 *****************************************/

nixl_status_t
nixlDocaEngine::getConnInfo (std::string &str) const {
    std::stringstream ss;
    ss << (int)ipv4_addr[0] << "." << (int)ipv4_addr[1] << "." << (int)ipv4_addr[2] << "."
       << (int)ipv4_addr[3];
    str = ss.str();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::connect (const std::string &remote_agent) {
    // Already connected to remote QP at loadRemoteConnInfo time
    // TODO: Connect part should be moved here from loadRemoteConnInfo
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::disconnect (const std::string &remote_agent) {
    // Disconnection should be handled here
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::loadRemoteConnInfo (const std::string &remote_agent,
                                    const std::string &remote_conn_info) {

    // TODO: Connect part should be moved into connect() method
    nixlDocaConnection conn;
    size_t size = remote_conn_info.size();
    // TODO: eventually std::byte?
    char *addr = new char[size];

    if (remoteConnMap.find (remote_agent) != remoteConnMap.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes::_stringToBytes ((void *)addr, remote_conn_info, size);

    int ret = oob_connection_client_setup (addr, &oob_sock_client);
    if (ret < 0) {
        NIXL_ERROR << "Can't connect to server " << ret;
        return NIXL_ERR_BACKEND;
    }

    sendLocalAgent (oob_sock_client, remote_agent);
    if (addRdmaQp (remote_agent) == NIXL_SUCCESS) {
        nixlDocaInitNotif (remote_agent, ddev, gdevs[0].second);
        connectClientRdmaQp (oob_sock_client, remote_agent);
    } else
        NIXL_WARN << "loadRemoteConnInfo: agent " << remote_agent << " already present";

    conn.remoteAgent = remote_agent;
    conn.connected = true;
    remoteConnMap[remote_agent] = conn;

    NIXL_INFO << "DOCA loadRemoteConnInfo connected agent " << remote_agent;

    close (oob_sock_client);

    delete[] addr;

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
 *****************************************/
nixl_status_t
nixlDocaEngine::registerMem (const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    nixlDocaPrivateMetadata *priv = new nixlDocaPrivateMetadata;
    doca_error_t result;

    auto it = std::find_if (
            gdevs.begin(), gdevs.end(), [&mem] (std::pair<uint32_t, struct doca_gpu *> &x) {
                return x.first == mem.devId;
            });
    if (it == gdevs.end()) {
        NIXL_ERROR << "Can't register memory for unknown device " << mem.devId;
        return NIXL_ERR_INVALID_PARAM;
    }

    result = create_doca_mmap ((void *)mem.addr, 1, (size_t)mem.len, ddev, &priv->mem.mmap);
    if (result != DOCA_SUCCESS) goto error;

    /* export mmap for rdma */
    result = doca_mmap_export_rdma (
            priv->mem.mmap, ddev, (const void **)&(priv->mem.export_mmap), &(priv->mem.export_len));
    if (result != DOCA_SUCCESS) goto error;

    priv->mem.addr = (void *)mem.addr;
    priv->mem.len = mem.len;
    priv->mem.devId = mem.devId;
    priv->remoteMmapStr =
            nixlSerDes::_bytesToString ((void *)priv->mem.export_mmap, priv->mem.export_len);

    /* Local buffer array */
    result = create_doca_buf_arr (priv->mem.mmap,
                                  1,
                                  (size_t)mem.len,
                                  gdevs[0].second,
                                  &(priv->mem.barr),
                                  &(priv->mem.barr_gpu));
    if (result != DOCA_SUCCESS) goto error;

    out = (nixlBackendMD *)priv;

    return NIXL_SUCCESS;

error:
    destroy_doca_buf_arr (priv->mem.barr);
    destroy_doca_mmap (priv->mem.mmap);

    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlDocaEngine::deregisterMem (nixlBackendMD *meta) {
    doca_error_t result;
    nixlDocaPrivateMetadata *priv = (nixlDocaPrivateMetadata *)meta;

    result = doca_buf_arr_destroy (priv->mem.barr);
    if (result != DOCA_SUCCESS)
        NIXL_ERROR << "Failed to call doca_buf_arr_destroy " << doca_error_get_descr (result);

    result = doca_mmap_destroy (priv->mem.mmap);
    if (result != DOCA_SUCCESS)
        NIXL_ERROR << "Failed to call doca_mmap_destroy " << doca_error_get_descr (result);

    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::getPublicData (const nixlBackendMD *meta, std::string &str) const {
    const nixlDocaPrivateMetadata *priv = (nixlDocaPrivateMetadata *)meta;
    str = priv->remoteMmapStr;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::loadRemoteMD (const nixlBlobDesc &input,
                              const nixl_mem_t &nixl_mem,
                              const std::string &remote_agent,
                              nixlBackendMD *&output) {
    // TODO: connection setup should move to connect
    doca_error_t result;
    nixlDocaConnection conn;
    nixlDocaPublicMetadata *md = new nixlDocaPublicMetadata;
    size_t size = input.metaInfo.size();
    auto search = remoteConnMap.find (remote_agent);

    if (search == remoteConnMap.end()) {
        NIXL_ERROR << "err: remote connection not found remote_agent " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    conn = (nixlDocaConnection)search->second;

    // directly copy underlying conn struct
    md->conn = conn;

    char *addr = new char[size];
    nixlSerDes::_stringToBytes (addr, input.metaInfo, size);

    result = doca_mmap_create_from_export (NULL, addr, size, ddev, &md->mem.mmap);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function doca_mmap_create_from_export failed "
                   << doca_error_get_descr (result);
        return NIXL_ERR_BACKEND;
    }

    /* Remote buffer array */
    result = create_doca_buf_arr (
            md->mem.mmap, 1, (size_t)size, gdevs[0].second, &(md->mem.barr), &(md->mem.barr_gpu));
    if (result != DOCA_SUCCESS) goto error;

    output = (nixlBackendMD *)md;

    delete[] addr;

    return NIXL_SUCCESS;

error:
    destroy_doca_buf_arr (md->mem.barr);

    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlDocaEngine::unloadMD (nixlBackendMD *input) {
    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
 *****************************************/
nixl_status_t
nixlDocaEngine::prepXfer (const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    uint32_t pos;
    nixlDocaBckndReq *treq = new nixlDocaBckndReq;
    nixlDocaPrivateMetadata *lmd;
    nixlDocaPublicMetadata *rmd;
    uint32_t lcnt = (uint32_t)local.descCount();
    uint32_t rcnt = (uint32_t)remote.descCount();
    uint32_t stream_id;
    struct nixlDocaRdmaQp *rdma_qp;

    // check device id from local dlist mr that should be all the same and same of
    // the engine
    for (uint32_t idx = 0; idx < lcnt; idx++) {
        lmd = (nixlDocaPrivateMetadata *)local[idx].metadataP;
        if (lmd->mem.devId != gdevs[0].first) return NIXL_ERR_INVALID_PARAM;
    }

    auto search = qpMap.find (remote_agent);
    if (search == qpMap.end()) {
        NIXL_ERROR << "Can't find remote_agent " << remote_agent;
        return NIXL_ERR_INVALID_PARAM;
    }

    rdma_qp = search->second;

    if (lcnt != rcnt) return NIXL_ERR_INVALID_PARAM;

    if (lcnt == 0) return NIXL_ERR_INVALID_PARAM;

    if (opt_args->customParam.empty()) {
        stream_id = (xferStream.fetch_add (1) & (nstreams - 1));
        treq->stream = post_stream[stream_id];
    } else {
        treq->stream = (cudaStream_t) * ((uintptr_t *)opt_args->customParam.data());
    }

#if 0
            auto it = std::find_if(gdevs.begin(), gdevs.end(),
                    [&treq](std::pair<uint32_t, struct doca_gpu*> &x)
                    { return x.first == treq->devId; }
                );
            if (it == gdevs.end()) {
                NIXL_ERROR << "Can't prepare transfer for unknown device " << treq->devId << std::endl;
                return NIXL_ERR_INVALID_PARAM;
            }
#endif

    treq->start_pos = (xferRingPos.fetch_add (1) & (DOCA_XFER_REQ_MAX - 1));
    pos = treq->start_pos;

    do {
        for (uint32_t idx = 0; idx < lcnt && idx < DOCA_XFER_REQ_SIZE; idx++) {
            size_t lsize = local[idx].len;
            size_t rsize = remote[idx].len;
            if (lsize != rsize) return NIXL_ERR_INVALID_PARAM;

            lmd = (nixlDocaPrivateMetadata *)local[idx].metadataP;
            rmd = (nixlDocaPublicMetadata *)remote[idx].metadataP;

            xferReqRingCpu[pos].larr[idx] = (uintptr_t)lmd->mem.barr_gpu;
            xferReqRingCpu[pos].rarr[idx] = (uintptr_t)rmd->mem.barr_gpu;
            xferReqRingCpu[pos].size[idx] = lsize;
            xferReqRingCpu[pos].num++;
        }

        xferReqRingCpu[pos].last_rsvd = last_rsvd_flags;
        xferReqRingCpu[pos].last_posted = last_posted_flags;

        xferReqRingCpu[pos].rdma_gpu_data = rdma_qp->rdma_gpu_data;
        xferReqRingCpu[pos].rdma_gpu_notif = rdma_qp->rdma_gpu_notif;

        if (lcnt > DOCA_XFER_REQ_SIZE) {
            lcnt -= DOCA_XFER_REQ_SIZE;
            pos = (xferRingPos.fetch_add (1) & (DOCA_XFER_REQ_MAX - 1));
        } else {
            lcnt = 0;
        }
    } while (lcnt > 0);

    treq->end_pos = xferRingPos;

    if (opt_args && opt_args->hasNotif) {
        struct nixlDocaNotif *notif;

        auto search = notifMap.find (remote_agent);
        if (search == notifMap.end()) {
            NIXL_ERROR << "Can't find notif for remote_agent " << remote_agent;
            return NIXL_ERR_INVALID_PARAM;
        }

        notif = search->second;

        // Check notifMsg size
        std::string newMsg = msg_tag_start + std::to_string (opt_args->notifMsg.size()) +
                msg_tag_end + opt_args->notifMsg;

        xferReqRingCpu[treq->end_pos - 1].has_notif_msg_idx =
                (notif->send_pi.fetch_add (1) & (notif->elems_num - 1));
        xferReqRingCpu[treq->end_pos - 1].msg_sz = newMsg.size();
        xferReqRingCpu[treq->end_pos - 1].notif_barr_gpu = notif->send_barr_gpu;

        memcpy (notif->send_addr +
                        (xferReqRingCpu[treq->end_pos - 1].has_notif_msg_idx * notif->elems_size),
                newMsg.c_str(),
                newMsg.size());

        NIXL_DEBUG << "DOCA prepXfer with notif to " << remote_agent << " at "
                   << xferReqRingCpu[treq->end_pos - 1].has_notif_msg_idx << " msg " << newMsg
                   << " to " << remote_agent;

    } else {
        xferReqRingCpu[treq->end_pos - 1].has_notif_msg_idx = DOCA_NOTIF_NULL;
    }

    NIXL_DEBUG << "DOCA REQUEST from " << treq->start_pos << " to " << treq->end_pos - 1
               << " stream " << stream_id << std::endl;

    treq->backendHandleGpu = 0;

    handle = treq;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::postXfer (const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    nixlDocaBckndReq *treq = (nixlDocaBckndReq *)handle;

    for (uint32_t idx = treq->start_pos; idx < treq->end_pos; idx++) {
        xferReqRingCpu[idx].id =
                (lastPostedReq.fetch_add (1) & (DOCA_MAX_COMPLETION_INFLIGHT_MASK));
        completion_list_cpu[xferReqRingCpu[idx].id].xferReqRingGpu = xferReqRingGpu + idx;
        completion_list_cpu[xferReqRingCpu[idx].id].completed = 0;

        switch (operation) {
        case NIXL_READ:
            doca_kernel_read (treq->stream, xferReqRingCpu[idx].rdma_gpu_data, xferReqRingGpu, idx);
            break;
        case NIXL_WRITE:
            doca_kernel_write (
                    treq->stream, xferReqRingCpu[idx].rdma_gpu_data, xferReqRingGpu, idx);
            break;
        default:
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlDocaEngine::checkXfer (nixlBackendReqH *handle) const {
    nixlDocaBckndReq *treq = (nixlDocaBckndReq *)handle;
    uint32_t completion_index;

    for (uint32_t idx = treq->start_pos; idx < treq->end_pos; idx++) {
        completion_index = xferReqRingCpu[idx].id & (DOCA_MAX_COMPLETION_INFLIGHT_MASK);

        if (((volatile docaXferCompletion *)completion_list_cpu)[completion_index].completed == 1) {
            *((volatile uint8_t *)&xferReqRingCpu[idx].in_use) = 0;
            NIXL_DEBUG << "DOCA checkXfer pos " << idx << " compl_idx " << completion_index
                       << " COMPLETED!\n";
            return NIXL_SUCCESS;
        } else
            return NIXL_IN_PROG;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::releaseReqH (nixlBackendReqH *handle) const {
    uint32_t tmp = xferRingPos.load() & (DOCA_XFER_REQ_MAX - 1);
    if (((volatile docaXferCompletion *)completion_list_cpu)[tmp].completed > 0)
        return NIXL_SUCCESS;
    else
        return NIXL_IN_PROG;
}

nixl_status_t
nixlDocaEngine::getNotifs (notif_list_t &notif_list) {
    uint32_t recv_idx;
    std::string msg_src;
    uint32_t num_msg = 0;
    char *addr;
    size_t position;

    // Lock required to prevent inconsistency if another notifyQp (new peer) is added
    // while getNotifs is running
    notifLock.lock();
    for (auto &notif : notifMap) {
        ((volatile struct docaNotifRecv *)notif_progress_cpu)->rdma_qp =
                qpMap[notif.first]->rdma_gpu_notif;
        std::atomic_thread_fence (std::memory_order_release);
        while (((volatile struct docaNotifRecv *)notif_progress_cpu)->rdma_qp != nullptr)
            ;
        num_msg = ((volatile struct docaNotifRecv *)notif_progress_cpu)->num_msg;
        while (num_msg > 0) {
            NIXL_DEBUG << "CPU num_msg " << num_msg;

            recv_idx = notif.second->recv_pi.load() & (DOCA_MAX_NOTIF_INFLIGHT - 1);
            addr = (char *)(notif.second->recv_addr + (recv_idx * notif.second->elems_size));
            msg_src = addr;
            position = msg_src.find (msg_tag_start);

            NIXL_DEBUG << "getNotifs idx " << recv_idx << "addr "
                       << (void *)((notif.second->recv_addr +
                                    (recv_idx * notif.second->elems_size)))
                       << " msg " << msg_src << " position " << position << std::endl;

            if (position != std::string::npos && position == 0) {
                unsigned last = msg_src.find (msg_tag_end);
                std::string msg_sz =
                        msg_src.substr (position + msg_tag_start.size(), last - position);
                int sz = std::stoi (msg_sz);

                std::string msg (addr + last + msg_tag_end.size(),
                                 addr + last + msg_tag_end.size() + sz);

                NIXL_DEBUG << "getNotifs propagating notif from " << notif.first << " msg " << msg
                           << " size " << sz << " num " << num_msg << std::endl;

                notif_list.push_back (std::pair (notif.first, msg));
                // Tag cleanup
                memset (addr, 0, msg_tag_start.size());
                recv_idx = notif.second->recv_pi.fetch_add (1);
                num_msg--;
            } else {
                std::cerr << "getNotifs error message at " << num_msg;
                break;
            }
        }
    }

    notifLock.unlock();

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::genNotif (const std::string &remote_agent, const std::string &msg) const {
    struct nixlDocaNotif *notif;
    struct doca_gpu_dev_rdma *rdma_gpu;
    uint32_t buf_idx;

    auto searchNotif = notifMap.find (remote_agent);
    if (searchNotif == notifMap.end()) {
        NIXL_ERROR << "Can't find notif for remote_agent " << remote_agent;
        return NIXL_ERR_INVALID_PARAM;
    }

    // 16B is uint16_t msg size
    if (msg.size() > DOCA_MAX_NOTIF_MESSAGE_SIZE - msg_tag_start.size() - msg_tag_end.size() - 16) {
        NIXL_ERROR << "Can't send notif as message size " << msg.size() << " is bigger than max "
                   << (DOCA_MAX_NOTIF_MESSAGE_SIZE - msg_tag_start.size() - msg_tag_end.size() -
                       16);
        return NIXL_ERR_INVALID_PARAM;
    }

    notif = searchNotif->second;

    auto searchQp = qpMap.find (remote_agent);
    if (searchQp == qpMap.end()) {
        NIXL_ERROR << "Can't find QP for remote_agent " << remote_agent;
        return NIXL_ERR_INVALID_PARAM;
    }

    rdma_gpu = searchQp->second->rdma_gpu_notif;
    std::string newMsg = msg_tag_start + std::to_string ((int)msg.size()) + msg_tag_end + msg;
    buf_idx = (notif->send_pi.fetch_add (1) & (notif->elems_num - 1));
    memcpy (notif->send_addr + (buf_idx * notif->elems_size), newMsg.c_str(), newMsg.size());

    NIXL_DEBUG << "genNotif to " << remote_agent << " msg size " << std::to_string ((int)msg.size())
               << " msg " << newMsg << " at " << buf_idx;

    notifSendLock.lock();
    ((volatile struct docaNotifSend *)notif_send_cpu)->barr_gpu = notif->send_barr_gpu;
    ((volatile struct docaNotifSend *)notif_send_cpu)->buf_idx = buf_idx;
    ((volatile struct docaNotifSend *)notif_send_cpu)->msg_sz = newMsg.size();
    // membar
    std::atomic_thread_fence (std::memory_order_release);
    ((volatile struct docaNotifSend *)notif_send_cpu)->rdma_qp = rdma_gpu;
    while (((volatile struct docaNotifSend *)notif_send_cpu)->rdma_qp != nullptr)
        ;
    notifSendLock.unlock();

    return NIXL_SUCCESS;
}
