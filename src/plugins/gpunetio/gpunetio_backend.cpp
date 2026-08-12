/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpunetio_backend.h"
#include "serdes/serdes.h"
#include <arpa/inet.h>
#include <cassert>
#include <stdexcept>
#include <unistd.h>
#include "common/nixl_log.h"
#include <absl/strings/str_split.h>

const char info_delimiter = '-';

/****************************************
 * Constructor/Destructor
 *****************************************/

nixlDocaEngine::nixlDocaEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    std::vector<std::string> ndevs, tmp_gdevs; /* Empty vector */
    doca_error_t result;
    nixl_b_params_t *custom_params = init_params->customParams;
    int ret;
    union ibv_gid rgid;

    result = doca_log_backend_create_standard();
    if (result != DOCA_SUCCESS) throw std::invalid_argument("Can't initialize doca log");

    result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
    if (result != DOCA_SUCCESS) throw std::invalid_argument("Can't initialize doca log");

    result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_ERROR);
    if (result != DOCA_SUCCESS) throw std::invalid_argument("Can't initialize doca log");

    NIXL_INFO << "DOCA network devices ";
    // Temporary: will extend to more GPUs in a dedicated PR
    if (custom_params->count("network_devices") > 1)
        throw std::invalid_argument("Only 1 network device is allowed");

    if (custom_params->count("network_devices") == 0 || (*custom_params)["network_devices"] == "" ||
        (*custom_params)["network_devices"] == "all") {
        ndevs.push_back("mlx5_0");
        NIXL_INFO << "Using default network device mlx5_0";
    } else {
        ndevs = absl::StrSplit((*custom_params)["network_devices"], " ");
        NIXL_INFO << "Using network devices" << ndevs[0];
    }
    NIXL_INFO << std::endl;

    if (custom_params->count("oob_interface") > 0) {
        NIXL_INFO << "DOCA network devices ";
        // Temporary: will extend to more GPUs in a dedicated PR
        if (custom_params->count("oob_interface") > 1)
            throw std::invalid_argument("Only 1 oob interface is allowed");

        oobdev = absl::StrSplit((*custom_params)["oob_interface"], " ");
        NIXL_INFO << "Using oob interface" << oobdev[0];
        NIXL_INFO << std::endl;
    }

    NIXL_INFO << "DOCA GPU devices: ";
    // Temporary: will extend to more GPUs in a dedicated PR
    if (custom_params->count("gpu_devices") > 1)
        throw std::invalid_argument("Only 1 GPU device is allowed");

    if (custom_params->count("gpu_devices") == 0 || (*custom_params)["gpu_devices"] == "" ||
        (*custom_params)["gpu_devices"] == "all") {
        gdevs.push_back(std::pair((uint32_t)0, nullptr));
        NIXL_INFO << "Using default CUDA device ID 0";
    } else {
        tmp_gdevs = absl::StrSplit((*custom_params)["gpu_devices"], " ");
        for (auto &cuda_id : tmp_gdevs) {
            gdevs.push_back(std::pair((uint32_t)std::stoi(cuda_id), nullptr));
            NIXL_INFO << "cuda_id " << cuda_id;
        }
    }
    NIXL_INFO << std::endl;

    nstreams = 0;
    if (custom_params->count("cuda_streams") != 0 && (*custom_params)["cuda_streams"] != "")
        nstreams = std::stoi((*custom_params)["cuda_streams"]);
    if (nstreams == 0) nstreams = DOCA_POST_STREAM_NUM;

    NIXL_INFO << "CUDA streams used for pool mode: " << nstreams;

    /* Open DOCA device */
    verbs_context = open_ib_device((char *)(ndevs[0].c_str()));
    if (verbs_context == nullptr) {
        throw std::invalid_argument("Failed to open DOCA device");
    }

    // Todo: fix any leak if error in constructor
    result = doca_verbs_pd_create(verbs_context, &verbs_pd);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create doca verbs pd: %s", doca_error_get_descr(result);
        throw std::invalid_argument("Failed to create doca verbs pd");
    }

    pd = doca_verbs_bridge_verbs_pd_get_ibv_pd(verbs_pd);
    if (pd == NULL) throw std::invalid_argument("Failed to get ibv_pd");

    result = doca_rdma_bridge_open_dev_from_pd(pd, &ddev);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create doca verbs pd: %s", doca_error_get_descr(result);
        throw std::invalid_argument("Failed to create doca verbs pd");
    }

    ret = ibv_query_port(pd->context, 1, &port_attr);
    if (ret) {
        throw std::invalid_argument("Failed to query ibv port attributes");
    }

    gid_index = 0;

    ret = ibv_query_gid(pd->context, 1, gid_index, &rgid);
    if (ret) {
        NIXL_ERROR << "Failed to query ibv gid attributes";
        throw std::invalid_argument("Failed to query ibv gid attributes");
    }
    memcpy(gid.raw, rgid.raw, DOCA_GID_BYTE_LENGTH);

    if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        result = create_verbs_ah_attr(
            verbs_context, gid_index, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH, &verbs_ah_attr);
        if (result != DOCA_SUCCESS)
            throw std::invalid_argument("Failed to create doca verbs ah attributes");

        lid = port_attr.lid;
    } else {
        result = create_verbs_ah_attr(
            verbs_context, gid_index, DOCA_VERBS_ADDR_TYPE_IPv4, &verbs_ah_attr);
        if (result != DOCA_SUCCESS) {
            throw std::invalid_argument("Failed to create doca verbs ah attributes");
        }
    }

    int cuda_id;
    char pciBusId[DOCA_DEVINFO_IBDEV_NAME_SIZE];
    for (auto &item : gdevs) {
        nixlDocaEngineCheckCudaError(
            cudaDeviceGetPCIBusId(pciBusId, DOCA_DEVINFO_IBDEV_NAME_SIZE, item.first),
            "cudaDeviceGetPCIBusId");

        nixlDocaEngineCheckCudaError(cudaDeviceGetByPCIBusId(&cuda_id, pciBusId),
                                     "cudaDeviceGetByPCIBusId");

        /* Initialize default CUDA context implicitly via CUDA RT API */
        cudaSetDevice(cuda_id);
        cudaFree(0);

        result = doca_gpu_create(pciBusId, &item.second);
        if (result != DOCA_SUCCESS)
            NIXL_ERROR << "Failed to create DOCA GPU device " << doca_error_get_descr(result);
    }

    if (oobdev.size() > 0 && oobdev[0] != "") {
        netif_get_addr(oobdev[0].c_str(), AF_INET, &oob_saddr, &oob_netmask);
        struct sockaddr_in *addr_in = (struct sockaddr_in *)&oob_saddr;
        memcpy(ipv4_addr, (uint8_t *)&(addr_in->sin_addr.s_addr), 4);
        NIXL_DEBUG << "Eth IP address " << static_cast<unsigned>(ipv4_addr[0]) << " "
                   << static_cast<unsigned>(ipv4_addr[1]) << " "
                   << static_cast<unsigned>(ipv4_addr[2]) << " "
                   << static_cast<unsigned>(ipv4_addr[3]) << " " << "ifface " << oobdev[0].c_str();
    } else {
        doca_devinfo_get_ipv4_addr(
            doca_dev_as_devinfo(ddev), (uint8_t *)ipv4_addr, DOCA_DEVINFO_IPV4_ADDR_SIZE);
        NIXL_DEBUG << "DOCA IP address " << static_cast<unsigned>(ipv4_addr[0]) << " "
                   << static_cast<unsigned>(ipv4_addr[1]) << " "
                   << static_cast<unsigned>(ipv4_addr[2]) << " "
                   << static_cast<unsigned>(ipv4_addr[3]);
    }

    // DOCA_GPU_MEM_TYPE_GPU_CPU == GDRCopy
    result = doca_gpu_mem_alloc(gdevs[0].second,
                                sizeof(struct docaXferReqGpu) * DOCA_XFER_REQ_MAX,
                                4096,
                                DOCA_GPU_MEM_TYPE_GPU_CPU,
                                (void **)&xferReqRingGpu,
                                (void **)&xferReqRingCpu);
    if (result != DOCA_SUCCESS || xferReqRingGpu == nullptr || xferReqRingCpu == nullptr) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc with DOCA_GPU_MEM_TYPE_GPU_CPU returned "
                   << doca_error_get_descr(result);
        NIXL_ERROR << "Allocating memory with DOCA_GPU_MEM_TYPE_CPU_GPU";
        result = doca_gpu_mem_alloc(gdevs[0].second,
                                    sizeof(struct docaXferReqGpu) * DOCA_XFER_REQ_MAX,
                                    4096,
                                    DOCA_GPU_MEM_TYPE_CPU_GPU,
                                    (void **)&xferReqRingGpu,
                                    (void **)&xferReqRingCpu);
        if (result != DOCA_SUCCESS || xferReqRingGpu == nullptr || xferReqRingCpu == nullptr) {
            NIXL_ERROR << "Function doca_gpu_mem_alloc with DOCA_GPU_MEM_TYPE_CPU_GPU returned "
                       << doca_error_get_descr(result);
            throw std::invalid_argument("Can't allocate memory");
        }
    }

    nixlDocaEngineCheckCudaError(
        cudaMemset(xferReqRingGpu, 0, sizeof(struct docaXferReqGpu) * DOCA_XFER_REQ_MAX),
        "Failed to memset GPU memory");

    result = doca_gpu_mem_alloc(gdevs[0].second,
                                sizeof(uint64_t),
                                4096,
                                DOCA_GPU_MEM_TYPE_GPU,
                                (void **)&last_rsvd_flags,
                                nullptr);
    if (result != DOCA_SUCCESS || last_rsvd_flags == nullptr) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr(result);
    }

    nixlDocaEngineCheckCudaError(cudaMemset(last_rsvd_flags, 0, sizeof(uint64_t)),
                                 "Failed to memset GPU memory");

    result = doca_gpu_mem_alloc(gdevs[0].second,
                                sizeof(uint64_t),
                                4096,
                                DOCA_GPU_MEM_TYPE_GPU,
                                (void **)&last_posted_flags,
                                nullptr);
    if (result != DOCA_SUCCESS || last_posted_flags == nullptr) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr(result);
    }

    nixlDocaEngineCheckCudaError(cudaMemset(last_posted_flags, 0, sizeof(uint64_t)),
                                 "Failed to memset GPU memory");

    nixlDocaEngineCheckCudaError(cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking),
                                 "Failed to create CUDA stream");
    for (int i = 0; i < nstreams; i++)
        nixlDocaEngineCheckCudaError(
            cudaStreamCreateWithFlags(&post_stream[i], cudaStreamNonBlocking),
            "Failed to create CUDA stream");
    xferStream = 0;

    result = doca_gpu_mem_alloc(gdevs[0].second,
                                sizeof(struct docaXferCompletion) * DOCA_MAX_COMPLETION_INFLIGHT,
                                4096,
                                DOCA_GPU_MEM_TYPE_CPU_GPU,
                                (void **)&completion_list_gpu,
                                (void **)&completion_list_cpu);
    if (result != DOCA_SUCCESS || completion_list_gpu == nullptr ||
        completion_list_cpu == nullptr) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr(result);
    }

    memset(
        completion_list_cpu, 0, sizeof(struct docaXferCompletion) * DOCA_MAX_COMPLETION_INFLIGHT);

    // DOCA_GPU_MEM_TYPE_GPU_CPU == GDRCopy
    result = doca_gpu_mem_alloc(gdevs[0].second,
                                sizeof(uint32_t),
                                4096,
                                DOCA_GPU_MEM_TYPE_GPU_CPU,
                                (void **)&wait_exit_gpu,
                                (void **)&wait_exit_cpu);
    if (result != DOCA_SUCCESS || wait_exit_gpu == nullptr || wait_exit_cpu == nullptr) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc with DOCA_GPU_MEM_TYPE_GPU_CPU returned "
                   << doca_error_get_descr(result);
        NIXL_ERROR << "Allocating memory with DOCA_GPU_MEM_TYPE_CPU_GPU";
        result = doca_gpu_mem_alloc(gdevs[0].second,
                                    sizeof(uint32_t),
                                    4096,
                                    DOCA_GPU_MEM_TYPE_CPU_GPU,
                                    (void **)&wait_exit_gpu,
                                    (void **)&wait_exit_cpu);
        if (result != DOCA_SUCCESS || wait_exit_gpu == nullptr || wait_exit_cpu == nullptr) {
            NIXL_ERROR << "Function doca_gpu_mem_alloc with DOCA_GPU_MEM_TYPE_CPU_GPU returned "
                       << doca_error_get_descr(result);
            throw std::invalid_argument("Can't allocate memory");
        }
    }

    ((volatile uint8_t *)wait_exit_cpu)[0] = 0;

    result = doca_gpu_mem_alloc(gdevs[0].second,
                                sizeof(struct docaNotif),
                                4096,
                                DOCA_GPU_MEM_TYPE_CPU_GPU,
                                (void **)&notif_fill_gpu,
                                (void **)&notif_fill_cpu);
    if (result != DOCA_SUCCESS || notif_fill_gpu == nullptr || notif_fill_cpu == nullptr) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr(result);
    }

    result = doca_gpu_mem_alloc(gdevs[0].second,
                                sizeof(struct docaNotif),
                                4096,
                                DOCA_GPU_MEM_TYPE_CPU_GPU,
                                (void **)&notif_progress_gpu,
                                (void **)&notif_progress_cpu);
    if (result != DOCA_SUCCESS || notif_progress_gpu == nullptr || notif_progress_cpu == nullptr) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr(result);
    }

    memset(notif_progress_cpu, 0, sizeof(struct docaNotif));

    result = doca_gpu_mem_alloc(gdevs[0].second,
                                sizeof(struct docaNotif),
                                4096,
                                DOCA_GPU_MEM_TYPE_CPU_GPU,
                                (void **)&notif_send_gpu,
                                (void **)&notif_send_cpu);
    if (result != DOCA_SUCCESS || notif_send_gpu == nullptr || notif_send_cpu == nullptr) {
        NIXL_ERROR << "Function doca_gpu_mem_alloc return " << doca_error_get_descr(result);
    }

    memset(notif_send_cpu, 0, sizeof(struct docaNotif));

    // We may need a GPU warmup with relevant DOCA engine kernels
    doca_kernel_write(0, nullptr, nullptr, 0);
    doca_kernel_read(0, nullptr, nullptr, 0);
    nixlDocaEngineCheckCudaError(cudaStreamSynchronize(0), "stream synchronize");

    // Warmup
    doca_kernel_progress(
        wait_stream, nullptr, notif_fill_gpu, notif_progress_gpu, notif_send_gpu, wait_exit_gpu);
    nixlDocaEngineCheckCudaError(cudaStreamSynchronize(wait_stream), "stream synchronize");
    doca_kernel_progress(wait_stream,
                         completion_list_gpu,
                         notif_fill_gpu,
                         notif_progress_gpu,
                         notif_send_gpu,
                         wait_exit_gpu);

    lastPostedReq = 0;
    xferRingPos = 0;

    progressThreadStart();
}

nixl_mem_list_t
nixlDocaEngine::getSupportedMems() const {
    return {DRAM_SEG, VRAM_SEG};
}

nixlDocaEngine::~nixlDocaEngine() {
    doca_error_t result;

    NIXL_DEBUG << "Before progressThreadStop ";
    progressThreadStop();

    ((volatile uint8_t *)wait_exit_cpu)[0] = 1;
    NIXL_DEBUG << "Before cudaStreamSynchronize ";
    nixlDocaEngineCheckCudaError(cudaStreamSynchronize(wait_stream), "stream synchronize");
    nixlDocaEngineCheckCudaError(cudaStreamDestroy(wait_stream), "stream destroy");
    doca_gpu_mem_free(gdevs[0].second, wait_exit_gpu);
    doca_gpu_mem_free(gdevs[0].second, xferReqRingGpu);
    doca_gpu_mem_free(gdevs[0].second, last_rsvd_flags);
    doca_gpu_mem_free(gdevs[0].second, last_posted_flags);

    for (int i = 0; i < nstreams; i++) {
        NIXL_DEBUG << "Before cudaStreamSynchronize post_stream " << i;
        nixlDocaEngineCheckCudaError(cudaStreamSynchronize(post_stream[i]), "stream synchronize");
        nixlDocaEngineCheckCudaError(cudaStreamDestroy(post_stream[i]), "stream destroy");
    }

    NIXL_DEBUG << "Before nixlDocaDestroyNotif ";
    for (auto notif : notifMap)
        nixlDocaDestroyNotif(gdevs[0].second, notif.second);

    doca_gpu_mem_free(gdevs[0].second, notif_fill_gpu);
    doca_gpu_mem_free(gdevs[0].second, notif_progress_gpu);
    doca_gpu_mem_free(gdevs[0].second, notif_send_gpu);
    doca_gpu_mem_free(gdevs[0].second, completion_list_gpu);

    NIXL_DEBUG << "Before qpMap.clear ";

    qpMap.clear();

    result = doca_dev_close(ddev);
    if (result != DOCA_SUCCESS)
        NIXL_ERROR << "Failed to close DOCA device " << doca_error_get_descr(result);

    result = doca_gpu_destroy(gdevs[0].second);
    if (result != DOCA_SUCCESS)
        NIXL_ERROR << "Failed to close DOCA GPU device " << doca_error_get_descr(result);
}

/****************************************
 * DOCA request management
 *****************************************/

nixl_status_t
nixlDocaEngine::nixlDocaInitNotif(const std::string &remote_agent, doca_dev *dev, doca_gpu *gpu) {
    struct nixlDocaNotif *notif;

    std::lock_guard<std::mutex> lock(notifLock);
    // Same peer can be server or client
    if (notifMap.find(remote_agent) != notifMap.end()) {
        NIXL_INFO << "nixlDocaInitNotif already found " << remote_agent << std::endl;
        return NIXL_SUCCESS;
    }

    notif = new struct nixlDocaNotif;

    notif->elems_num = DOCA_MAX_NOTIF_INFLIGHT;
    notif->elems_size = DOCA_MAX_NOTIF_MESSAGE_SIZE;
    notif->send_addr = (uint8_t *)calloc(notif->elems_size * notif->elems_num, sizeof(uint8_t));
    if (notif->send_addr == nullptr) {
        NIXL_ERROR << "Can't alloc memory for send notif";
        return NIXL_ERR_BACKEND;
    }
    memset(notif->send_addr, 0, notif->elems_size * notif->elems_num);

    try {
        notif->send_mr = std::make_unique<nixl::doca::verbs::mr>(
            gpu, (void *)notif->send_addr, notif->elems_num, notif->elems_size, pd);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }

    notif->recv_addr = (uint8_t *)calloc(notif->elems_size * notif->elems_num, sizeof(uint8_t));
    if (notif->recv_addr == nullptr) {
        NIXL_ERROR << "Can't alloc memory for send notif";
        return NIXL_ERR_BACKEND;
    }
    memset(notif->recv_addr, 0, notif->elems_size * notif->elems_num);

    try {
        notif->recv_mr = std::make_unique<nixl::doca::verbs::mr>(
            gpu, (void *)notif->recv_addr, notif->elems_num, notif->elems_size, pd);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }

    notif->send_pi = 0;
    notif->recv_pi = 0;

    // Ensure notif list is not added twice for the same peer
    notifMap[remote_agent] = notif;
    ((volatile struct docaNotif *)notif_fill_cpu)->msg_buf = (uintptr_t)notif->recv_addr;
    ((volatile struct docaNotif *)notif_fill_cpu)->msg_lkey = notif->recv_mr->get_lkey();
    ((volatile struct docaNotif *)notif_fill_cpu)->msg_size = notif->elems_size;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    ((volatile struct docaNotif *)notif_fill_cpu)->qp_gpu =
        qpMap[remote_agent]->qp_notif->get_qp_gpu_dev();
    while (((volatile struct docaNotif *)notif_fill_cpu)->qp_gpu != nullptr)
        ;

    NIXL_INFO << "nixlDocaInitNotif added new qp for " << remote_agent << std::endl;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::nixlDocaDestroyNotif(doca_gpu *gpu, struct nixlDocaNotif *notif) {
    delete notif;

    return NIXL_SUCCESS;
}

// For now just connection setup, not used for xfers to be a complete progThread, so supportsProgTh
// is false
nixl_status_t
nixlDocaEngine::progressThreadStart() {
    struct sockaddr_in server_addr = {0};
    int enable = 1;
    int result;
    noSyncIters = 32;

    pthrStop = (volatile uint32_t *)calloc(1, sizeof(uint32_t));
    *pthrStop = 0;
    /* Create socket */

    oob_sock_server = socket(AF_INET, SOCK_STREAM, 0);
    if (oob_sock_server < 0) {
        NIXL_ERROR << "Error while creating socket " << oob_sock_server;
        return NIXL_ERR_NOT_SUPPORTED;
    }
    NIXL_INFO << "DOCA Server socket created successfully";

    if (setsockopt(oob_sock_server, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(enable))) {
        NIXL_ERROR << "Error setting socket options";
        close(oob_sock_server);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (setsockopt(oob_sock_server, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable))) {
        NIXL_ERROR << "Error setting socket options";
        close(oob_sock_server);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (oobdev.size() > 0 && oobdev[0] != "") {
        struct sockaddr_in *addr_in = (struct sockaddr_in *)&oob_saddr;
        /* Bind to the set port and IP: */
        addr_in->sin_port = htons(DOCA_RDMA_CM_LOCAL_PORT_SERVER);
        if (bind(oob_sock_server, (struct sockaddr *)addr_in, sizeof(struct sockaddr_in)) < 0) {
            NIXL_ERROR << "Couldn't bind to the port " << DOCA_RDMA_CM_LOCAL_PORT_SERVER;
            close(oob_sock_server);
            return NIXL_ERR_NOT_SUPPORTED;
        }
    } else {
        /* Set port and IP: */
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(DOCA_RDMA_CM_LOCAL_PORT_SERVER);
        server_addr.sin_addr.s_addr = INADDR_ANY; /* listen on any interface */

        /* Bind to the set port and IP: */
        if (bind(oob_sock_server, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            NIXL_ERROR << "Couldn't bind to the port " << DOCA_RDMA_CM_LOCAL_PORT_SERVER;
            close(oob_sock_server);
            return NIXL_ERR_NOT_SUPPORTED;
        }
    }

    NIXL_INFO << "Done with binding";

    /* Listen for clients: */
    if (listen(oob_sock_server, 1) < 0) {
        NIXL_ERROR << "Error while listening";
        close(oob_sock_server);
        return NIXL_ERR_NOT_SUPPORTED;
    }
    NIXL_INFO << "Listening for incoming connections";

    // Start the thread
    // TODO [Relaxed mem] mem barrier to ensure pthr_x updates are complete
    // new (&pthr) std::thread(&nixlDocaEngine::threadProgressFunc, this);

    cuCtxGetCurrent(&main_cuda_ctx);

    result = pthread_create(&server_thread_id, nullptr, threadProgressFunc, (void *)this);
    if (result != 0) {
        NIXL_ERROR << "Failed to create threadProgressFunc thread";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

void
nixlDocaEngine::progressThreadStop() {
    int fake_sock_fd;
    std::stringstream ss;

    ACCESS_ONCE(*pthrStop) = 1;
    ss << (int)ipv4_addr[0] << "." << (int)ipv4_addr[1] << "." << (int)ipv4_addr[2] << "."
       << (int)ipv4_addr[3];
    std::atomic_thread_fence(std::memory_order_seq_cst);
    oob_connection_client_setup(ss.str().c_str(), &fake_sock_fd);
    // pthr.join();
    pthread_join(server_thread_id, nullptr);
    close(oob_sock_server);
    close(fake_sock_fd);
}

uint32_t
nixlDocaEngine::getGpuCudaId() {
    return gdevs[0].first;
}

nixl_status_t
nixlDocaEngine::addRdmaQp(const std::string &remote_agent) {
    struct nixlDocaRdmaQp *rdma_qp;

    std::lock_guard<std::mutex> lock(qpLock);

    NIXL_DEBUG << "addRdmaQp for " << remote_agent << std::endl;

    // if client or server already created this QP, no need to re-create
    if (qpMap.find(remote_agent) != qpMap.end()) {
        return NIXL_IN_PROG;
    }

    NIXL_DEBUG << "DOCA addRdmaQp for remote " << remote_agent << std::endl;

    rdma_qp = new struct nixlDocaRdmaQp;

    try {
        rdma_qp->qp_data =
            std::make_unique<nixl::doca::verbs::qp>(gdevs[0].second,
                                                    ddev,
                                                    verbs_context,
                                                    verbs_pd,
                                                    RDMA_SEND_QUEUE_SIZE,
                                                    RDMA_RECV_QUEUE_SIZE,
                                                    DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }

    rdma_qp->qpn_data = doca_verbs_qp_get_qpn(rdma_qp->qp_data->get_qp());

    /* NOTIF QP */
    try {
        rdma_qp->qp_notif =
            std::make_unique<nixl::doca::verbs::qp>(gdevs[0].second,
                                                    ddev,
                                                    verbs_context,
                                                    verbs_pd,
                                                    RDMA_SEND_QUEUE_SIZE,
                                                    RDMA_RECV_QUEUE_SIZE,
                                                    DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }

    rdma_qp->qpn_notif = doca_verbs_qp_get_qpn(rdma_qp->qp_notif->get_qp());

    qpMap[remote_agent] = rdma_qp;

    NIXL_DEBUG << "DOCA addRdmaQp new QP added for " << remote_agent;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::connectClientRdmaQp(int oob_sock_client, const std::string &remote_agent) {
    doca_error_t result;
    struct nixlDocaRdmaQp *rdma_qp = qpMap[remote_agent];
    uint32_t lack = 0, rack = 1;

    NIXL_DEBUG << "connectClientRdmaQp: Send to server data qp connection details";
    // Data QP
    if (send(oob_sock_client, &rdma_qp->qpn_data, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    if (send(oob_sock_client, &rdma_qp->qpn_notif, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (send(oob_sock_client, &gid.raw, sizeof(gid.raw), 0) < 0) {
        NIXL_ERROR << "Failed to send local GID raw address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (send(oob_sock_client, &lid, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to send LID address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Data QP
    NIXL_DEBUG << "connectClientRdmaQp: Receive client remote data qp connection details";
    if (recv(oob_sock_client, &rdma_qp->rqpn_data, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    NIXL_INFO << "Receive remote notif qp connection details";
    if (recv(oob_sock_client, &rdma_qp->rqpn_notif, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (recv(oob_sock_client, &remote_gid.raw, sizeof(gid.raw), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote GID raw address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (recv(oob_sock_client, &dlid, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote GID address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Avoid duplicating RDMA connection to the same QP by client/server threads
    NIXL_DEBUG << "connectClientRdmaQp: before lock";
    // std::lock_guard<std::mutex> lock(connectLock);
    connectLock.lock();
    if (connMap.find(remote_agent) != connMap.end()) {
        NIXL_INFO << "QP for " << remote_agent << " already connected" << std::endl;
        goto sync;
        // return NIXL_SUCCESS;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- data";
    result = connect_verbs_qp(
        this, rdma_qp->qp_data->get_qp(), rdma_qp->rqpn_data, rdma_qp->remote_gid_data);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function connect_verbs_qp data failed " << doca_error_get_descr(result);
        connectLock.unlock();
        return NIXL_ERR_BACKEND;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- notif";
    result = connect_verbs_qp(
        this, rdma_qp->qp_notif->get_qp(), rdma_qp->rqpn_notif, rdma_qp->remote_gid_data);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function connect_verbs_qp notif failed " << doca_error_get_descr(result);
        connectLock.unlock();
        return NIXL_ERR_BACKEND;
    }

sync:
    connectLock.unlock();
    NIXL_DEBUG << "Client recv lack";
    if (recv(oob_sock_client, &lack, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote ACK connection";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Client received lack " << lack;
    if (lack != 1) {
        NIXL_ERROR << "Wrong remote ACK connection value " << lack;
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Client sending rack" << rack;
    if (send(oob_sock_client, &rack, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    connMap[remote_agent] = 1;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::recvRemoteAgentName(int oob_sock_client, std::string &remote_agent) {
    size_t msg_size;

    // Msg
    if (recv(oob_sock_client, &msg_size, sizeof(size_t), 0) < 0) {
        NIXL_ERROR << "Failed to recv msg details";
        close(oob_sock_client);
        return NIXL_ERR_BACKEND;
    }

    if (msg_size == 0) {
        NIXL_ERROR << "recvRemoteAgentName received msg size 0";
        close(oob_sock_client);
        return NIXL_ERR_BACKEND;
    }

    remote_agent.resize(msg_size);

    if (recv(oob_sock_client, remote_agent.data(), msg_size, 0) < 0) {
        NIXL_ERROR << "Failed to recv msg details";
        close(oob_sock_client);
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::sendLocalAgentName(int oob_sock_client) {
    size_t agent_size = localAgent.size();

    if (send(oob_sock_client, &agent_size, sizeof(size_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        return NIXL_ERR_BACKEND;
    }

    if (send(oob_sock_client, localAgent.c_str(), localAgent.size(), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << " sendLocalAgentName localAgent " << localAgent << std::endl;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::connectServerRdmaQp(int oob_sock_client, const std::string &remote_agent) {
    doca_error_t result;
    struct nixlDocaRdmaQp *rdma_qp = qpMap[remote_agent]; // validate
    uint32_t lack = 0, rack = 1;

    NIXL_DEBUG << "DOCA connectServerRdmaQp for agent " << remote_agent.c_str();

    // Data QP
    NIXL_DEBUG << "Server Receive client remote data qp connection details";
    if (recv(oob_sock_client, &rdma_qp->rqpn_data, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    NIXL_DEBUG << "Server Receive remote notif qp connection details";
    if (recv(oob_sock_client, &rdma_qp->rqpn_notif, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (recv(oob_sock_client, &remote_gid.raw, sizeof(gid.raw), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote GID raw address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (recv(oob_sock_client, &dlid, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote GID address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Data QP
    NIXL_DEBUG << "Server Send remote notif qp connection details";
    if (send(oob_sock_client, &rdma_qp->qpn_data, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    NIXL_DEBUG << "Server Send remote notif qp connection details";
    if (send(oob_sock_client, &rdma_qp->qpn_notif, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (send(oob_sock_client, &gid.raw, sizeof(gid.raw), 0) < 0) {
        NIXL_ERROR << "Failed to send local GID raw address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Server Send remote notif qp connection details 4";
    if (send(oob_sock_client, &lid, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to send local GID address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Avoid duplicating RDMA connection to the same QP by client/server threads
    NIXL_DEBUG << "connectServerRdmaQp: before lock";
    // std::lock_guard<std::mutex> lock(connectLock);
    connectLock.lock();
    if (connMap.find(remote_agent) != connMap.end()) {
        NIXL_DEBUG << "QP for " << remote_agent << " already connected";
        goto sync;
        // return NIXL_SUCCESS;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- data";
    result = connect_verbs_qp(
        this, rdma_qp->qp_data->get_qp(), rdma_qp->rqpn_data, rdma_qp->remote_gid_data);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function connect_verbs_qp data failed " << doca_error_get_descr(result);
        connectLock.unlock();
        return NIXL_ERR_BACKEND;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- notif";
    result = connect_verbs_qp(
        this, rdma_qp->qp_notif->get_qp(), rdma_qp->rqpn_notif, rdma_qp->remote_gid_data);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function connect_verbs_qp notif failed " << doca_error_get_descr(result);
        connectLock.unlock();
        return NIXL_ERR_BACKEND;
    }

    connMap[remote_agent] = 1;

sync:

    connectLock.unlock();

    NIXL_DEBUG << "Server send rack " << rack;
    if (send(oob_sock_client, &rack, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Server recv lack";
    if (recv(oob_sock_client, &lack, sizeof(uint32_t), 0) < 0) {
        NIXL_ERROR << "Failed to receive remote ACK connection";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Server received lack " << lack;
    if (lack != 1) {
        NIXL_ERROR << "Wrong remote ACK connection value " << lack;
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

/****************************************
 * Connection management
 *****************************************/

nixl_status_t
nixlDocaEngine::getConnInfo(std::string &str) const {
    std::stringstream ss;
    ss << (int)ipv4_addr[0] << "." << (int)ipv4_addr[1] << "." << (int)ipv4_addr[2] << "."
       << (int)ipv4_addr[3];
    str = ss.str();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::connect(const std::string &remote_agent) {
    // Already connected to remote QP at loadRemoteConnInfo time
    // TODO: Connect part should be moved here from loadRemoteConnInfo
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::disconnect(const std::string &remote_agent) {
    // Disconnection should be handled here
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                   const std::string &remote_conn_info) {

    int oob_sock_client;

    // TODO: Connect part should be moved into connect() method
    nixlDocaConnection conn;
    size_t size = remote_conn_info.size();
    // TODO: eventually std::byte?
    char *addr = new char[size];

    if (remoteConnMap.find(remote_agent) != remoteConnMap.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes::_stringToBytes((void *)addr, remote_conn_info, size);

    int ret = oob_connection_client_setup(addr, &oob_sock_client);
    if (ret < 0) {
        NIXL_ERROR << "Can't connect to server " << ret;
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << "loadRemoteConnInfo calling addRdmaQp for " << remote_agent.c_str();
    sendLocalAgentName(oob_sock_client);
    addRdmaQp(remote_agent);
    nixlDocaInitNotif(remote_agent, ddev, gdevs[0].second);
    connectClientRdmaQp(oob_sock_client, remote_agent);

    conn.remoteAgent = remote_agent;
    conn.connected = true;
    // if client or server already created this QP, no need to re-create
    if (remoteConnMap.find(remote_agent) == remoteConnMap.end()) {
        remoteConnMap[remote_agent] = conn;
        NIXL_INFO << "remoteConnMap extended with remote agent " << remote_agent << std::endl;
    }

    NIXL_INFO << "DOCA loadRemoteConnInfo connected agent " << remote_agent;

    close(oob_sock_client);

    delete[] addr;

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
 *****************************************/
nixl_status_t
nixlDocaEngine::registerMem(const nixlBlobDesc &mem,
                            const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) {
    nixlDocaPrivateMetadata *priv = new nixlDocaPrivateMetadata;
    std::stringstream ss;

    auto it = std::find_if(gdevs.begin(), gdevs.end(), [&mem](std::pair<uint32_t, doca_gpu *> &x) {
        return x.first == mem.devId;
    });

    if (it == gdevs.end()) {
        NIXL_ERROR << "Can't register memory for unknown device " << mem.devId;
        return NIXL_ERR_INVALID_PARAM;
    }

    try {
        priv->mr = std::make_unique<nixl::doca::verbs::mr>(
            it->second, (void *)mem.addr, 1, (size_t)mem.len, pd);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }

    priv->devId = mem.devId;
    ss << (uint32_t)priv->mr->get_rkey() << info_delimiter << ((uintptr_t)priv->mr->get_addr())
       << info_delimiter << ((size_t)priv->mr->get_tot_size());
    priv->remoteMrStr = ss.str();

    out = (nixlBackendMD *)priv;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::deregisterMem(nixlBackendMD *meta) {
    nixlDocaPrivateMetadata *priv = (nixlDocaPrivateMetadata *)meta;

    delete priv;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    const nixlDocaPrivateMetadata *priv = (nixlDocaPrivateMetadata *)meta;
    str = priv->remoteMrStr;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::loadRemoteMD(const nixlBlobDesc &input,
                             const nixl_mem_t &nixl_mem,
                             const std::string &remote_agent,
                             nixlBackendMD *&output) {
    // TODO: connection setup should move to connect
    nixlDocaConnection conn;
    std::vector<std::string> tokens;
    std::string token;
    nixlDocaPublicMetadata *md = new nixlDocaPublicMetadata;
    auto search = remoteConnMap.find(remote_agent);

    if (search == remoteConnMap.end()) {
        NIXL_ERROR << "err: remote connection not found remote_agent " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    conn = (nixlDocaConnection)search->second;

    // directly copy underlying conn struct
    md->conn = conn;

    std::stringstream ss(input.metaInfo.data());
    while (std::getline(ss, token, info_delimiter))
        tokens.push_back(token);

    uint32_t rkey = static_cast<uint32_t>(atoi(tokens[0].c_str()));
    uintptr_t addr = static_cast<uintptr_t>(atol(tokens[1].c_str()));
    size_t tot_size = static_cast<size_t>(atol(tokens[2].c_str()));

    // Empty mmap, filled with imported data
    try {
        md->mr = std::make_unique<nixl::doca::verbs::mr>((void *)addr, tot_size, rkey);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }

    output = (nixlBackendMD *)md;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::unloadMD(nixlBackendMD *input) {
    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
 *****************************************/
nixl_status_t
nixlDocaEngine::prepXfer(const nixl_xfer_op_t &operation,
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
    uintptr_t notif_addr;

    // TODO: check device id from local dlist mr that should be all the same and same of
    // the engine
    for (uint32_t idx = 0; idx < lcnt; idx++) {
        lmd = (nixlDocaPrivateMetadata *)local[idx].metadataP;
        if (lmd->devId != gdevs[0].first) return NIXL_ERR_INVALID_PARAM;
    }

    auto search = qpMap.find(remote_agent);
    if (search == qpMap.end()) {
        NIXL_ERROR << "Can't find remote_agent " << remote_agent;
        return NIXL_ERR_INVALID_PARAM;
    }

    rdma_qp = search->second;

    if (lcnt != rcnt) return NIXL_ERR_INVALID_PARAM;

    if (lcnt == 0) return NIXL_ERR_INVALID_PARAM;

    if (opt_args->customParam.empty()) {
        stream_id = (xferStream.fetch_add(1) & (nstreams - 1));
        treq->stream = post_stream[stream_id];
    } else {
        treq->stream = (cudaStream_t) * ((uintptr_t *)opt_args->customParam.data());
    }

    treq->start_pos = (xferRingPos.fetch_add(1) & (DOCA_XFER_REQ_MAX - 1));
    pos = treq->start_pos;

    do {
        for (uint32_t idx = 0; idx < lcnt && idx < DOCA_XFER_REQ_SIZE; idx++) {
            size_t lsize = local[idx].len;
            size_t rsize = remote[idx].len;
            if (lsize != rsize) return NIXL_ERR_INVALID_PARAM;

            lmd = (nixlDocaPrivateMetadata *)local[idx].metadataP;
            rmd = (nixlDocaPublicMetadata *)remote[idx].metadataP;

            xferReqRingCpu[pos].lbuf[idx] = (uintptr_t)lmd->mr->get_addr();
            xferReqRingCpu[pos].lkey[idx] = (uintptr_t)lmd->mr->get_lkey();
            xferReqRingCpu[pos].rbuf[idx] = (uintptr_t)rmd->mr->get_addr();
            xferReqRingCpu[pos].rkey[idx] = (uintptr_t)rmd->mr->get_rkey();
            xferReqRingCpu[pos].size[idx] = lsize;
            xferReqRingCpu[pos].num++;
        }

        xferReqRingCpu[pos].last_rsvd = last_rsvd_flags;
        xferReqRingCpu[pos].last_posted = last_posted_flags;

        xferReqRingCpu[pos].qp_data = rdma_qp->qp_data->get_qp_gpu_dev();
        xferReqRingCpu[pos].qp_notif = rdma_qp->qp_notif->get_qp_gpu_dev();

        if (lcnt > DOCA_XFER_REQ_SIZE) {
            lcnt -= DOCA_XFER_REQ_SIZE;
            pos = (xferRingPos.fetch_add(1) & (DOCA_XFER_REQ_MAX - 1));
        } else {
            lcnt = 0;
        }
    } while (lcnt > 0);

    treq->end_pos = xferRingPos;

    if (opt_args && opt_args->hasNotif) {
        struct nixlDocaNotif *notif;

        auto search = notifMap.find(remote_agent);
        if (search == notifMap.end()) {
            NIXL_ERROR << "Can't find notif for remote_agent " << remote_agent;
            return NIXL_ERR_INVALID_PARAM;
        }

        notif = search->second;

        // Check notifMsg size
        std::string newMsg = msg_tag_start + std::to_string(opt_args->notifMsg.size()) +
            msg_tag_end + opt_args->notifMsg;

        notif_addr =
            (uintptr_t)(notif->send_addr +
                        (xferReqRingCpu[treq->end_pos - 1].has_notif_msg_idx * notif->elems_size));
        xferReqRingCpu[treq->end_pos - 1].has_notif_msg_idx =
            (notif->send_pi.fetch_add(1) & (notif->elems_num - 1));
        xferReqRingCpu[treq->end_pos - 1].msg_sz = newMsg.size();
        xferReqRingCpu[treq->end_pos - 1].lbuf_notif = notif_addr;
        xferReqRingCpu[treq->end_pos - 1].lkey_notif = notif->send_mr->get_lkey();

        memcpy((void *)notif_addr, newMsg.c_str(), newMsg.size());

        NIXL_INFO << "DOCA prepXfer with notif to " << remote_agent << " at "
                  << xferReqRingCpu[treq->end_pos - 1].has_notif_msg_idx << " msg " << newMsg
                  << " to " << remote_agent;

    } else {
        xferReqRingCpu[treq->end_pos - 1].has_notif_msg_idx = DOCA_NOTIF_NULL;
    }

    NIXL_INFO << "DOCA REQUEST from " << treq->start_pos << " to " << treq->end_pos - 1
              << " stream " << stream_id << std::endl;

    treq->backendHandleGpu = 0;

    handle = treq;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::postXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    nixlDocaBckndReq *treq = (nixlDocaBckndReq *)handle;

    for (uint32_t idx = treq->start_pos; idx < treq->end_pos; idx++) {
        xferReqRingCpu[idx].id = (lastPostedReq.fetch_add(1) & (DOCA_MAX_COMPLETION_INFLIGHT_MASK));
        completion_list_cpu[xferReqRingCpu[idx].id].xferReqRingGpu = xferReqRingGpu + idx;
        completion_list_cpu[xferReqRingCpu[idx].id].completed = 0;

        switch (operation) {
        case NIXL_READ:
            doca_kernel_read(treq->stream, xferReqRingCpu[idx].qp_data, xferReqRingGpu, idx);
            break;
        case NIXL_WRITE:
            doca_kernel_write(treq->stream, xferReqRingCpu[idx].qp_data, xferReqRingGpu, idx);
            break;
        default:
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlDocaEngine::checkXfer(nixlBackendReqH *handle) const {
    nixlDocaBckndReq *treq = (nixlDocaBckndReq *)handle;
    uint32_t completion_index;

    for (uint32_t idx = treq->start_pos; idx < treq->end_pos; idx++) {
        completion_index = xferReqRingCpu[idx].id & (DOCA_MAX_COMPLETION_INFLIGHT_MASK);

        if (((volatile docaXferCompletion *)completion_list_cpu)[completion_index].completed == 1) {
            *((volatile uint8_t *)&xferReqRingCpu[idx].in_use) = 0;
            NIXL_INFO << "DOCA checkXfer pos " << idx << " compl_idx " << completion_index
                      << " COMPLETED!\n";
            return NIXL_SUCCESS;
        } else
            return NIXL_IN_PROG;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::releaseReqH(nixlBackendReqH *handle) const {
    uint32_t tmp = xferRingPos.load() & (DOCA_XFER_REQ_MAX - 1);
    if (((volatile docaXferCompletion *)completion_list_cpu)[tmp].completed > 0)
        return NIXL_SUCCESS;
    else
        return NIXL_IN_PROG;
}

nixl_status_t
nixlDocaEngine::getNotifs(notif_list_t &notif_list) {
    uint32_t recv_idx;
    std::string msg_src;
    uint32_t num_msg = 0;
    char *addr;
    size_t position;

    // Lock required to prevent inconsistency if another notifyQp (new peer) is added
    // while getNotifs is running
    std::lock_guard<std::mutex> lock(notifLock);
    for (auto &notif : notifMap) {
        ((volatile struct docaNotif *)notif_progress_cpu)->qp_gpu =
            qpMap[notif.first]->qp_notif->get_qp_gpu_dev();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        while (((volatile struct docaNotif *)notif_progress_cpu)->qp_gpu != nullptr)
            ;
        num_msg = ((volatile struct docaNotif *)notif_progress_cpu)->msg_num;
        while (num_msg > 0) {
            recv_idx = notif.second->recv_pi.load() & (DOCA_MAX_NOTIF_INFLIGHT - 1);
            addr = (char *)(notif.second->recv_addr + (recv_idx * notif.second->elems_size));
            msg_src = addr;

            NIXL_DEBUG << "CPU num_msg " << num_msg << " at " << recv_idx << " addr "
                       << (void *)addr << " msg " << msg_src << std::endl;

            position = msg_src.find(msg_tag_start);

            NIXL_DEBUG << "getNotifs idx " << recv_idx << " addr "
                       << (void *)((notif.second->recv_addr +
                                    (recv_idx * notif.second->elems_size)))
                       << " msg " << msg_src << " position " << (int)position << std::endl;

            if (position != std::string::npos && position == 0) {
                unsigned last = msg_src.find(msg_tag_end);
                std::string msg_sz =
                    msg_src.substr(position + msg_tag_start.size(), last - position);
                int sz = std::stoi(msg_sz);

                std::string msg(addr + last + msg_tag_end.size(),
                                addr + last + msg_tag_end.size() + sz);

                NIXL_DEBUG << "getNotifs propagating notif from " << notif.first << " msg " << msg
                           << " size " << sz << " num " << num_msg << std::endl;

                notif_list.push_back(std::pair(notif.first, msg));
                // Tag cleanup
                memset(addr, 0, msg_tag_start.size());
                recv_idx = notif.second->recv_pi.fetch_add(1);
                num_msg--;
            } else {
                NIXL_ERROR << "getNotifs error message at " << num_msg << " size " << msg_src.size()
                           << " msg " << msg_src;
                break;
            }
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    struct nixlDocaNotif *notif;
    uint32_t buf_idx;
    uintptr_t msg_buf;

    auto searchNotif = notifMap.find(remote_agent);
    if (searchNotif == notifMap.end()) {
        NIXL_ERROR << "genNotif: can't find notif for remote_agent " << remote_agent << std::endl;
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

    auto searchQp = qpMap.find(remote_agent);
    if (searchQp == qpMap.end()) {
        NIXL_ERROR << "Can't find QP for remote_agent " << remote_agent;
        return NIXL_ERR_INVALID_PARAM;
    }

    std::string newMsg = msg_tag_start + std::to_string((int)msg.size()) + msg_tag_end + msg;
    buf_idx = (notif->send_pi.fetch_add(1) & (notif->elems_num - 1));
    msg_buf = (uintptr_t)notif->send_addr + (buf_idx * notif->elems_size);
    memcpy((void *)msg_buf, newMsg.c_str(), newMsg.size());

    NIXL_DEBUG << "genNotif to " << remote_agent << " msg size " << std::to_string((int)msg.size())
               << " msg " << newMsg << " at " << buf_idx << " msg_buf " << msg_buf << "\n";

    std::lock_guard<std::mutex> lock(notifSendLock);
    ((volatile struct docaNotif *)notif_send_cpu)->msg_buf = msg_buf;
    ((volatile struct docaNotif *)notif_send_cpu)->msg_lkey = notif->send_mr->get_lkey();
    ((volatile struct docaNotif *)notif_send_cpu)->msg_size = newMsg.size();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    ((volatile struct docaNotif *)notif_send_cpu)->qp_gpu =
        searchQp->second->qp_notif->get_qp_gpu_dev();
    while (((volatile struct docaNotif *)notif_send_cpu)->qp_gpu != nullptr)
        ;

    return NIXL_SUCCESS;
}
