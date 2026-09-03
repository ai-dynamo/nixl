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
#include <arpa/inet.h>
#include <cassert>
#include <cerrno>
#include <iterator>
#include <stdexcept>
#include <unistd.h>
#include "common/nixl_log.h"
#include <absl/strings/str_split.h>

const char info_delimiter = '-';

namespace {
class ScopedCudaDevice {
public:
    explicit ScopedCudaDevice(int device) : previousDevice(0), restore(false) {
        nixlDocaEngineCheckCudaError(cudaGetDevice(&previousDevice), "Failed to get CUDA device");
        if (previousDevice != device) {
            nixlDocaEngineCheckCudaError(cudaSetDevice(device), "Failed to set CUDA device");
            restore = true;
        }
    }

    ~ScopedCudaDevice() {
        if (restore && cudaSetDevice(previousDevice) != cudaSuccess) {
            NIXL_ERROR << "Failed to restore CUDA device";
        }
    }

private:
    int previousDevice;
    bool restore;
};

int
parseGidIndex(const std::string &value) {
    if (value.empty()) {
        return 0;
    }

    size_t parsed_chars = 0;
    int parsed_value = 0;
    try {
        parsed_value = std::stoi(value, &parsed_chars);
    }
    catch (const std::exception &) {
        throw std::invalid_argument("gid_index must be an integer in the range [0, 255]");
    }

    if (parsed_chars != value.size() || parsed_value < 0 || parsed_value > 255) {
        throw std::invalid_argument("gid_index must be an integer in the range [0, 255]");
    }

    return parsed_value;
}

bool
sendAll(int fd, const void *buffer, size_t size) {
    const auto *cursor = static_cast<const uint8_t *>(buffer);
    while (size > 0) {
        const ssize_t sent = send(fd, cursor, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent <= 0) {
            return false;
        }
        cursor += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

bool
recvAll(int fd, void *buffer, size_t size) {
    auto *cursor = static_cast<uint8_t *>(buffer);
    while (size > 0) {
        const ssize_t received = recv(fd, cursor, size, 0);
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received <= 0) {
            return false;
        }
        cursor += received;
        size -= static_cast<size_t>(received);
    }
    return true;
}
} // namespace

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

    for (auto &reserved : xferReqReserved) {
        reserved.store(false, std::memory_order_relaxed);
    }

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

    gid_index = parseGidIndex((*custom_params)["gid_index"]);
    NIXL_INFO << "RoCE GID index: " << gid_index;

    local_port = parseGpunetioOobPort((*custom_params)["oob_port"]);
    NIXL_INFO << "OOB listen port: " << local_port;
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

    // The first configured GPU owns the QPs, streams, and progress kernels.
    nixlDocaEngineCheckCudaError(cudaSetDevice(gdevs[0].first), "Failed to set QP owner device");

    if (oobdev.size() > 0 && oobdev[0] != "") {
        if (netif_get_addr(oobdev[0].c_str(), AF_INET, &oob_saddr, &oob_netmask) != 0) {
            throw std::invalid_argument("Failed to get IPv4 address for GPUNETIO OOB interface '" +
                                        oobdev[0] + "'");
        }
        struct sockaddr_in *addr_in = (struct sockaddr_in *)&oob_saddr;
        memcpy(ipv4_addr, (uint8_t *)&(addr_in->sin_addr.s_addr), 4);
        NIXL_DEBUG << "Eth IP address " << static_cast<unsigned>(ipv4_addr[0]) << " "
                   << static_cast<unsigned>(ipv4_addr[1]) << " "
                   << static_cast<unsigned>(ipv4_addr[2]) << " "
                   << static_cast<unsigned>(ipv4_addr[3]) << " " << "ifface " << oobdev[0].c_str();
    } else {
        result = doca_devinfo_get_ipv4_addr(
            doca_dev_as_devinfo(ddev), (uint8_t *)ipv4_addr, DOCA_DEVINFO_IPV4_ADDR_SIZE);
        if (result != DOCA_SUCCESS) {
            throw std::invalid_argument(
                "Failed to determine the GPUNETIO IPv4 address; set oob_interface explicitly "
                "when using a bonded network device");
        }
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

    *reinterpret_cast<volatile uint32_t *>(wait_exit_cpu) = 0;

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

    if (progressThreadStart() != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to start GPUNETIO connection thread");
    }
}

nixl_mem_list_t
nixlDocaEngine::getSupportedMems() const {
    return {DRAM_SEG, VRAM_SEG};
}

nixlDocaEngine::~nixlDocaEngine() {
    doca_error_t result;
    ScopedCudaDevice deviceGuard(gdevs[0].first);

    NIXL_DEBUG << "Before progressThreadStop ";
    progressThreadStop();

    *reinterpret_cast<volatile uint32_t *>(wait_exit_cpu) = 1;
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

    for (auto &entry : qpMap) {
        delete entry.second;
    }
    qpMap.clear();

    result = doca_dev_close(ddev);
    if (result != DOCA_SUCCESS)
        NIXL_ERROR << "Failed to close DOCA device " << doca_error_get_descr(result);

    for (auto &item : gdevs) {
        result = doca_gpu_destroy(item.second);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to close DOCA GPU device " << doca_error_get_descr(result);
        }
    }
}

/****************************************
 * DOCA request management
 *****************************************/

nixl_status_t
nixlDocaEngine::nixlDocaInitNotif(const std::string &remote_agent, doca_dev *dev, doca_gpu *gpu) {
    ScopedCudaDevice deviceGuard(gdevs[0].first);

    std::lock_guard<std::mutex> lock(notifLock);
    // Same peer can be server or client
    if (notifMap.find(remote_agent) != notifMap.end()) {
        NIXL_INFO << "nixlDocaInitNotif already found " << remote_agent << std::endl;
        return NIXL_SUCCESS;
    }

    auto notif = std::make_unique<nixlDocaNotif>();

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
        free(notif->send_addr);
        return NIXL_ERR_BACKEND;
    }

    notif->recv_addr = (uint8_t *)calloc(notif->elems_size * notif->elems_num, sizeof(uint8_t));
    if (notif->recv_addr == nullptr) {
        NIXL_ERROR << "Can't alloc memory for send notif";
        notif->send_mr.reset();
        free(notif->send_addr);
        return NIXL_ERR_BACKEND;
    }
    memset(notif->recv_addr, 0, notif->elems_size * notif->elems_num);

    try {
        notif->recv_mr = std::make_unique<nixl::doca::verbs::mr>(
            gpu, (void *)notif->recv_addr, notif->elems_num, notif->elems_size, pd);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        notif->send_mr.reset();
        free(notif->send_addr);
        free(notif->recv_addr);
        return NIXL_ERR_BACKEND;
    }

    notif->send_pi = 0;
    notif->recv_pi = 0;

    // Ensure notif list is not added twice for the same peer
    auto *notif_ptr = notif.get();
    notifMap[remote_agent] = notif.release();
    doca_gpu_dev_verbs_qp *notif_qp_gpu;
    {
        std::lock_guard<std::mutex> qp_lock(qpLock);
        auto qp = qpMap.find(remote_agent);
        if (qp == qpMap.end()) {
            nixlDocaDestroyNotif(gpu, notifMap.extract(remote_agent).mapped());
            return NIXL_ERR_INVALID_PARAM;
        }
        notif_qp_gpu = qp->second->qp_notif->get_qp_gpu_dev();
    }
    ((volatile struct docaNotif *)notif_fill_cpu)->msg_buf = (uintptr_t)notif_ptr->recv_addr;
    ((volatile struct docaNotif *)notif_fill_cpu)->msg_lkey = notif_ptr->recv_mr->get_lkey();
    ((volatile struct docaNotif *)notif_fill_cpu)->msg_size = notif_ptr->elems_size;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    ((volatile struct docaNotif *)notif_fill_cpu)->qp_gpu = notif_qp_gpu;
    while (((volatile struct docaNotif *)notif_fill_cpu)->qp_gpu != nullptr)
        ;

    NIXL_INFO << "nixlDocaInitNotif added new qp for " << remote_agent << std::endl;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::nixlDocaDestroyNotif(doca_gpu *gpu, struct nixlDocaNotif *notif) {
    notif->send_mr.reset();
    notif->recv_mr.reset();
    free(notif->send_addr);
    free(notif->recv_addr);
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

    if (setsockopt(oob_sock_server, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable))) {
        NIXL_ERROR << "Error setting socket options";
        close(oob_sock_server);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (oobdev.size() > 0 && oobdev[0] != "") {
        struct sockaddr_in *addr_in = (struct sockaddr_in *)&oob_saddr;
        /* Bind to the set port and IP: */
        addr_in->sin_port = htons(local_port);
        if (bind(oob_sock_server, (struct sockaddr *)addr_in, sizeof(struct sockaddr_in)) < 0) {
            NIXL_ERROR << "Couldn't bind to the port " << local_port;
            close(oob_sock_server);
            return NIXL_ERR_NOT_SUPPORTED;
        }
    } else {
        /* Set port and IP: */
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(local_port);
        server_addr.sin_addr.s_addr = INADDR_ANY; /* listen on any interface */

        /* Bind to the set port and IP: */
        if (bind(oob_sock_server, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            NIXL_ERROR << "Couldn't bind to the port " << local_port;
            close(oob_sock_server);
            return NIXL_ERR_NOT_SUPPORTED;
        }
    }

    NIXL_INFO << "Done with binding";

    /* Listen for clients: */
    if (listen(oob_sock_server, SOMAXCONN) < 0) {
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
        close(oob_sock_server);
        free((void *)pthrStop);
        pthrStop = nullptr;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

void
nixlDocaEngine::progressThreadStop() {
    int fake_sock_fd = -1;
    std::stringstream ss;

    ACCESS_ONCE(*pthrStop) = 1;
    ss << (int)ipv4_addr[0] << "." << (int)ipv4_addr[1] << "." << (int)ipv4_addr[2] << "."
       << (int)ipv4_addr[3];
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (oob_connection_client_setup(ss.str().c_str(), &fake_sock_fd, local_port) < 0) {
        shutdown(oob_sock_server, SHUT_RDWR);
    }
    // pthr.join();
    pthread_join(server_thread_id, nullptr);
    close(oob_sock_server);
    if (fake_sock_fd >= 0) {
        close(fake_sock_fd);
    }
    free((void *)pthrStop);
    pthrStop = nullptr;
}

uint32_t
nixlDocaEngine::getGpuCudaId() {
    return gdevs[0].first;
}

nixl_status_t
nixlDocaEngine::addRdmaQp(const std::string &remote_agent) {
    ScopedCudaDevice deviceGuard(gdevs[0].first);

    std::lock_guard<std::mutex> lock(qpLock);

    NIXL_DEBUG << "addRdmaQp for " << remote_agent << std::endl;

    // if client or server already created this QP, no need to re-create
    if (qpMap.find(remote_agent) != qpMap.end()) {
        return NIXL_IN_PROG;
    }

    NIXL_DEBUG << "DOCA addRdmaQp for remote " << remote_agent << std::endl;

    auto rdma_qp = std::make_unique<nixlDocaRdmaQp>();

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

    qpMap[remote_agent] = rdma_qp.release();

    NIXL_DEBUG << "DOCA addRdmaQp new QP added for " << remote_agent;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::connectClientRdmaQp(int oob_sock_client, const std::string &remote_agent) {
    doca_error_t result;
    struct nixlDocaRdmaQp *rdma_qp;
    uint32_t remote_qpn_data;
    uint32_t remote_qpn_notif;
    doca_verbs_gid remote_gid{};
    uint32_t remote_lid;
    uint32_t lack = 0, rack = 1;

    {
        std::lock_guard<std::mutex> lock(qpLock);
        auto qp = qpMap.find(remote_agent);
        if (qp == qpMap.end()) {
            NIXL_ERROR << "Can't find QP for remote agent " << remote_agent;
            return NIXL_ERR_INVALID_PARAM;
        }
        rdma_qp = qp->second;
    }

    NIXL_DEBUG << "connectClientRdmaQp: Send to server data qp connection details";
    // Data QP
    if (!sendAll(oob_sock_client, &rdma_qp->qpn_data, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    if (!sendAll(oob_sock_client, &rdma_qp->qpn_notif, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (!sendAll(oob_sock_client, &gid.raw, sizeof(gid.raw))) {
        NIXL_ERROR << "Failed to send local GID raw address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (!sendAll(oob_sock_client, &lid, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to send LID address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Data QP
    NIXL_DEBUG << "connectClientRdmaQp: Receive client remote data qp connection details";
    if (!recvAll(oob_sock_client, &remote_qpn_data, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    NIXL_INFO << "Receive remote notif qp connection details";
    if (!recvAll(oob_sock_client, &remote_qpn_notif, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (!recvAll(oob_sock_client, &remote_gid.raw, sizeof(gid.raw))) {
        NIXL_ERROR << "Failed to receive remote GID raw address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (!recvAll(oob_sock_client, &remote_lid, sizeof(uint32_t))) {
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

    rdma_qp->rqpn_data = remote_qpn_data;
    rdma_qp->rqpn_notif = remote_qpn_notif;
    rdma_qp->remote_gid = remote_gid;
    rdma_qp->remote_lid = remote_lid;

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- data";
    result =
        connect_verbs_qp(this, rdma_qp->qp_data->get_qp(), remote_qpn_data, remote_gid, remote_lid);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function connect_verbs_qp data failed " << doca_error_get_descr(result);
        connectLock.unlock();
        return NIXL_ERR_BACKEND;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- notif";
    result = connect_verbs_qp(
        this, rdma_qp->qp_notif->get_qp(), remote_qpn_notif, remote_gid, remote_lid);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function connect_verbs_qp notif failed " << doca_error_get_descr(result);
        connectLock.unlock();
        return NIXL_ERR_BACKEND;
    }

sync:
    // Record the QP transition before the final control ACK. If that ACK fails,
    // a retry must reuse the RTS QP rather than trying to transition it again.
    // remoteConnMap is published only after the ACK succeeds.
    connMap[remote_agent] = 1;
    connectLock.unlock();
    NIXL_DEBUG << "Client recv lack";
    if (!recvAll(oob_sock_client, &lack, sizeof(uint32_t))) {
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
    if (!sendAll(oob_sock_client, &rack, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::recvRemoteAgentName(int oob_sock_client, std::string &remote_agent) {
    size_t msg_size;

    // Msg
    if (!recvAll(oob_sock_client, &msg_size, sizeof(size_t))) {
        NIXL_ERROR << "Failed to recv msg details";
        return NIXL_ERR_BACKEND;
    }

    if (msg_size == 0) {
        NIXL_ERROR << "recvRemoteAgentName received msg size 0";
        return NIXL_ERR_BACKEND;
    }

    remote_agent.resize(msg_size);

    if (!recvAll(oob_sock_client, remote_agent.data(), msg_size)) {
        NIXL_ERROR << "Failed to recv msg details";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::sendLocalAgentName(int oob_sock_client) {
    size_t agent_size = localAgent.size();

    if (!sendAll(oob_sock_client, &agent_size, sizeof(size_t))) {
        NIXL_ERROR << "Failed to send connection details";
        return NIXL_ERR_BACKEND;
    }

    if (!sendAll(oob_sock_client, localAgent.c_str(), localAgent.size())) {
        NIXL_ERROR << "Failed to send connection details";
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << " sendLocalAgentName localAgent " << localAgent << std::endl;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::connectServerRdmaQp(int oob_sock_client, const std::string &remote_agent) {
    doca_error_t result;
    struct nixlDocaRdmaQp *rdma_qp;
    uint32_t remote_qpn_data;
    uint32_t remote_qpn_notif;
    doca_verbs_gid remote_gid{};
    uint32_t remote_lid;
    uint32_t lack = 0, rack = 1;

    {
        std::lock_guard<std::mutex> lock(qpLock);
        auto qp = qpMap.find(remote_agent);
        if (qp == qpMap.end()) {
            NIXL_ERROR << "Can't find QP for remote agent " << remote_agent;
            return NIXL_ERR_INVALID_PARAM;
        }
        rdma_qp = qp->second;
    }

    NIXL_DEBUG << "DOCA connectServerRdmaQp for agent " << remote_agent.c_str();

    // Data QP
    NIXL_DEBUG << "Server Receive client remote data qp connection details";
    if (!recvAll(oob_sock_client, &remote_qpn_data, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    NIXL_DEBUG << "Server Receive remote notif qp connection details";
    if (!recvAll(oob_sock_client, &remote_qpn_notif, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to receive remote connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (!recvAll(oob_sock_client, &remote_gid.raw, sizeof(gid.raw))) {
        NIXL_ERROR << "Failed to receive remote GID raw address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (!recvAll(oob_sock_client, &remote_lid, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to receive remote GID address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Data QP
    NIXL_DEBUG << "Server Send remote notif qp connection details";
    if (!sendAll(oob_sock_client, &rdma_qp->qpn_data, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    // Notif QP
    NIXL_DEBUG << "Server Send remote notif qp connection details";
    if (!sendAll(oob_sock_client, &rdma_qp->qpn_notif, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    if (!sendAll(oob_sock_client, &gid.raw, sizeof(gid.raw))) {
        NIXL_ERROR << "Failed to send local GID raw address";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Server Send remote notif qp connection details 4";
    if (!sendAll(oob_sock_client, &lid, sizeof(uint32_t))) {
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

    rdma_qp->rqpn_data = remote_qpn_data;
    rdma_qp->rqpn_notif = remote_qpn_notif;
    rdma_qp->remote_gid = remote_gid;
    rdma_qp->remote_lid = remote_lid;

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- data";
    result =
        connect_verbs_qp(this, rdma_qp->qp_data->get_qp(), remote_qpn_data, remote_gid, remote_lid);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function connect_verbs_qp data failed " << doca_error_get_descr(result);
        connectLock.unlock();
        return NIXL_ERR_BACKEND;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- notif";
    result = connect_verbs_qp(
        this, rdma_qp->qp_notif->get_qp(), remote_qpn_notif, remote_gid, remote_lid);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Function connect_verbs_qp notif failed " << doca_error_get_descr(result);
        connectLock.unlock();
        return NIXL_ERR_BACKEND;
    }

    connMap[remote_agent] = 1;

sync:

    connectLock.unlock();

    NIXL_DEBUG << "Server send rack " << rack;
    if (!sendAll(oob_sock_client, &rack, sizeof(uint32_t))) {
        NIXL_ERROR << "Failed to send connection details";
        result = DOCA_ERROR_CONNECTION_ABORTED;
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Server recv lack";
    if (!recvAll(oob_sock_client, &lack, sizeof(uint32_t))) {
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
    str = formatGpunetioOobEndpoint(ss.str(), local_port);
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
    {
        std::lock_guard<std::mutex> lock(remoteConnLock);
        if (remoteConnMap.find(remote_agent) != remoteConnMap.end()) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    GpunetioOobEndpoint endpoint;
    try {
        endpoint = parseGpunetioOobEndpoint(remote_conn_info);
    }
    catch (const std::invalid_argument &error) {
        NIXL_ERROR << error.what();
        return NIXL_ERR_INVALID_PARAM;
    }

    int ret = oob_connection_client_setup(endpoint.ipv4.c_str(), &oob_sock_client, endpoint.port);
    if (ret < 0) {
        NIXL_ERROR << "Can't connect to server " << ret;
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << "loadRemoteConnInfo calling addRdmaQp for " << remote_agent.c_str();
    nixl_status_t status = sendLocalAgentName(oob_sock_client);
    if (status != NIXL_SUCCESS) {
        close(oob_sock_client);
        return status;
    }
    status = addRdmaQp(remote_agent);
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        close(oob_sock_client);
        return status;
    }
    status = nixlDocaInitNotif(remote_agent, ddev, gdevs[0].second);
    if (status != NIXL_SUCCESS) {
        close(oob_sock_client);
        return status;
    }
    status = connectClientRdmaQp(oob_sock_client, remote_agent);
    if (status != NIXL_SUCCESS) {
        close(oob_sock_client);
        return status;
    }

    conn.remoteAgent = remote_agent;
    conn.connected = true;
    // if client or server already created this QP, no need to re-create
    {
        std::lock_guard<std::mutex> lock(remoteConnLock);
        if (remoteConnMap.find(remote_agent) == remoteConnMap.end()) {
            remoteConnMap[remote_agent] = conn;
            NIXL_INFO << "remoteConnMap extended with remote agent " << remote_agent << std::endl;
        }
    }

    NIXL_INFO << "DOCA loadRemoteConnInfo connected agent " << remote_agent;

    close(oob_sock_client);

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
 *****************************************/
nixl_status_t
nixlDocaEngine::registerMem(const nixlBlobDesc &mem,
                            const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) {
    auto priv = std::make_unique<nixlDocaPrivateMetadata>();
    std::stringstream ss;

    auto it = std::find_if(gdevs.begin(), gdevs.end(), [&mem](std::pair<uint32_t, doca_gpu *> &x) {
        return x.first == mem.devId;
    });
    if (it == gdevs.end()) {
        int device_count = 0;
        if (nixl_mem != VRAM_SEG || cudaGetDeviceCount(&device_count) != cudaSuccess ||
            mem.devId >= static_cast<uint64_t>(device_count)) {
            NIXL_ERROR << "Can't register memory for unknown device " << mem.devId;
            return NIXL_ERR_INVALID_PARAM;
        }
    }
    doca_gpu *memory_gpu = it == gdevs.end() ? nullptr : it->second;

    try {
        if (nixl_mem == VRAM_SEG) {
            ScopedCudaDevice deviceGuard(static_cast<int>(mem.devId));
            priv->mr = std::make_unique<nixl::doca::verbs::mr>(
                memory_gpu, (void *)mem.addr, 1, (size_t)mem.len, pd);
        } else {
            priv->mr = std::make_unique<nixl::doca::verbs::mr>(
                memory_gpu, (void *)mem.addr, 1, (size_t)mem.len, pd);
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }

    priv->devId = mem.devId;
    ss << (uint32_t)priv->mr->get_rkey() << info_delimiter << ((uintptr_t)priv->mr->get_addr())
       << info_delimiter << ((size_t)priv->mr->get_tot_size());
    priv->remoteMrStr = ss.str();

    out = priv.release();

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
    auto md = std::make_unique<nixlDocaPublicMetadata>();
    {
        std::lock_guard<std::mutex> lock(remoteConnLock);
        auto search = remoteConnMap.find(remote_agent);
        if (search == remoteConnMap.end()) {
            NIXL_ERROR << "err: remote connection not found remote_agent " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }
        conn = search->second;
    }

    // directly copy underlying conn struct
    md->conn = conn;

    std::stringstream ss(input.metaInfo.data());
    while (std::getline(ss, token, info_delimiter))
        tokens.push_back(token);

    if (tokens.size() != 3) {
        return NIXL_ERR_INVALID_PARAM;
    }

    uint32_t rkey;
    uintptr_t addr;
    size_t tot_size;
    try {
        size_t parsed = 0;
        const auto parsed_rkey = std::stoull(tokens[0], &parsed);
        if (parsed != tokens[0].size() || parsed_rkey > UINT32_MAX) {
            return NIXL_ERR_INVALID_PARAM;
        }
        rkey = static_cast<uint32_t>(parsed_rkey);
        addr = static_cast<uintptr_t>(std::stoull(tokens[1], &parsed));
        if (parsed != tokens[1].size()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        tot_size = static_cast<size_t>(std::stoull(tokens[2], &parsed));
        if (parsed != tokens[2].size()) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }
    catch (const std::exception &) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Empty mmap, filled with imported data
    try {
        md->mr = std::make_unique<nixl::doca::verbs::mr>((void *)addr, tot_size, rkey);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }

    output = md.release();

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::unloadMD(nixlBackendMD *input) {
    delete static_cast<nixlDocaPublicMetadata *>(input);
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
    nixlDocaBckndReq *treq;
    nixlDocaPrivateMetadata *lmd;
    nixlDocaPublicMetadata *rmd;
    uint32_t lcnt = (uint32_t)local.descCount();
    uint32_t rcnt = (uint32_t)remote.descCount();
    uint32_t stream_id;
    struct nixlDocaRdmaQp *rdma_qp;
    uintptr_t notif_addr;
    bool peer_memory = false;

    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (lcnt != rcnt || lcnt == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    for (uint32_t idx = 0; idx < lcnt; ++idx) {
        if (local[idx].len != remote[idx].len) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    for (uint32_t idx = 0; idx < lcnt; idx++) {
        lmd = (nixlDocaPrivateMetadata *)local[idx].metadataP;
        peer_memory |= lmd->devId != gdevs[0].first;
    }
    if (peer_memory && opt_args && !opt_args->customParam.empty()) {
        NIXL_ERROR << "Attached CUDA streams are not supported for peer-GPU payload memory";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    {
        std::lock_guard<std::mutex> lock(qpLock);
        auto search = qpMap.find(remote_agent);
        if (search == qpMap.end()) {
            NIXL_ERROR << "Can't find remote_agent " << remote_agent;
            return NIXL_ERR_INVALID_PARAM;
        }
        rdma_qp = search->second;
    }

    treq = new nixlDocaBckndReq;
    auto abandon_request = [&]() {
        for (uint32_t reserved_pos : treq->positions) {
            xferReqReserved[reserved_pos].store(false, std::memory_order_release);
        }
        delete treq;
    };

    if (opt_args->customParam.empty()) {
        stream_id = (xferStream.fetch_add(1) & (nstreams - 1));
        treq->stream = post_stream[stream_id];
    } else {
        treq->stream = (cudaStream_t) * ((uintptr_t *)opt_args->customParam.data());
    }

    auto reserve_position = [&]() -> bool {
        pos = xferRingPos.fetch_add(1) & DOCA_XFER_REQ_MASK;
        bool expected = false;
        if (!xferReqReserved[pos].compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            NIXL_ERROR << "GPUNETIO transfer ring exhausted at position " << pos;
            return false;
        }
        treq->positions.push_back(pos);
        return true;
    };
    treq->positions.reserve((lcnt + DOCA_XFER_REQ_SIZE - 1) / DOCA_XFER_REQ_SIZE);
    if (!reserve_position()) {
        abandon_request();
        return NIXL_ERR_BACKEND;
    }

    uint32_t desc_offset = 0;
    do {
        // Build in cacheable CPU memory, then publish the request with one
        // sequential copy to the GPU-mapped ring.
        docaXferReqGpu staged_req{};
        staged_req.has_notif_msg_idx = DOCA_NOTIF_NULL;
        while (desc_offset < lcnt && staged_req.num < DOCA_XFER_REQ_SIZE) {
            const uint32_t idx = staged_req.num;
            const uint32_t desc_idx = desc_offset;
            const size_t lsize = local[desc_idx].len;
            const size_t rsize = remote[desc_idx].len;
            lmd = (nixlDocaPrivateMetadata *)local[desc_idx].metadataP;
            rmd = (nixlDocaPublicMetadata *)remote[desc_idx].metadataP;

            staged_req.lbuf[idx] = local[desc_idx].addr;
            staged_req.lkey[idx] = lmd->mr->get_lkey();
            staged_req.rbuf[idx] = remote[desc_idx].addr;
            staged_req.rkey[idx] = rmd->mr->get_rkey();
            staged_req.size[idx] = lsize;
            staged_req.num_sge[idx] = 1;
            staged_req.num++;

            // Keep sparse local rows direct: adjacent remote ranges can share
            // one two-SGE RDMA WRITE WQE without a gather buffer. Preserve the
            // final descriptor as the protocol ordering boundary.
            if (operation == NIXL_WRITE && desc_idx + 1 < lcnt - 1) {
                auto *next_lmd =
                    static_cast<nixlDocaPrivateMetadata *>(local[desc_idx + 1].metadataP);
                auto *next_rmd =
                    static_cast<nixlDocaPublicMetadata *>(remote[desc_idx + 1].metadataP);
                const size_t next_lsize = local[desc_idx + 1].len;
                const size_t next_rsize = remote[desc_idx + 1].len;
                const uintptr_t current_remote = remote[desc_idx].addr;
                const uintptr_t next_remote = remote[desc_idx + 1].addr;
                const bool contiguous_remote =
                    next_remote >= current_remote && next_remote - current_remote == rsize;
                if (next_lsize == next_rsize && contiguous_remote &&
                    next_rmd->mr->get_rkey() == staged_req.rkey[idx]) {
                    staged_req.lbuf2[idx] = local[desc_idx + 1].addr;
                    staged_req.lkey2[idx] = next_lmd->mr->get_lkey();
                    staged_req.size2[idx] = next_lsize;
                    staged_req.num_sge[idx] = 2;
                    ++desc_offset;
                }
            }
            ++desc_offset;
        }

        staged_req.last_rsvd = last_rsvd_flags;
        staged_req.last_posted = last_posted_flags;
        staged_req.qp_data = rdma_qp->qp_data->get_qp_gpu_dev();
        staged_req.qp_notif = rdma_qp->qp_notif->get_qp_gpu_dev();
        memcpy(&xferReqRingCpu[pos], &staged_req, sizeof(staged_req));

        if (desc_offset < lcnt) {
            if (!reserve_position()) {
                abandon_request();
                return NIXL_ERR_BACKEND;
            }
        }
    } while (desc_offset < lcnt);

    const uint32_t final_pos = treq->positions.back();

    if (opt_args && opt_args->hasNotif) {
        struct nixlDocaNotif *notif;

        {
            std::lock_guard<std::mutex> lock(notifLock);
            auto search = notifMap.find(remote_agent);
            if (search == notifMap.end()) {
                NIXL_ERROR << "Can't find notif for remote_agent " << remote_agent;
                abandon_request();
                return NIXL_ERR_INVALID_PARAM;
            }
            notif = search->second;
        }

        // Check notifMsg size
        std::string newMsg = msg_tag_start + std::to_string(opt_args->notifMsg.size()) +
            msg_tag_end + opt_args->notifMsg;

        auto &final_request = xferReqRingCpu[final_pos];
        final_request.has_notif_msg_idx = (notif->send_pi.fetch_add(1) & (notif->elems_num - 1));
        notif_addr =
            (uintptr_t)(notif->send_addr + (final_request.has_notif_msg_idx * notif->elems_size));
        final_request.msg_sz = newMsg.size();
        final_request.lbuf_notif = notif_addr;
        final_request.lkey_notif = notif->send_mr->get_lkey();

        memcpy((void *)notif_addr, newMsg.c_str(), newMsg.size());

        NIXL_INFO << "DOCA prepXfer with notif to " << remote_agent << " at "
                  << final_request.has_notif_msg_idx << " msg " << newMsg << " to " << remote_agent;

    } else {
        xferReqRingCpu[final_pos].has_notif_msg_idx = DOCA_NOTIF_NULL;
    }

    NIXL_INFO << "DOCA REQUEST with " << treq->positions.size() << " ring positions, first "
              << treq->positions.front() << ", last " << final_pos << ", stream " << stream_id
              << std::endl;

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
    ScopedCudaDevice deviceGuard(gdevs[0].first);

    for (uint32_t idx : treq->positions) {
        doca_error_t result;
        xferReqRingCpu[idx].id = (lastPostedReq.fetch_add(1) & (DOCA_MAX_COMPLETION_INFLIGHT_MASK));
        completion_list_cpu[xferReqRingCpu[idx].id].xferReqRingGpu = xferReqRingGpu + idx;
        completion_list_cpu[xferReqRingCpu[idx].id].completed = 0;

        switch (operation) {
        case NIXL_READ:
            result =
                doca_kernel_read(treq->stream, xferReqRingCpu[idx].qp_data, xferReqRingGpu, idx);
            break;
        case NIXL_WRITE:
            result =
                doca_kernel_write(treq->stream, xferReqRingCpu[idx].qp_data, xferReqRingGpu, idx);
            break;
        default:
            return NIXL_ERR_INVALID_PARAM;
        }
        if (result != DOCA_SUCCESS) {
            *reinterpret_cast<volatile uint32_t *>(wait_exit_cpu) = 1;
            return NIXL_ERR_BACKEND;
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlDocaEngine::checkXfer(nixlBackendReqH *handle) const {
    if (*reinterpret_cast<volatile uint32_t *>(wait_exit_cpu) != 0) {
        return NIXL_ERR_BACKEND;
    }
    nixlDocaBckndReq *treq = (nixlDocaBckndReq *)handle;
    uint32_t completion_index;

    for (uint32_t idx : treq->positions) {
        completion_index = xferReqRingCpu[idx].id & (DOCA_MAX_COMPLETION_INFLIGHT_MASK);

        if (((volatile docaXferCompletion *)completion_list_cpu)[completion_index].completed != 1) {
            return NIXL_IN_PROG;
        }
    }
    for (uint32_t idx : treq->positions) {
        *((volatile uint8_t *)&xferReqRingCpu[idx].in_use) = 0;
        NIXL_INFO << "DOCA checkXfer pos " << idx << " COMPLETED!\n";
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::releaseReqH(nixlBackendReqH *handle) const {
    nixl_status_t status = checkXfer(handle);
    if (status == NIXL_IN_PROG) {
        return status;
    }
    auto *treq = static_cast<nixlDocaBckndReq *>(handle);
    for (uint32_t idx : treq->positions) {
        xferReqReserved[idx].store(false, std::memory_order_release);
    }
    delete treq;
    return status;
}

nixl_status_t
nixlDocaEngine::getNotifs(notif_list_t &notif_list) {
    if (*reinterpret_cast<volatile uint32_t *>(wait_exit_cpu) != 0) {
        return NIXL_ERR_BACKEND;
    }
    uint32_t recv_idx;
    std::string msg_src;
    char *addr;
    size_t position;

    // Lock required to prevent inconsistency if another notifyQp (new peer) is added
    // while getNotifs is running
    std::lock_guard<std::mutex> lock(notifLock);
    auto *progress = (volatile struct docaNotif *)notif_progress_cpu;
    if (progress->qp_gpu != nullptr) {
        return NIXL_SUCCESS;
    }

    if (!notifProgressPeer.empty()) {
        auto notif = notifMap.find(notifProgressPeer);
        if (notif != notifMap.end() && progress->msg_num > 0) {
            recv_idx = notif->second->recv_pi.load() & (DOCA_MAX_NOTIF_INFLIGHT - 1);
            addr = (char *)(notif->second->recv_addr + (recv_idx * notif->second->elems_size));
            msg_src.assign(addr, notif->second->elems_size);
            position = msg_src.find(msg_tag_start);
            size_t last = msg_src.find(msg_tag_end, msg_tag_start.size());
            size_t parsed = 0;
            unsigned long long msg_size = 0;
            bool valid = position == 0 && last != std::string::npos;
            if (valid) {
                std::string msg_sz =
                    msg_src.substr(msg_tag_start.size(), last - msg_tag_start.size());
                try {
                    msg_size = std::stoull(msg_sz, &parsed);
                }
                catch (const std::exception &) {
                    valid = false;
                }
                size_t payload_offset = last + msg_tag_end.size();
                valid = valid && parsed == msg_sz.size() && payload_offset <= msg_src.size() &&
                    msg_size <= msg_src.size() - payload_offset;
                if (valid) {
                    notif_list.emplace_back(
                        notif->first,
                        std::string(addr + payload_offset, addr + payload_offset + msg_size));
                }
            }
            memset(addr, 0, msg_tag_start.size());
            notif->second->recv_pi.fetch_add(1);
            notifProgressPeer.clear();
            if (!valid) {
                NIXL_ERROR << "getNotifs received malformed notification from " << notif->first;
                return NIXL_ERR_BACKEND;
            }
            return NIXL_SUCCESS;
        }
        notifProgressPeer.clear();
        return NIXL_SUCCESS;
    }

    const size_t peer_count = notifMap.size();
    for (size_t offset = 0; offset < peer_count; ++offset) {
        const size_t peer_index = (notifProgressCursor + offset) % peer_count;
        auto notif_iter = notifMap.begin();
        std::advance(notif_iter, peer_index);
        auto &notif = *notif_iter;
        recv_idx = notif.second->recv_pi.load() & (DOCA_MAX_NOTIF_INFLIGHT - 1);
        addr = (char *)(notif.second->recv_addr + (recv_idx * notif.second->elems_size));
        if (memcmp(addr, msg_tag_start.data(), msg_tag_start.size()) != 0) {
            continue;
        }
        progress->msg_num = 0;
        notifProgressPeer = notif.first;
        notifProgressCursor = (peer_index + 1) % peer_count;
        {
            std::lock_guard<std::mutex> qp_lock(qpLock);
            auto qp = qpMap.find(notif.first);
            if (qp == qpMap.end()) {
                notifProgressPeer.clear();
                return NIXL_ERR_BACKEND;
            }
            progress->qp_gpu = qp->second->qp_notif->get_qp_gpu_dev();
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        break;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    struct nixlDocaNotif *notif;
    doca_gpu_dev_verbs_qp *notif_qp_gpu;
    uint32_t buf_idx;
    uintptr_t msg_buf;

    {
        std::lock_guard<std::mutex> lock(notifLock);
        auto searchNotif = notifMap.find(remote_agent);
        if (searchNotif == notifMap.end()) {
            NIXL_ERROR << "genNotif: can't find notif for remote_agent " << remote_agent
                       << std::endl;
            return NIXL_ERR_INVALID_PARAM;
        }
        notif = searchNotif->second;
    }

    // 16B is uint16_t msg size
    if (msg.size() > DOCA_MAX_NOTIF_MESSAGE_SIZE - msg_tag_start.size() - msg_tag_end.size() - 16) {
        NIXL_ERROR << "Can't send notif as message size " << msg.size() << " is bigger than max "
                   << (DOCA_MAX_NOTIF_MESSAGE_SIZE - msg_tag_start.size() - msg_tag_end.size() -
                       16);
        return NIXL_ERR_INVALID_PARAM;
    }

    {
        std::lock_guard<std::mutex> lock(qpLock);
        auto searchQp = qpMap.find(remote_agent);
        if (searchQp == qpMap.end()) {
            NIXL_ERROR << "Can't find QP for remote_agent " << remote_agent;
            return NIXL_ERR_INVALID_PARAM;
        }
        notif_qp_gpu = searchQp->second->qp_notif->get_qp_gpu_dev();
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
    ((volatile struct docaNotif *)notif_send_cpu)->qp_gpu = notif_qp_gpu;
    while (((volatile struct docaNotif *)notif_send_cpu)->qp_gpu != nullptr)
        ;

    return NIXL_SUCCESS;
}
