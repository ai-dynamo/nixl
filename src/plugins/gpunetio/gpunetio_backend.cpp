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
#include "gpunetio_cuda_device_guard.h"
#include "gpunetio_completion_status.h"
#include "serdes/serdes.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cassert>
#include <charconv>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include "common/nixl_log.h"
#include <absl/strings/str_split.h>

const char info_delimiter = '-';

void
nixlDocaGpuCounterDeleter::operator()(uint64_t *counter) const {
    if (doca_gpu_mem_free(gpu, counter) != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to free striped WRITE completion counter";
    }
}


namespace {
template<typename unsignedType>
inline bool
parseUnsigned(std::string_view field, unsignedType &value) {
    if (field.empty()) {
        return false;
    }
    const auto result = std::from_chars(field.data(), field.data() + field.size(), value);
    return result.ec == std::errc{} && result.ptr == field.data() + field.size();
}

/** Parse the existing decimal rkey-address-size metadata without unchecked indexing. */
inline bool
parseRemoteMetadata(std::string_view input, uint32_t &rkey, uintptr_t &addr, size_t &size) {
    const size_t first = input.find(info_delimiter);
    if (first == std::string_view::npos) {
        return false;
    }
    const size_t second = input.find(info_delimiter, first + 1);
    if (second == std::string_view::npos) {
        return false;
    }
    uint32_t parsed_rkey;
    uintptr_t parsed_addr;
    size_t parsed_size;
    if (!parseUnsigned(input.substr(0, first), parsed_rkey) ||
        !parseUnsigned(input.substr(first + 1, second - first - 1), parsed_addr) ||
        !parseUnsigned(input.substr(second + 1), parsed_size)) {
        return false;
    }
    rkey = parsed_rkey;
    addr = parsed_addr;
    size = parsed_size;
    return true;
}

/** Decode a notification within its slot; neither header nor payload needs a trailing NUL. */
inline bool
parseNotification(std::string_view slot,
                  std::string_view start_tag,
                  std::string_view end_tag,
                  std::string_view &payload) {
    if (!slot.starts_with(start_tag)) {
        return false;
    }
    const size_t end = slot.find(end_tag, start_tag.size());
    if (end == std::string_view::npos) {
        return false;
    }
    size_t payload_size;
    if (!parseUnsigned(slot.substr(start_tag.size(), end - start_tag.size()), payload_size)) {
        return false;
    }
    const size_t offset = end + end_tag.size();
    if (payload_size > slot.size() - offset) {
        return false;
    }
    payload = slot.substr(offset, payload_size);
    return true;
}

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

void
rollbackOobDevice(doca_dev *device,
                  doca_verbs_pd *protection_domain,
                  doca_verbs_context *context) noexcept {
    // Discovery runs before GPU/AH allocation; retain dependents if a close fails.
    doca_error_t result = doca_dev_close(device);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to close DOCA device after OOB discovery failure: "
                   << doca_error_get_descr(result);
        return;
    }
    result = doca_verbs_pd_destroy(protection_domain);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to destroy DOCA PD after OOB discovery failure: "
                   << doca_error_get_descr(result);
        return;
    }
    result = doca_verbs_context_destroy(context);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to destroy DOCA context after OOB discovery failure: "
                   << doca_error_get_descr(result);
    }
}

constexpr uint32_t data_qp_protocol_magic = 0x4e585150;
constexpr uint32_t data_qp_protocol_version = 2;

uint32_t
parseDataQpCount(const std::string &value) {
    if (value.empty()) {
        return 1;
    }
    uint32_t count;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || count < 1 ||
        count > 4) {
        throw std::invalid_argument("data_qp_count must be an integer in [1, 4]");
    }
    return count;
}

bool
sendDataQps(int fd, const nixlDocaRdmaQp &peer, uint32_t count) {
    if (count > 1) {
        const uint32_t header[]{data_qp_protocol_magic, data_qp_protocol_version, count, count + 1};
        if (!sendAll(fd, header, sizeof(header))) {
            return false;
        }
    }
    return sendAll(fd, &peer.qpn_data, sizeof(peer.qpn_data)) &&
        (count == 1 || sendAll(fd, peer.qpn_write.data(), count * sizeof(uint32_t)));
}

bool
recvDataQps(int fd, uint32_t &data_qpn, std::vector<uint32_t> &write_qpns, uint32_t count) {
    if (count > 1) {
        uint32_t header[4]{};
        if (!recvAll(fd, header, sizeof(header)) || header[0] != data_qp_protocol_magic ||
            header[1] != data_qp_protocol_version || header[2] != count || header[3] != count + 1) {
            return false;
        }
        write_qpns.resize(count);
    }
    return recvAll(fd, &data_qpn, sizeof(data_qpn)) && data_qpn != data_qp_protocol_magic &&
        (count == 1 || recvAll(fd, write_qpns.data(), count * sizeof(uint32_t)));
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

    for (auto &reserved : xferReqReserved_) {
        reserved.store(false, std::memory_order_relaxed);
    }

    result = doca_log_backend_create_standard();
    if (result != DOCA_SUCCESS) {
        throw std::invalid_argument("Can't initialize doca log");
    }

    result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
    if (result != DOCA_SUCCESS) {
        throw std::invalid_argument("Can't initialize doca log");
    }

    result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_ERROR);
    if (result != DOCA_SUCCESS) {
        throw std::invalid_argument("Can't initialize doca log");
    }

    NIXL_INFO << "DOCA network devices ";
    // Temporary: will extend to more GPUs in a dedicated PR
    if (custom_params->count("network_devices") > 1) {
        throw std::invalid_argument("Only 1 network device is allowed");
    }

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
        if (custom_params->count("oob_interface") > 1) {
            throw std::invalid_argument("Only 1 oob interface is allowed");
        }

        oobdev = absl::StrSplit((*custom_params)["oob_interface"], " ");
        NIXL_INFO << "Using oob interface" << oobdev[0];
        NIXL_INFO << std::endl;
    }

    NIXL_INFO << "DOCA GPU devices: ";
    // Temporary: will extend to more GPUs in a dedicated PR
    if (custom_params->count("gpu_devices") > 1) {
        throw std::invalid_argument("Only 1 GPU device is allowed");
    }

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
    if (custom_params->count("cuda_streams") != 0 && (*custom_params)["cuda_streams"] != "") {
        nstreams = std::stoi((*custom_params)["cuda_streams"]);
    }
    if (nstreams == 0) {
        nstreams = DOCA_POST_STREAM_NUM;
    }

    NIXL_INFO << "CUDA streams used for pool mode: " << nstreams;

    local_port = parseGpunetioOobPort((*custom_params)["oob_port"]);
    dataQpCount_ = parseDataQpCount((*custom_params)["data_qp_count"]);
    uint32_t byte_striping = 0;
    const auto byte_option = custom_params->find("write_byte_striping");
    if (byte_option != custom_params->end() && !byte_option->second.empty() &&
        (!parseUnsigned(byte_option->second, byte_striping) || byte_striping > 1)) {
        throw std::invalid_argument("write_byte_striping must be 0 or 1");
    }
    writeByteStriping_ = byte_striping != 0;
    if (dataQpCount_ > 1 &&
        (nstreams < static_cast<int>(dataQpCount_) ||
         nstreams > static_cast<int>(DOCA_POST_STREAM_NUM))) {
        throw std::invalid_argument("striped WRITE requires data_qp_count <= cuda_streams <= 4");
    }
    NIXL_INFO << "WRITE data QPs per peer: " << dataQpCount_;
    NIXL_INFO << "OOB listen port: " << local_port;

    // Reject an unusable explicit interface before acquiring RDMA/GPU resources.
    const bool explicit_oob = !oobdev.empty() && !oobdev[0].empty();
    if (explicit_oob) {
        if (netif_get_addr(oobdev[0].c_str(), AF_INET, &oob_saddr, &oob_netmask) != 0) {
            throw std::invalid_argument("Failed to find an active IPv4 address for oob_interface " +
                                        oobdev[0]);
        }
        const auto *address = reinterpret_cast<const sockaddr_in *>(&oob_saddr);
        memcpy(ipv4_addr, &address->sin_addr.s_addr, sizeof(ipv4_addr));
    }

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
    if (pd == NULL) {
        throw std::invalid_argument("Failed to get ibv_pd");
    }

    result = doca_rdma_bridge_open_dev_from_pd(pd, &ddev);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create doca verbs pd: %s", doca_error_get_descr(result);
        throw std::invalid_argument("Failed to create doca verbs pd");
    }

    if (!explicit_oob) {
        result = doca_devinfo_get_ipv4_addr(
            doca_dev_as_devinfo(ddev), ipv4_addr, DOCA_DEVINFO_IPV4_ADDR_SIZE);
        if (result != DOCA_SUCCESS) {
            rollbackOobDevice(ddev, verbs_pd, verbs_context);
            throw std::invalid_argument(
                "Failed to determine the GPUNETIO IPv4 address; set oob_interface explicitly");
        }
    }

    ret = ibv_query_port(pd->context, 1, &port_attr);
    if (ret) {
        throw std::invalid_argument("Failed to query ibv port attributes");
    }

    gid_index = parseGidIndex((*custom_params)["gid_index"]);

    ret = ibv_query_gid(pd->context, 1, gid_index, &rgid);
    if (ret) {
        NIXL_ERROR << "Failed to query ibv gid attributes";
        throw std::invalid_argument("Failed to query ibv gid attributes");
    }
    memcpy(gid.raw, rgid.raw, DOCA_GID_BYTE_LENGTH);

    if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        result = create_verbs_ah_attr(
            verbs_context, gid_index, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH, &verbs_ah_attr);
        if (result != DOCA_SUCCESS) {
            throw std::invalid_argument("Failed to create doca verbs ah attributes");
        }

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
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to create DOCA GPU device " << doca_error_get_descr(result);
        }
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
    for (int i = 0; i < nstreams; i++) {
        nixlDocaEngineCheckCudaError(
            cudaStreamCreateWithFlags(&post_stream[i], cudaStreamNonBlocking),
            "Failed to create CUDA stream");
    }
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
    if (dataQpCount_ > 1) {
        doca_kernel_write_striped(0, nullptr, {}, dataQpCount_);
    }
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

    NIXL_DEBUG << "Before progressThreadStop ";
    progressThreadStop();

    *reinterpret_cast<volatile uint32_t *>(wait_exit_cpu) = 1;
    nixl::doca::cudaDeviceGuard cuda_device(gdevs[0].first);
    if (cuda_device.status() != cudaSuccess) {
        NIXL_ERROR << "Cannot select GPUNETIO engine device; retaining device resources";
        // Host progress is joined; do not destroy device-owned objects in another context.
        retainedQp_.release();
        return;
    }
    NIXL_DEBUG << "Before cudaStreamSynchronize ";
    nixlDocaEngineCheckCudaError(cudaStreamSynchronize(wait_stream), "stream synchronize");
    nixlDocaEngineCheckCudaError(cudaStreamDestroy(wait_stream), "stream destroy");

    for (int i = 0; i < nstreams; i++) {
        NIXL_DEBUG << "Before cudaStreamSynchronize post_stream " << i;
        nixlDocaEngineCheckCudaError(cudaStreamSynchronize(post_stream[i]), "stream synchronize");
        nixlDocaEngineCheckCudaError(cudaStreamDestroy(post_stream[i]), "stream destroy");
    }

    bool qps_closed = true;
    bool retained_closed = true;
    for (auto &qp : qpMap) {
        const bool data_closed = qp.second->closeDataQps();
        const bool notif_closed = qp.second->qp_notif->close();
        qps_closed = data_closed && notif_closed && qps_closed;
    }
    if (retainedQp_ != nullptr) {
        const bool data_closed = retainedQp_->closeDataQps();
        const bool notif_closed =
            retainedQp_->qp_notif == nullptr || retainedQp_->qp_notif->close();
        retained_closed = data_closed && notif_closed;
        qps_closed = retained_closed && qps_closed;
    }
    if (!qps_closed) {
        NIXL_ERROR << "GPUNETIO QP unexport failed; retaining dependent resources";
        if (!retained_closed) {
            retainedQp_.release();
        }
        return;
    }

    for (auto &qp : qpMap) {
        delete qp.second;
    }
    qpMap.clear();
    retainedQp_.reset();

    NIXL_DEBUG << "Before nixlDocaDestroyNotif ";
    for (auto notif : notifMap) {
        nixlDocaDestroyNotif(gdevs[0].second, notif.second);
    }
    notifMap.clear();

    doca_gpu_mem_free(gdevs[0].second, wait_exit_gpu);
    doca_gpu_mem_free(gdevs[0].second, xferReqRingGpu);
    doca_gpu_mem_free(gdevs[0].second, last_rsvd_flags);
    doca_gpu_mem_free(gdevs[0].second, last_posted_flags);
    doca_gpu_mem_free(gdevs[0].second, notif_fill_gpu);
    doca_gpu_mem_free(gdevs[0].second, notif_progress_gpu);
    doca_gpu_mem_free(gdevs[0].second, notif_send_gpu);
    doca_gpu_mem_free(gdevs[0].second, completion_list_gpu);

    result = doca_gpu_destroy(gdevs[0].second);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to close DOCA GPU device " << doca_error_get_descr(result);
    }
    gdevs[0].second = nullptr;

    result = doca_verbs_ah_attr_destroy(verbs_ah_attr);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to destroy DOCA verbs AH " << doca_error_get_descr(result);
    }
    verbs_ah_attr = nullptr;

    result = doca_dev_close(ddev);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to close DOCA device " << doca_error_get_descr(result);
    }
    ddev = nullptr;

    result = doca_verbs_pd_destroy(verbs_pd);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to destroy DOCA verbs PD " << doca_error_get_descr(result);
    }
    verbs_pd = nullptr;
    pd = nullptr;

    result = doca_verbs_context_destroy(verbs_context);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to destroy DOCA verbs context " << doca_error_get_descr(result);
    }
    verbs_context = nullptr;
}

/****************************************
 * DOCA request management
 *****************************************/

nixl_status_t
nixlDocaEngine::nixlDocaInitNotif(const std::string &remote_agent, doca_dev *dev, doca_gpu *gpu) {
    const nixl_status_t completion_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
    if (completion_status != NIXL_SUCCESS) {
        return completion_status;
    }

    std::lock_guard<std::mutex> lock(notifLock);
    // Same peer can be server or client
    if (notifMap.find(remote_agent) != notifMap.end()) {
        NIXL_INFO << "nixlDocaInitNotif already found " << remote_agent << std::endl;
        return NIXL_SUCCESS;
    }

    doca_gpu_dev_verbs_qp *notif_qp_gpu;
    {
        std::lock_guard<std::mutex> qp_lock(qpLock);
        auto qp = qpMap.find(remote_agent);
        if (qp == qpMap.end()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        notif_qp_gpu = qp->second->qp_notif->get_qp_gpu_dev();
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
            gpu, (void *)notif->send_addr, notif->elems_num, notif->elems_size, pd, false);
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
            gpu, (void *)notif->recv_addr, notif->elems_num, notif->elems_size, pd, false);
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
    const auto [notif_it, inserted] = notifMap.emplace(remote_agent, notif.get());
    if (!inserted) {
        return NIXL_SUCCESS;
    }
    notif.release();
    auto *notif_ptr = notif_it->second;
    ((volatile struct docaNotif *)notif_fill_cpu)->msg_buf = (uintptr_t)notif_ptr->recv_addr;
    ((volatile struct docaNotif *)notif_fill_cpu)->msg_lkey = notif_ptr->recv_mr->get_lkey();
    ((volatile struct docaNotif *)notif_fill_cpu)->msg_size = notif_ptr->elems_size;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    ((volatile struct docaNotif *)notif_fill_cpu)->qp_gpu = notif_qp_gpu;
    while (((volatile struct docaNotif *)notif_fill_cpu)->qp_gpu != nullptr) {
        const nixl_status_t wait_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
        if (wait_status != NIXL_SUCCESS) {
            return wait_status;
        }
    }

    NIXL_INFO << "nixlDocaInitNotif added new qp for " << remote_agent << std::endl;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::nixlDocaDestroyNotif(doca_gpu *gpu, struct nixlDocaNotif *notif) {
    if (notif == nullptr) {
        return NIXL_SUCCESS;
    }

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
        free((void *)pthrStop);
        pthrStop = nullptr;
        return NIXL_ERR_NOT_SUPPORTED;
    }
    NIXL_INFO << "DOCA Server socket created successfully";

    if (setsockopt(oob_sock_server, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable))) {
        NIXL_ERROR << "Error setting socket options";
        close(oob_sock_server);
        free((void *)pthrStop);
        pthrStop = nullptr;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (oobdev.size() > 0 && oobdev[0] != "") {
        struct sockaddr_in *addr_in = (struct sockaddr_in *)&oob_saddr;
        /* Bind to the set port and IP: */
        addr_in->sin_port = htons(local_port);
        if (bind(oob_sock_server, (struct sockaddr *)addr_in, sizeof(struct sockaddr_in)) < 0) {
            NIXL_ERROR << "Couldn't bind to the port " << local_port;
            close(oob_sock_server);
            free((void *)pthrStop);
            pthrStop = nullptr;
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
            free((void *)pthrStop);
            pthrStop = nullptr;
            return NIXL_ERR_NOT_SUPPORTED;
        }
    }

    NIXL_INFO << "Done with binding";

    /* Listen for clients: */
    if (listen(oob_sock_server, SOMAXCONN) < 0) {
        NIXL_ERROR << "Error while listening";
        close(oob_sock_server);
        free((void *)pthrStop);
        pthrStop = nullptr;
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
    serverThreadStarted = true;

    return NIXL_SUCCESS;
}

void
nixlDocaEngine::progressThreadStop() {
    if (!serverThreadStarted) {
        return;
    }

    ACCESS_ONCE(*pthrStop) = 1;
    std::atomic_thread_fence(std::memory_order_seq_cst);

    const int active_socket = activeOobSocket.exchange(-1);
    if (active_socket >= 0) {
        shutdown(active_socket, SHUT_RDWR);
    }
    shutdown(oob_sock_server, SHUT_RDWR);
    pthread_join(server_thread_id, nullptr);
    serverThreadStarted = false;
    close(oob_sock_server);
    free((void *)pthrStop);
    pthrStop = nullptr;
}

uint32_t
nixlDocaEngine::getGpuCudaId() {
    return gdevs[0].first;
}

nixl_status_t
nixlDocaEngine::addRdmaQp(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(qpLock);

    NIXL_DEBUG << "addRdmaQp for " << remote_agent << std::endl;

    // if client or server already created this QP, no need to re-create
    if (qpMap.find(remote_agent) != qpMap.end()) {
        return NIXL_IN_PROG;
    }
    if (retainedQp_ != nullptr) {
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "DOCA addRdmaQp for remote " << remote_agent << std::endl;

    nixl::doca::cudaDeviceGuard cuda_device(gdevs[0].first);
    if (cuda_device.status() != cudaSuccess) {
        NIXL_ERROR << "Failed to select CUDA device " << gdevs[0].first
                   << " for QP setup: " << cudaGetErrorString(cuda_device.status());
        return NIXL_ERR_BACKEND;
    }

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
        if (dataQpCount_ > 1) {
            for (uint32_t index = 0; index < dataQpCount_; ++index) {
                auto qp = std::make_unique<nixl::doca::verbs::qp>(
                    gdevs[0].second,
                    ddev,
                    verbs_context,
                    verbs_pd,
                    RDMA_SEND_QUEUE_SIZE,
                    RDMA_RECV_QUEUE_SIZE,
                    DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB);
                rdma_qp->qpn_write.push_back(doca_verbs_qp_get_qpn(qp->get_qp()));
                rdma_qp->qp_write.push_back(std::move(qp));
            }
            rdma_qp->writeProgrammed.resize(dataQpCount_, false);
            uint64_t *gpu_counter = nullptr;
            const auto result =
                doca_gpu_mem_alloc(gdevs[0].second,
                                   sizeof(uint64_t),
                                   64,
                                   DOCA_GPU_MEM_TYPE_GPU_CPU,
                                   reinterpret_cast<void **>(&gpu_counter),
                                   reinterpret_cast<void **>(&rdma_qp->writeCompletedCpu));
            if (result != DOCA_SUCCESS) {
                throw std::runtime_error("Failed to allocate striped WRITE completion counter");
            }
            rdma_qp->writeCompletedGpu.get_deleter().gpu = gdevs[0].second;
            rdma_qp->writeCompletedGpu.reset(gpu_counter);
            *rdma_qp->writeCompletedCpu = 0;
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        if (!rdma_qp->closeDataQps()) {
            retainedQp_ = std::move(rdma_qp);
        }
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
        if (!rdma_qp->closeDataQps()) {
            retainedQp_ = std::move(rdma_qp);
        }
        return NIXL_ERR_BACKEND;
    }

    rdma_qp->qpn_notif = doca_verbs_qp_get_qpn(rdma_qp->qp_notif->get_qp());

    const auto [qp_it, inserted] = qpMap.emplace(remote_agent, rdma_qp.get());
    if (!inserted) {
        return NIXL_IN_PROG;
    }
    rdma_qp.release();

    NIXL_DEBUG << "DOCA addRdmaQp new QP added for " << remote_agent;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::connectClientRdmaQp(int oob_sock_client, const std::string &remote_agent) {
    doca_error_t result;
    struct nixlDocaRdmaQp *rdma_qp;
    uint32_t lack = 0, rack = 1;
    uint32_t remote_qpn_data = 0, remote_qpn_notif = 0, remote_lid = 0;
    std::vector<uint32_t> remote_write_qpns;
    doca_verbs_gid remote_gid{};

    {
        std::lock_guard<std::mutex> lock(qpLock);
        auto qp = qpMap.find(remote_agent);
        if (qp == qpMap.end()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        rdma_qp = qp->second;
    }

    NIXL_DEBUG << "connectClientRdmaQp: Send to server data qp connection details";
    // Data QP
    if (!sendDataQps(oob_sock_client, *rdma_qp, dataQpCount_)) {
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
    if (!recvDataQps(oob_sock_client, remote_qpn_data, remote_write_qpns, dataQpCount_)) {
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
    std::unique_lock<std::mutex> lock(connectLock);
    if (rdma_qp->dataProgrammed || rdma_qp->notifProgrammed) {
        if (rdma_qp->rqpn_data != remote_qpn_data || rdma_qp->rqpn_write != remote_write_qpns ||
            rdma_qp->rqpn_notif != remote_qpn_notif || rdma_qp->remote_lid != remote_lid ||
            memcmp(rdma_qp->remote_gid.raw, remote_gid.raw, sizeof(remote_gid.raw)) != 0) {
            NIXL_ERROR << "Remote QP route changed during retry";
            return NIXL_ERR_BACKEND;
        }
    } else {
        rdma_qp->rqpn_data = remote_qpn_data;
        rdma_qp->rqpn_write = remote_write_qpns;
        rdma_qp->rqpn_notif = remote_qpn_notif;
        rdma_qp->remote_gid = remote_gid;
        rdma_qp->remote_lid = remote_lid;
    }
    if (connMap.find(remote_agent) != connMap.end()) {
        NIXL_INFO << "QP for " << remote_agent << " already connected" << std::endl;
        goto sync;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- data";
    if (!rdma_qp->dataProgrammed) {
        result = connect_verbs_qp(
            this, rdma_qp->qp_data->get_qp(), remote_qpn_data, remote_gid, remote_lid);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Function connect_verbs_qp data failed " << doca_error_get_descr(result);
            return NIXL_ERR_BACKEND;
        }
        rdma_qp->dataProgrammed = true;
    }

    for (size_t index = 0; index < rdma_qp->qp_write.size(); ++index) {
        if (!rdma_qp->writeProgrammed[index]) {
            result = connect_verbs_qp(this,
                                      rdma_qp->qp_write[index]->get_qp(),
                                      remote_write_qpns[index],
                                      remote_gid,
                                      remote_lid);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to connect striped WRITE QP " << index;
                return NIXL_ERR_BACKEND;
            }
            rdma_qp->writeProgrammed[index] = true;
        }
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- notif";
    if (!rdma_qp->notifProgrammed) {
        result = connect_verbs_qp(
            this, rdma_qp->qp_notif->get_qp(), remote_qpn_notif, remote_gid, remote_lid);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Function connect_verbs_qp notif failed " << doca_error_get_descr(result);
            return NIXL_ERR_BACKEND;
        }
        rdma_qp->notifProgrammed = true;
    }

    // QP programming is complete even if the final ACK exchange later fails.
    // A retry must skip another QP state transition and repeat only the handshake.
    connMap[remote_agent] = 1;

sync:
    lock.unlock();
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

    if (!isValidGpunetioAgentNameSize(msg_size)) {
        NIXL_ERROR << "recvRemoteAgentName received invalid msg size " << msg_size;
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

    if (!isValidGpunetioAgentNameSize(agent_size)) {
        NIXL_ERROR << "sendLocalAgentName has invalid msg size " << agent_size;
        return NIXL_ERR_INVALID_PARAM;
    }

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
    uint32_t lack = 0, rack = 1;
    uint32_t remote_qpn_data = 0, remote_qpn_notif = 0, remote_lid = 0;
    std::vector<uint32_t> remote_write_qpns;
    doca_verbs_gid remote_gid{};

    {
        std::lock_guard<std::mutex> lock(qpLock);
        auto qp = qpMap.find(remote_agent);
        if (qp == qpMap.end()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        rdma_qp = qp->second;
    }

    NIXL_DEBUG << "DOCA connectServerRdmaQp for agent " << remote_agent.c_str();

    // Data QP
    NIXL_DEBUG << "Server Receive client remote data qp connection details";
    if (!recvDataQps(oob_sock_client, remote_qpn_data, remote_write_qpns, dataQpCount_)) {
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
    if (!sendDataQps(oob_sock_client, *rdma_qp, dataQpCount_)) {
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
    std::unique_lock<std::mutex> lock(connectLock);
    if (rdma_qp->dataProgrammed || rdma_qp->notifProgrammed) {
        if (rdma_qp->rqpn_data != remote_qpn_data || rdma_qp->rqpn_write != remote_write_qpns ||
            rdma_qp->rqpn_notif != remote_qpn_notif || rdma_qp->remote_lid != remote_lid ||
            memcmp(rdma_qp->remote_gid.raw, remote_gid.raw, sizeof(remote_gid.raw)) != 0) {
            NIXL_ERROR << "Remote QP route changed during retry";
            return NIXL_ERR_BACKEND;
        }
    } else {
        rdma_qp->rqpn_data = remote_qpn_data;
        rdma_qp->rqpn_write = remote_write_qpns;
        rdma_qp->rqpn_notif = remote_qpn_notif;
        rdma_qp->remote_gid = remote_gid;
        rdma_qp->remote_lid = remote_lid;
    }
    if (connMap.find(remote_agent) != connMap.end()) {
        NIXL_DEBUG << "QP for " << remote_agent << " already connected";
        goto sync;
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- data";
    if (!rdma_qp->dataProgrammed) {
        result = connect_verbs_qp(
            this, rdma_qp->qp_data->get_qp(), remote_qpn_data, remote_gid, remote_lid);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Function connect_verbs_qp data failed " << doca_error_get_descr(result);
            return NIXL_ERR_BACKEND;
        }
        rdma_qp->dataProgrammed = true;
    }

    for (size_t index = 0; index < rdma_qp->qp_write.size(); ++index) {
        if (!rdma_qp->writeProgrammed[index]) {
            result = connect_verbs_qp(this,
                                      rdma_qp->qp_write[index]->get_qp(),
                                      remote_write_qpns[index],
                                      remote_gid,
                                      remote_lid);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to connect striped WRITE QP " << index;
                return NIXL_ERR_BACKEND;
            }
            rdma_qp->writeProgrammed[index] = true;
        }
    }

    /* Connect local rdma to the remote rdma */
    NIXL_DEBUG << "Connect DOCA RDMA to remote RDMA -- notif";
    if (!rdma_qp->notifProgrammed) {
        result = connect_verbs_qp(
            this, rdma_qp->qp_notif->get_qp(), remote_qpn_notif, remote_gid, remote_lid);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Function connect_verbs_qp notif failed " << doca_error_get_descr(result);
            return NIXL_ERR_BACKEND;
        }
        rdma_qp->notifProgrammed = true;
    }

    // QP programming is complete even if the final ACK exchange later fails.
    // A retry must skip another QP state transition and repeat only the handshake.
    connMap[remote_agent] = 1;

sync:
    lock.unlock();

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

    GpunetioOobEndpoint endpoint;
    try {
        endpoint = parseGpunetioOobEndpoint(remote_conn_info);
    }
    catch (const std::invalid_argument &error) {
        NIXL_ERROR << error.what();
        return NIXL_ERR_INVALID_PARAM;
    }

    // TODO: Connect part should be moved into connect() method
    nixlDocaConnection conn;
    conn.remoteAgent = remote_agent;
    conn.connected = false;
    {
        std::lock_guard<std::mutex> lock(remoteConnLock);
        if (!remoteConnMap.emplace(remote_agent, conn).second) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }
    auto clear_pending = [&]() {
        std::lock_guard<std::mutex> lock(remoteConnLock);
        auto pending = remoteConnMap.find(remote_agent);
        if (pending != remoteConnMap.end() && !pending->second.connected) {
            remoteConnMap.erase(pending);
        }
    };

    int ret = oob_connection_client_setup(endpoint.ipv4.c_str(), &oob_sock_client, endpoint.port);
    if (ret < 0) {
        NIXL_ERROR << "Can't connect to server " << ret;
        clear_pending();
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << "loadRemoteConnInfo calling addRdmaQp for " << remote_agent.c_str();
    nixl_status_t status = sendLocalAgentName(oob_sock_client);
    if (status == NIXL_SUCCESS) {
        status = addRdmaQp(remote_agent);
    }
    if (status == NIXL_IN_PROG) {
        status = NIXL_SUCCESS;
    }
    if (status == NIXL_SUCCESS) {
        status = nixlDocaInitNotif(remote_agent, ddev, gdevs[0].second);
    }
    if (status == NIXL_SUCCESS) {
        status = connectClientRdmaQp(oob_sock_client, remote_agent);
    }
    if (status != NIXL_SUCCESS) {
        close(oob_sock_client);
        clear_pending();
        return status;
    }

    conn.connected = true;
    {
        std::lock_guard<std::mutex> lock(remoteConnLock);
        remoteConnMap[remote_agent] = conn;
        NIXL_INFO << "remoteConnMap connected remote agent " << remote_agent << std::endl;
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

    // A foreign payload GPU has no engine-owned DOCA handle; use its PeerDirect MR.
    doca_gpu *memory_gpu = it == gdevs.end() ? nullptr : it->second;
    nixl::doca::cudaDeviceGuard memory_device(
        nixl_mem == VRAM_SEG ? static_cast<uint32_t>(mem.devId) : gdevs[0].first);
    if (memory_device.status() != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    try {
        priv->mr =
            std::make_unique<nixl::doca::verbs::mr>(memory_gpu,
                                                    (void *)mem.addr,
                                                    1,
                                                    (size_t)mem.len,
                                                    pd,
                                                    nixl_mem == VRAM_SEG && memory_gpu != nullptr);
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
    auto md = std::make_unique<nixlDocaPublicMetadata>();
    auto search = remoteConnMap.find(remote_agent);

    if (search == remoteConnMap.end()) {
        NIXL_ERROR << "err: remote connection not found remote_agent " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    conn = (nixlDocaConnection)search->second;

    // directly copy underlying conn struct
    md->conn = conn;

    uint32_t rkey;
    uintptr_t addr;
    size_t tot_size;
    if (!parseRemoteMetadata(input.metaInfo, rkey, addr, tot_size)) {
        NIXL_ERROR << "Malformed remote metadata from " << remote_agent;
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
    const uint32_t lcnt = (uint32_t)local.descCount();
    const uint32_t rcnt = (uint32_t)remote.descCount();
    uint32_t stream_id = DOCA_POST_STREAM_NUM;
    struct nixlDocaRdmaQp *rdma_qp;
    uintptr_t notif_addr;

    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (lcnt != rcnt || lcnt == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if ((lcnt + DOCA_XFER_REQ_SIZE - 1) / DOCA_XFER_REQ_SIZE > DOCA_XFER_REQ_MAX) {
        return NIXL_ERR_INVALID_PARAM;
    }

    for (uint32_t idx = 0; idx < lcnt; idx++) {
        if (local[idx].len != remote[idx].len) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    size_t write_bytes = 0;
    if (operation == NIXL_WRITE && writeByteStriping_ && dataQpCount_ > 1 &&
        (lcnt + DOCA_XFER_REQ_SIZE - 1) / DOCA_XFER_REQ_SIZE < dataQpCount_) {
        for (uint32_t idx = 0; idx < lcnt; ++idx) {
            if (local[idx].len > std::numeric_limits<size_t>::max() - write_bytes ||
                local[idx].addr > std::numeric_limits<uintptr_t>::max() - local[idx].len ||
                remote[idx].addr > std::numeric_limits<uintptr_t>::max() - remote[idx].len) {
                return NIXL_ERR_INVALID_PARAM;
            }
            write_bytes += local[idx].len;
        }
    }
    const size_t final_bytes = lcnt > 1 && write_bytes != 0 ? local[lcnt - 1].len : 0;
    const size_t striped_bytes = write_bytes - final_bytes;
    const bool byte_striping = striped_bytes >= 4 * 1024 * 1024;
    const size_t ordering_tail = byte_striping && lcnt == 1 ? sizeof(uint64_t) : 0;
    const size_t bulk_bytes = striped_bytes - ordering_tail;
    const size_t chunk_bytes_limit = byte_striping ?
        bulk_bytes / dataQpCount_ + (bulk_bytes % dataQpCount_ != 0) :
        std::numeric_limits<size_t>::max();

    if (operation == NIXL_WRITE && dataQpCount_ > 1 && opt_args != nullptr &&
        !opt_args->customParam.empty()) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    for (uint32_t idx = 0; idx < lcnt; idx++) {
        lmd = (nixlDocaPrivateMetadata *)local[idx].metadataP;
        if (lmd->devId != gdevs[0].first && opt_args && !opt_args->customParam.empty()) {
            NIXL_ERROR << "Attached CUDA streams are not supported for peer-GPU payloads";
            return NIXL_ERR_NOT_SUPPORTED;
        }
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
            xferReqReserved_[reserved_pos].store(false, std::memory_order_release);
        }
        delete treq;
    };

    if (opt_args == nullptr || opt_args->customParam.empty()) {
        stream_id = (xferStream.fetch_add(1) & (nstreams - 1));
        treq->stream = post_stream[stream_id];
    } else {
        if (opt_args->customParam.size() != sizeof(cudaStream_t)) {
            abandon_request();
            return NIXL_ERR_INVALID_PARAM;
        }
        std::memcpy(&treq->stream, opt_args->customParam.data(), sizeof(treq->stream));
    }

    auto reserve_position = [&]() -> bool {
        for (uint32_t attempt = 0; attempt < DOCA_XFER_REQ_MAX; ++attempt) {
            const uint32_t candidate = xferRingPos.fetch_add(1) & DOCA_XFER_REQ_MASK;
            bool expected = false;
            if (xferReqReserved_[candidate].compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel)) {
                pos = candidate;
                treq->positions.push_back(candidate);
                return true;
            }
        }
        NIXL_ERROR << "GPUNETIO transfer ring exhausted";
        return false;
    };

    treq->positions.reserve(byte_striping ? DOCA_XFER_REQ_MAX :
                                            (lcnt + DOCA_XFER_REQ_SIZE - 1) / DOCA_XFER_REQ_SIZE);
    if (!reserve_position()) {
        abandon_request();
        return NIXL_ERR_BACKEND;
    }

    uint32_t desc_offset = 0;
    size_t desc_bytes_consumed = 0;
    do {
        docaXferReqGpu staged_req{};
        staged_req.has_notif_msg_idx = DOCA_NOTIF_NULL;

        uint32_t chunk_desc_count = 0;
        size_t chunk_bytes = 0;
        while (desc_offset < lcnt && staged_req.num < DOCA_XFER_REQ_SIZE &&
               (operation == NIXL_READ || chunk_desc_count < DOCA_XFER_REQ_SIZE)) {
            const uint32_t desc_idx = desc_offset;

            if (byte_striping && staged_req.num > 0 &&
                (chunk_bytes >= chunk_bytes_limit || desc_idx == lcnt - 1)) {
                break;
            }

            lmd = (nixlDocaPrivateMetadata *)local[desc_idx].metadataP;
            rmd = (nixlDocaPublicMetadata *)remote[desc_idx].metadataP;

            bool mergeable = false;
            if (operation == NIXL_READ && staged_req.num > 0) {
                const uint32_t prev = staged_req.num - 1;
                const nixl::gpunetio::readRange previous{staged_req.lbuf[prev],
                                                         staged_req.rbuf[prev],
                                                         staged_req.size[prev],
                                                         staged_req.lkey[prev],
                                                         staged_req.rkey[prev]};
                const nixl::gpunetio::readRange next{local[desc_idx].addr,
                                                     remote[desc_idx].addr,
                                                     local[desc_idx].len,
                                                     lmd->mr->get_lkey(),
                                                     rmd->mr->get_rkey()};
                constexpr size_t max_read_size =
                    std::min<size_t>(8ULL * 1024 * 1024, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
                mergeable = nixl::gpunetio::canCoalesceRead(previous, next, max_read_size);
                if (mergeable) {
                    staged_req.size[prev] += local[desc_idx].len;
                }
            }

            // After the input boundary, consume only extensions of the current READ run.
            if (chunk_desc_count >= DOCA_XFER_REQ_SIZE && !mergeable) {
                break;
            }
            const size_t remaining = local[desc_idx].len - desc_bytes_consumed;
            // A single range keeps only its last word behind the existing final-chunk barrier.
            const size_t bulk_remaining =
                ordering_tail && remaining > ordering_tail ? remaining - ordering_tail : remaining;
            const size_t piece_bytes = byte_striping && !(lcnt > 1 && desc_idx == lcnt - 1) ?
                std::min({bulk_remaining,
                          chunk_bytes_limit - chunk_bytes,
                          size_t(DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)}) :
                local[desc_idx].len;
            const size_t piece_offset = desc_bytes_consumed;
            desc_bytes_consumed += piece_bytes;
            if (desc_bytes_consumed == local[desc_idx].len) {
                ++desc_offset;
                desc_bytes_consumed = 0;
            }
            chunk_bytes += piece_bytes;
            ++chunk_desc_count;
            if (mergeable) {
                continue;
            }

            const uint32_t idx = staged_req.num;
            staged_req.lbuf[idx] = local[desc_idx].addr + piece_offset;
            staged_req.lkey[idx] = lmd->mr->get_lkey();
            staged_req.rbuf[idx] = remote[desc_idx].addr + piece_offset;
            staged_req.rkey[idx] = rmd->mr->get_rkey();
            staged_req.size[idx] = piece_bytes;
            staged_req.num_sge[idx] = 1;
            staged_req.num++;

            if (operation == NIXL_WRITE && lcnt - desc_idx > 2 && desc_offset == desc_idx + 1 &&
                (!byte_striping || local[desc_offset].len <= chunk_bytes_limit - chunk_bytes)) {
                auto *next_lmd =
                    static_cast<nixlDocaPrivateMetadata *>(local[desc_offset].metadataP);
                auto *next_rmd =
                    static_cast<nixlDocaPublicMetadata *>(remote[desc_offset].metadataP);
                if (nixl::gpunetio::canPairWrite(remote[desc_idx].addr + piece_offset,
                                                 piece_bytes,
                                                 staged_req.rkey[idx],
                                                 remote[desc_offset].addr,
                                                 remote[desc_offset].len,
                                                 next_rmd->mr->get_rkey(),
                                                 lcnt - desc_idx,
                                                 DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)) {
                    staged_req.lbuf2[idx] = local[desc_offset].addr;
                    staged_req.lkey2[idx] = next_lmd->mr->get_lkey();
                    staged_req.size2[idx] = local[desc_offset].len;
                    staged_req.num_sge[idx] = 2;
                    staged_req.has_two_sge = 1;
                    chunk_bytes += local[desc_offset].len;
                    ++desc_offset;
                }
            }
        }

        staged_req.last_rsvd = last_rsvd_flags;
        staged_req.last_posted = last_posted_flags;
        staged_req.qp_data = rdma_qp->qp_data->get_qp_gpu_dev();
        staged_req.qp_notif = rdma_qp->qp_notif->get_qp_gpu_dev();
        const size_t publish_size =
            staged_req.has_two_sge ? sizeof(staged_req) : offsetof(docaXferReqGpu, lbuf2);
        memcpy(&xferReqRingCpu[pos], &staged_req, publish_size);

        if (byte_striping) {
            NIXL_DEBUG << "WRITE byte stripe position " << pos << " bytes " << chunk_bytes;
        }

        if (desc_offset < lcnt && !reserve_position()) {
            abandon_request();
            return NIXL_ERR_BACKEND;
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
        if (newMsg.size() + 1 > notif->elems_size) {
            abandon_request();
            return NIXL_ERR_INVALID_PARAM;
        }

        auto &final_request = xferReqRingCpu[final_pos];
        notif_addr =
            (uintptr_t)nixl::gpunetio::reserveNotificationSlot(notif->send_pi,
                                                               notif->elems_num,
                                                               notif->send_addr,
                                                               notif->elems_size,
                                                               final_request.has_notif_msg_idx);
        final_request.msg_sz = newMsg.size() + 1;
        final_request.lbuf_notif = notif_addr;
        final_request.lkey_notif = notif->send_mr->get_lkey();

        memcpy((void *)notif_addr, newMsg.c_str(), final_request.msg_sz);

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
    nixl::doca::cudaDeviceGuard cuda_device(gdevs[0].first);
    if (cuda_device.status() != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    const nixl_status_t completion_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
    if (completion_status != NIXL_SUCCESS) {
        return completion_status;
    }
    nixlDocaBckndReq *treq = (nixlDocaBckndReq *)handle;

    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (treq->postStatus != NIXL_SUCCESS) {
        return treq->postStatus;
    }
    treq->postedCount = 0;
    treq->postStatus = NIXL_SUCCESS;
    treq->completionState.store(nixlDocaBckndReq::completion_state::IN_PROGRESS,
                                std::memory_order_release);
    if (operation == NIXL_WRITE && dataQpCount_ > 1) {
        return postStripedWrite(treq, remote_agent);
    }
    for (uint32_t idx : treq->positions) {
        std::lock_guard<std::mutex> lock(postLock_);
        const uint32_t completion_index =
            lastPostedReq.load(std::memory_order_relaxed) & DOCA_MAX_COMPLETION_INFLIGHT_MASK;
        xferReqRingCpu[idx].id = completion_index;
        completion_list_cpu[completion_index].xferReqRingGpu = nullptr;
        completion_list_cpu[completion_index].completed = 0;

        const doca_error_t result = operation == NIXL_READ ?
            doca_kernel_read(treq->stream, xferReqRingCpu[idx].qp_data, xferReqRingGpu, idx) :
            doca_kernel_write(treq->stream, xferReqRingCpu[idx].qp_data, xferReqRingGpu, idx);
        if (result != DOCA_SUCCESS) {
            treq->postStatus = NIXL_ERR_BACKEND;
            break;
        }
        completion_list_cpu[completion_index].xferReqRingGpu = xferReqRingGpu + idx;
        lastPostedReq.fetch_add(1, std::memory_order_relaxed);
        ++treq->postedCount;
    }

    return treq->postedCount == 0 ? treq->postStatus : NIXL_IN_PROG;
}

nixl_status_t
nixlDocaEngine::postStripedWrite(nixlDocaBckndReq *request, const std::string &remote_agent) const {
    nixl::doca::cudaDeviceGuard cuda_device(gdevs[0].first);
    if (cuda_device.status() != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    std::lock_guard<std::mutex> post_lock(postLock_);
    nixlDocaRdmaQp *peer;
    {
        std::lock_guard<std::mutex> qp_lock(qpLock);
        auto found = qpMap.find(remote_agent);
        if (found == qpMap.end()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        peer = found->second;
    }
    if (request->positions.empty() || request->positions.size() > DOCA_XFER_REQ_MAX ||
        peer->qp_write.size() != dataQpCount_) {
        request->postStatus = NIXL_ERR_BACKEND;
        *reinterpret_cast<volatile uint32_t *>(wait_exit_cpu) = 1;
        return request->postStatus;
    }

    // Publish all dependencies before any kernel can consume the final chunk.
    docaStripedWriteLaunch launch{};
    launch.count = static_cast<uint32_t>(request->positions.size());
    const uint32_t first_id = lastPostedReq.load(std::memory_order_relaxed);
    for (size_t ordinal = 0; ordinal < request->positions.size(); ++ordinal) {
        launch.positions[ordinal] = request->positions[ordinal];
        auto &entry = xferReqRingCpu[request->positions[ordinal]];
        entry.id = (first_id + ordinal) & DOCA_MAX_COMPLETION_INFLIGHT_MASK;
        entry.qp_data = peer->qp_write[ordinal % dataQpCount_]->get_qp_gpu_dev();
        entry.completion_list = completion_list_gpu;
        entry.wait_exit = wait_exit_gpu;
        entry.wait_completion_count = 0;
        entry.peer_write_completed = nullptr;
        entry.write_ticket = 0;
        completion_list_cpu[entry.id].xferReqRingGpu = nullptr;
        completion_list_cpu[entry.id].completed = 0;
    }
    auto &last = xferReqRingCpu[request->positions.back()];
    last.posting_ctas_done = 0;
    last.peer_write_completed = peer->writeCompletedGpu.get();
    last.write_ticket = ++peer->postedWriteTicket;
    for (size_t ordinal = 0; ordinal + 1 < request->positions.size(); ++ordinal) {
        last.wait_completion_ids[last.wait_completion_count++] =
            xferReqRingCpu[request->positions[ordinal]].id;
    }
    std::atomic_thread_fence(std::memory_order_release);

    // One stream orders all grids, including the previous same-peer final post.
    // This can reduce overlap between requests compared with per-QP streams.
    const auto result =
        doca_kernel_write_striped(post_stream[0], xferReqRingGpu, launch, dataQpCount_);
    if (result != DOCA_SUCCESS) {
        request->postStatus = NIXL_ERR_BACKEND;
        *reinterpret_cast<volatile uint32_t *>(wait_exit_cpu) = 1;
        return request->postStatus;
    }
    // A successful launch owns every chunk, even if a later GPU failure aborts it.
    for (uint32_t position : request->positions) {
        completion_list_cpu[xferReqRingCpu[position].id].xferReqRingGpu = xferReqRingGpu + position;
        lastPostedReq.fetch_add(1, std::memory_order_relaxed);
        ++request->postedCount;
    }
    return NIXL_IN_PROG;
}

nixl_status_t
nixlDocaEngine::checkXfer(nixlBackendReqH *handle) const {
    const nixl_status_t completion_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
    if (completion_status != NIXL_SUCCESS) {
        return completion_status;
    }
    nixlDocaBckndReq *treq = (nixlDocaBckndReq *)handle;
    uint32_t completion_index;

    auto state = treq->completionState.load(std::memory_order_acquire);
    if (state == nixlDocaBckndReq::completion_state::COMPLETE) {
        return treq->postStatus;
    }
    if (state == nixlDocaBckndReq::completion_state::COMPLETING) {
        treq->completionState.wait(state, std::memory_order_acquire);
        return treq->postStatus;
    }

    for (size_t i = 0; i < treq->postedCount; ++i) {
        const uint32_t idx = treq->positions[i];
        completion_index = xferReqRingCpu[idx].id & (DOCA_MAX_COMPLETION_INFLIGHT_MASK);

        if (((volatile docaXferCompletion *)completion_list_cpu)[completion_index].completed != 1) {
            return NIXL_IN_PROG;
        }
    }

    retireRequest(treq);
    return nixl::gpunetio::completionStatus(wait_exit_cpu) == NIXL_SUCCESS ? treq->postStatus :
                                                                             NIXL_ERR_BACKEND;
}

void
nixlDocaEngine::retireRequest(nixlDocaBckndReq *request) const {
    auto state = request->completionState.load(std::memory_order_acquire);
    if (state == nixlDocaBckndReq::completion_state::COMPLETE) {
        return;
    }
    if (state == nixlDocaBckndReq::completion_state::COMPLETING) {
        request->completionState.wait(state, std::memory_order_acquire);
        return;
    }

    auto expected = nixlDocaBckndReq::completion_state::IN_PROGRESS;
    if (request->completionState.compare_exchange_strong(
            expected, nixlDocaBckndReq::completion_state::COMPLETING, std::memory_order_acq_rel)) {
        for (size_t i = 0; i < request->postedCount; ++i) {
            const uint32_t idx = request->positions[i];
            *((volatile uint8_t *)&xferReqRingCpu[idx].in_use) = 0;
            NIXL_INFO << "DOCA retireRequest pos " << idx << " COMPLETED!";
        }
        request->completionState.store(nixlDocaBckndReq::completion_state::COMPLETE,
                                       std::memory_order_release);
        request->completionState.notify_all();
    } else if (expected == nixlDocaBckndReq::completion_state::COMPLETING) {
        request->completionState.wait(expected, std::memory_order_acquire);
    }
}

nixl_status_t
nixlDocaEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *treq = static_cast<nixlDocaBckndReq *>(handle);
    if (dataQpCount_ > 1 && treq->postedCount > 0 &&
        nixl::gpunetio::completionStatus(wait_exit_cpu) != NIXL_SUCCESS) {
        nixl::doca::cudaDeviceGuard cuda_device(gdevs[0].first);
        if (cuda_device.status() != cudaSuccess) {
            return NIXL_ERR_BACKEND;
        }
        for (int stream = 0; stream < nstreams; ++stream) {
            if (cudaStreamSynchronize(post_stream[stream]) != cudaSuccess) {
                return NIXL_ERR_BACKEND;
            }
        }
    }
    if (treq->postedCount == 0) {
        for (uint32_t idx : treq->positions) {
            xferReqReserved_[idx].store(false, std::memory_order_release);
        }
        delete treq;
        return NIXL_SUCCESS;
    }

    if (treq->completionState.load(std::memory_order_acquire) !=
        nixlDocaBckndReq::completion_state::COMPLETE) {
        nixl_status_t status = checkXfer(handle);
        if (status == NIXL_IN_PROG) {
            return NIXL_ERR_BACKEND;
        }
    }

    for (uint32_t idx : treq->positions) {
        xferReqReserved_[idx].store(false, std::memory_order_release);
    }
    delete treq;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::getNotifs(notif_list_t &notif_list) {
    const nixl_status_t completion_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
    if (completion_status != NIXL_SUCCESS) {
        return completion_status;
    }
    uint32_t recv_idx;
    uint32_t num_msg = 0;
    char *addr;

    // Lock required to prevent inconsistency if another notifyQp (new peer) is added
    // while getNotifs is running
    std::lock_guard<std::mutex> lock(notifLock);
    for (auto &notif : notifMap) {
        const nixl_status_t preflight_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
        if (preflight_status != NIXL_SUCCESS) {
            return preflight_status;
        }
        doca_gpu_dev_verbs_qp *notif_qp_gpu;
        {
            std::lock_guard<std::mutex> qp_lock(qpLock);
            auto qp = qpMap.find(notif.first);
            if (qp == qpMap.end()) {
                return NIXL_ERR_BACKEND;
            }
            notif_qp_gpu = qp->second->qp_notif->get_qp_gpu_dev();
        }
        ((volatile struct docaNotif *)notif_progress_cpu)->qp_gpu = notif_qp_gpu;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        while (((volatile struct docaNotif *)notif_progress_cpu)->qp_gpu != nullptr) {
            const nixl_status_t wait_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
            if (wait_status != NIXL_SUCCESS) {
                return wait_status;
            }
        }
        const nixl_status_t progress_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
        if (progress_status != NIXL_SUCCESS) {
            return progress_status;
        }
        num_msg = ((volatile struct docaNotif *)notif_progress_cpu)->msg_num;
        while (num_msg > 0) {
            recv_idx = notif.second->recv_pi.load() & (DOCA_MAX_NOTIF_INFLIGHT - 1);
            addr = (char *)(notif.second->recv_addr + (recv_idx * notif.second->elems_size));
            std::string_view payload;
            const bool valid = parseNotification(std::string_view(addr, notif.second->elems_size),
                                                 msg_tag_start,
                                                 msg_tag_end,
                                                 payload);
            if (valid) {
                notif_list.emplace_back(notif.first, std::string(payload));
            }
            // Consume malformed messages too, rather than repeatedly parsing the same slot.
            memset(addr, 0, msg_tag_start.size());
            notif.second->recv_pi.fetch_add(1);
            num_msg--;
            if (!valid) {
                NIXL_ERROR << "Malformed notification from " << notif.first;
                return NIXL_ERR_BACKEND;
            }
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    const nixl_status_t completion_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
    if (completion_status != NIXL_SUCCESS) {
        return completion_status;
    }
    struct nixlDocaNotif *notif;
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

    doca_gpu_dev_verbs_qp *notif_qp_gpu;
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
    while (((volatile struct docaNotif *)notif_send_cpu)->qp_gpu != nullptr) {
        const nixl_status_t wait_status = nixl::gpunetio::completionStatus(wait_exit_cpu);
        if (wait_status != NIXL_SUCCESS) {
            return wait_status;
        }
    }

    return nixl::gpunetio::completionStatus(wait_exit_cpu);
}
