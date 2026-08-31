/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * ODM plugin round-trip test: VRAM -> ODM (write), then ODM -> VRAM (read).
 * Consistency is validated from the GPU read-back only; no DAX/BAR2 seeding.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <getopt.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "nixl.h"
#include "nixl_descriptors.h"
#include "odm_ioctl.h"
#include "test_utils.h"

namespace {

constexpr const char *kAgentName = "ODMNixlTestAgent";
constexpr size_t kDefaultTransferSize = 65536;
constexpr unsigned char kTestPattern = 0x33;

std::string
devicePath(const std::string &dev_name) {
    return (!dev_name.empty() && dev_name[0] == '/') ? dev_name : ("/dev/" + dev_name);
}

struct OdmIovaAlloc {
    int device_fd = -1;
    uint64_t addr = 0;
    uint32_t size = 0;
};

bool
allocOdmIova(const std::string &dev_name, size_t transfer_size, OdmIovaAlloc &out) {
    out = {};
    if (const char *env = std::getenv("ODM_ADDR")) {
        const uint64_t v = std::strtoull(env, nullptr, 0);
        if (v != 0) {
            out.addr = v;
            out.size = static_cast<uint32_t>(transfer_size);
            std::cout << "ODM: using IOVA 0x" << std::hex << out.addr << std::dec
                      << " from ODM_ADDR (no GET_IOVA alloc)" << std::endl;
            return true;
        }
    }

    const std::string path = devicePath(dev_name);
    out.device_fd = open(path.c_str(), O_RDWR);
    if (out.device_fd < 0) {
        std::cerr << "ODM: open(" << path << ") failed: " << std::strerror(errno) << std::endl;
        return false;
    }

    struct mrvl_dma_iova_commands cmd{};
    cmd.target_iova_size = static_cast<uint32_t>(transfer_size);
    if (ioctl(out.device_fd, MRVL_CXL_GET_IOVA_COMMAND, &cmd) < 0) {
        std::cerr << "ODM: GET_IOVA on " << path << " failed: " << std::strerror(errno)
                  << std::endl;
        close(out.device_fd);
        out.device_fd = -1;
        return false;
    }

    out.addr = cmd.target_iova_addr;
    out.size = cmd.target_iova_size;
    std::cout << "ODM: allocated IOVA 0x" << std::hex << out.addr << std::dec << " (size "
              << out.size << ") via GET_IOVA on " << path << std::endl;
    return true;
}

void
freeOdmIova(OdmIovaAlloc &alloc) {
    if (alloc.device_fd < 0 || alloc.addr == 0) {
        return;
    }
    struct mrvl_dma_iova_commands cmd{};
    cmd.target_iova_addr = alloc.addr;
    cmd.target_iova_size = alloc.size;
    if (ioctl(alloc.device_fd, MRVL_CXL_FREE_IOVA_COMMAND, &cmd) < 0) {
        std::cerr << "ODM: FREE_IOVA failed: " << std::strerror(errno) << std::endl;
    }
    close(alloc.device_fd);
    alloc = {};
}

void
fillPattern(void *buf, size_t len, unsigned char pattern) {
    std::memset(buf, pattern, len);
}

bool
validatePattern(const void *buf, size_t len, unsigned char expected) {
    const auto *bytes = static_cast<const unsigned char *>(buf);
    for (size_t i = 0; i < len; ++i) {
        if (bytes[i] != expected) {
            std::cerr << "Validation failed at offset " << i << ": got 0x" << std::hex
                      << static_cast<unsigned>(bytes[i]) << " expected 0x"
                      << static_cast<unsigned>(expected) << std::dec << std::endl;
            return false;
        }
    }
    return true;
}

void
printUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --device NAME       ODM device name (default: odm0)\n"
              << "  --qid ID            ODM queue id (default: 0)\n"
              << "  --qid-start ID      ODM queue range start (default: --qid)\n"
              << "  --qid-end ID        ODM queue range end (default: --qid)\n"
              << "  --odm-addr ADDR     ODM target IOVA (default: GET_IOVA / ODM_ADDR)\n"
              << "  --size BYTES        Transfer size (default: " << kDefaultTransferSize << ")\n"
              << "  --pattern BYTE      Fill/verify byte pattern (default: 0x33)\n"
              << "  --help              Show this help\n"
              << "\n"
              << "Runs a VRAM -> ODM write followed by an ODM -> VRAM read and validates\n"
              << "the round-trip from GPU memory. No DAX/BAR2 seeding is used.\n";
}

nixl_status_t
waitForXfer(nixlAgent &agent, nixlXferReqH *req) {
    nixl_status_t status = agent.postXferReq(req);
    if (status < NIXL_SUCCESS) {
        return status;
    }
    while (status == NIXL_IN_PROG) {
        status = agent.getXferStatus(req);
    }
    cudaDeviceSynchronize();
    return status;
}

} // namespace

int
main(int argc, char **argv) {
    bool odm_addr_set = false;
    OdmIovaAlloc odm_iova{};
    std::string dev_name = "odm0";
    std::string qid_str = "0";
    std::string qid_start_str;
    std::string qid_end_str;
    uint64_t odm_addr = 0;
    size_t transfer_size = kDefaultTransferSize;
    unsigned char test_pattern = kTestPattern;

    static struct option long_opts[] = {
        {"device", required_argument, nullptr, 'D'},
        {"qid", required_argument, nullptr, 'q'},
        {"qid-start", required_argument, nullptr, 'Q'},
        {"qid-end", required_argument, nullptr, 'R'},
        {"odm-addr", required_argument, nullptr, 'a'},
        {"size", required_argument, nullptr, 's'},
        {"pattern", required_argument, nullptr, 'p'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "D:q:Q:R:a:s:p:h", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'D':
            dev_name = optarg;
            break;
        case 'q':
            qid_str = optarg;
            break;
        case 'Q':
            qid_start_str = optarg;
            break;
        case 'R':
            qid_end_str = optarg;
            break;
        case 'a': {
            char *end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(optarg, &end, 0);
            if (errno != 0 || end == optarg || *end != '\0') {
                std::cerr << "Invalid --odm-addr value: " << optarg << std::endl;
                return 1;
            }
            odm_addr = static_cast<uint64_t>(parsed);
            odm_addr_set = true;
            break;
        }
        case 's': {
            char *end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(optarg, &end, 0);
            if (errno != 0 || end == optarg || *end != '\0' || parsed == 0) {
                std::cerr << "Invalid --size value: " << optarg << std::endl;
                return 1;
            }
            transfer_size = static_cast<size_t>(parsed);
            break;
        }
        case 'p': {
            char *end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(optarg, &end, 0);
            if (errno != 0 || end == optarg || *end != '\0' || parsed > 0xFF) {
                std::cerr << "Invalid --pattern value: " << optarg << std::endl;
                return 1;
            }
            test_pattern = static_cast<unsigned char>(parsed);
            break;
        }
        case 'h':
            printUsage(argv[0]);
            return 0;
        default:
            printUsage(argv[0]);
            return 1;
        }
    }

    if (qid_start_str.empty()) {
        qid_start_str = qid_str;
    }
    if (qid_end_str.empty()) {
        qid_end_str = qid_str;
    }

    int device_count = 0;
    const cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        std::cerr << "Error: CUDA GPU not available: " << cudaGetErrorString(cuda_err) << std::endl;
        return 1;
    }

    const std::string path = devicePath(dev_name);
    if (access(path.c_str(), R_OK | W_OK) != 0) {
        std::cerr << "Error: ODM device not accessible: " << path << ": " << std::strerror(errno)
                  << std::endl;
        return 1;
    }

    if (!odm_addr_set) {
        if (!allocOdmIova(dev_name, transfer_size, odm_iova)) {
            return 1;
        }
        odm_addr = odm_iova.addr;
    }

    CUresult cu_res = cuInit(0);
    if (cu_res != CUDA_SUCCESS) {
        std::cerr << "Error: cuInit failed" << std::endl;
        return 1;
    }

    std::cout << "Phase 1: Initialize NIXL agent and ODM backend" << std::endl;
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    nixlAgent agent(kAgentName, cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    nixl_status_t ret = agent.getPluginParams("ODM", mems, params);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Error: ODM plugin not available (build with -Denable_plugins=ODM)"
                  << std::endl;
        return 1;
    }

    params["dmadev_param"] = dev_name;
    params["odm_qid"] = qid_str;
    params["odm_qid_start"] = qid_start_str;
    params["odm_qid_end"] = qid_end_str;

    nixlBackendH *backend = nullptr;
    ret = agent.createBackend("ODM", params, backend);
    if (ret != NIXL_SUCCESS || backend == nullptr) {
        std::cerr << "Error: failed to create ODM backend for " << path << std::endl;
        return 1;
    }

    void *host_buf = nullptr;
    void *gpu_buf = nullptr;
    nixlXferReqH *write_req = nullptr;
    nixlXferReqH *read_req = nullptr;
    int result = 0;

    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (posix_memalign(&host_buf, page_size, transfer_size) != 0) {
        std::cerr << "Host allocation failed" << std::endl;
        return 1;
    }
    std::cout << "Using test pattern 0x" << std::hex << static_cast<unsigned>(test_pattern)
              << std::dec << std::endl;
    fillPattern(host_buf, transfer_size, test_pattern);

    cudaError_t cuerr = cudaMalloc(&gpu_buf, transfer_size);
    if (cuerr != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cuerr) << std::endl;
        free(host_buf);
        return 1;
    }

    unsigned int sync_memops = 1;
    cu_res = cuPointerSetAttribute(
        &sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, reinterpret_cast<CUdeviceptr>(gpu_buf));
    if (cu_res != CUDA_SUCCESS) {
        std::cerr << "Warning: cuPointerSetAttribute(SYNC_MEMOPS) failed" << std::endl;
    }

    cuerr = cudaMemcpy(gpu_buf, host_buf, transfer_size, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(cuerr) << std::endl;
        cudaFree(gpu_buf);
        free(host_buf);
        return 1;
    }

    nixl_opt_args_t extra;
    extra.backends.push_back(backend);

    std::cout << "Phase 2: Register VRAM and ODM memory at 0x" << std::hex << odm_addr << std::dec
              << std::endl;
    nixl_reg_dlist_t vram_list(VRAM_SEG);
    nixl_reg_dlist_t odm_list(ODM_MEM_SEG);
    nixlBlobDesc blob_vram;
    blob_vram.addr = reinterpret_cast<uintptr_t>(gpu_buf);
    blob_vram.len = transfer_size;
    blob_vram.devId = 0;
    nixlBlobDesc blob_odm;
    blob_odm.addr = odm_addr;
    blob_odm.len = transfer_size;
    blob_odm.devId = 0;
    vram_list.addDesc(blob_vram);
    odm_list.addDesc(blob_odm);

    ret = agent.registerMem(vram_list, &extra);
    nixl_exit_on_failure(ret, "registerMem VRAM", kAgentName);
    ret = agent.registerMem(odm_list, &extra);
    nixl_exit_on_failure(ret, "registerMem ODM", kAgentName);

    std::cout << "Phase 3: VRAM -> ODM write (seed device memory)" << std::endl;
    {
        nixl_xfer_dlist_t src_list = vram_list.trim();
        nixl_xfer_dlist_t dst_list = odm_list.trim();
        ret = agent.createXferReq(NIXL_WRITE, src_list, dst_list, kAgentName, write_req, &extra);
        nixl_exit_on_failure(ret, "createXferReq VRAM->ODM", kAgentName);
        ret = waitForXfer(agent, write_req);
        nixl_exit_on_failure(ret >= NIXL_SUCCESS, "VRAM->ODM transfer", kAgentName);
        agent.releaseXferReq(write_req);
        write_req = nullptr;
    }

    std::cout << "Phase 4: ODM -> VRAM read and validate round-trip" << std::endl;
    cuerr = cudaMemset(gpu_buf, 0, transfer_size);
    if (cuerr != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(cuerr) << std::endl;
        result = 1;
        goto cleanup;
    }

    {
        nixl_xfer_dlist_t src_list = vram_list.trim();
        nixl_xfer_dlist_t dst_list = odm_list.trim();
        ret = agent.createXferReq(NIXL_READ, src_list, dst_list, kAgentName, read_req, &extra);
        nixl_exit_on_failure(ret, "createXferReq ODM->VRAM", kAgentName);
        ret = waitForXfer(agent, read_req);
        nixl_exit_on_failure(ret >= NIXL_SUCCESS, "ODM->VRAM transfer", kAgentName);
        agent.releaseXferReq(read_req);
        read_req = nullptr;
    }

    std::memset(host_buf, 0, transfer_size);
    cuerr = cudaMemcpy(host_buf, gpu_buf, transfer_size, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(cuerr) << std::endl;
        result = 1;
        goto cleanup;
    }
    if (!validatePattern(host_buf, transfer_size, test_pattern)) {
        std::cerr << "ODM VRAM round-trip validation FAILED" << std::endl;
        result = 1;
    } else {
        std::cout << "ODM VRAM round-trip validation PASSED" << std::endl;
    }

cleanup:
    if (write_req != nullptr) {
        agent.releaseXferReq(write_req);
    }
    if (read_req != nullptr) {
        agent.releaseXferReq(read_req);
    }
    agent.deregisterMem(vram_list, &extra);
    agent.deregisterMem(odm_list, &extra);
    freeOdmIova(odm_iova);
    if (gpu_buf != nullptr) {
        cudaFree(gpu_buf);
    }
    free(host_buf);

    std::cout << (result == 0 ? "ODM test PASSED" : "ODM test FAILED") << std::endl;
    return result;
}
