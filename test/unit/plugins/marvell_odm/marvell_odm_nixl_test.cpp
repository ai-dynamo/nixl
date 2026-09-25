/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Marvell Technology, Inc. All rights reserved.
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
 * Consistency is validated from the GPU read-back only.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <getopt.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "nixl.h"
#include "nixl_descriptors.h"
#include "test_utils.h"

namespace {

constexpr const char *kAgentName = "MarvellOdmNixlTestAgent";
constexpr size_t kDefaultTransferSize = 65536;
constexpr unsigned char kTestPattern = 0x33;
constexpr int kMesonSkip = 77;

void
logOdmDeviceError(const std::string &path, const char *operation, int err) {
    switch (err) {
    case ENOENT:
        std::cerr << "ODM: " << operation << "(" << path
                  << ") failed: device node not present (is the ODM "
                     "kernel module loaded?)"
                  << std::endl;
        break;
    case EACCES:
        std::cerr << "ODM: " << operation << "(" << path
                  << ") failed: permission denied (check ODM char-device "
                     "permissions)"
                  << std::endl;
        break;
    case ENODEV:
    case ENXIO:
        std::cerr << "ODM: " << operation << "(" << path
                  << ") failed: device present but not ready (driver loaded, "
                     "hardware may be unavailable)"
                  << std::endl;
        break;
    default:
        std::cerr << "ODM: " << operation << "(" << path << ") failed: " << std::strerror(err)
                  << std::endl;
        break;
    }
}

int
odmDeviceFailureExitCode(int err) {
    return (err == ENOENT) ? kMesonSkip : 1;
}

std::string
devicePath(const std::string &dev_name) {
    return (!dev_name.empty() && dev_name[0] == '/') ? dev_name : ("/dev/" + dev_name);
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
              << "  --device NAME           ODM device name (default: odm0)\n"
              << "  --qid ID                ODM queue id (sets start/end when range unset)\n"
              << "  --odm_qid_start ID      ODM queue range start (default: 0)\n"
              << "  --odm_qid_end ID        ODM queue range end (default: 15)\n"
              << "  --size BYTES            Transfer size (default: " << kDefaultTransferSize
              << ")\n"
              << "  --pattern BYTE          Fill/verify byte pattern (default: 0x33)\n"
              << "  --odm_use_io_uring      Use io_uring kernel-side queue spray for transfers\n"
              << "  --host-dram             Test host DRAM <-> device DRAM (host-VA ioctl path)\n"
              << "  --help                  Show this help\n"
              << "\n"
              << "Default: VRAM -> ODM write, ODM -> VRAM read, validate GPU round-trip.\n"
              << "With --host-dram: host DRAM -> device DRAM write, device -> host read.\n";
}

nixl_status_t
waitForXfer(nixlAgent &agent, nixlXferReqH *req, bool sync_cuda = true) {
    nixl_status_t status = agent.postXferReq(req);
    if (status < NIXL_SUCCESS) {
        return status;
    }
    constexpr int kMaxPolls = 1000000;
    int polls = 0;
    while (status == NIXL_IN_PROG) {
        if (++polls > kMaxPolls) {
            std::cerr << "ODM: transfer timed out after " << kMaxPolls << " status polls"
                      << std::endl;
            return NIXL_ERR_BACKEND;
        }
        status = agent.getXferStatus(req);
    }
    if (sync_cuda) {
        cudaDeviceSynchronize();
    }
    return status;
}

} // namespace

int
main(int argc, char **argv) {
    std::string dev_name = "odm0";
    std::string qid_str;
    std::string qid_start_str = "0";
    std::string qid_end_str = "15";
    bool qid_start_set = false;
    bool qid_end_set = false;
    size_t transfer_size = kDefaultTransferSize;
    unsigned char test_pattern = kTestPattern;
    bool use_io_uring = false;
    bool host_dram = false;

    static struct option long_opts[] = {
        {"device", required_argument, nullptr, 'D'},
        {"qid", required_argument, nullptr, 'q'},
        {"odm_qid_start", required_argument, nullptr, 'Q'},
        {"odm_qid_end", required_argument, nullptr, 'R'},
        {"size", required_argument, nullptr, 's'},
        {"pattern", required_argument, nullptr, 'p'},
        {"odm_use_io_uring", no_argument, nullptr, 'u'},
        {"host-dram", no_argument, nullptr, 'H'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "D:q:Q:R:s:p:uHh", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'D':
            dev_name = optarg;
            break;
        case 'q':
            qid_str = optarg;
            break;
        case 'Q':
            qid_start_str = optarg;
            qid_start_set = true;
            break;
        case 'R':
            qid_end_str = optarg;
            qid_end_set = true;
            break;
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
        case 'u':
            use_io_uring = true;
            break;
        case 'H':
            host_dram = true;
            break;
        case 'h':
            printUsage(argv[0]);
            return 0;
        default:
            printUsage(argv[0]);
            return 1;
        }
    }

    if (!qid_str.empty()) {
        if (!qid_start_set) {
            qid_start_str = qid_str;
        }
        if (!qid_end_set) {
            qid_end_str = qid_str;
        }
    }

    if (!host_dram) {
        int device_count = 0;
        const cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        if (cuda_err != cudaSuccess || device_count == 0) {
            std::cerr << "Error: CUDA GPU not available: " << cudaGetErrorString(cuda_err)
                      << std::endl;
            return 1;
        }
    }

    const std::string path = devicePath(dev_name);
    if (access(path.c_str(), F_OK) != 0) {
        const int access_err = errno;
        logOdmDeviceError(path, "access", access_err);
        if (access_err == ENOENT) {
            std::cout << "SKIP: ODM device not present at " << path << std::endl;
            return kMesonSkip;
        }
        return 1;
    }
    if (access(path.c_str(), R_OK | W_OK) != 0) {
        const int access_err = errno;
        logOdmDeviceError(path, "access", access_err);
        return odmDeviceFailureExitCode(access_err);
    }

    if (!host_dram) {
        CUresult cu_res = cuInit(0);
        if (cu_res != CUDA_SUCCESS) {
            std::cerr << "Error: cuInit failed" << std::endl;
            return 1;
        }
    }

    std::cout << "Phase 1: Initialize NIXL agent and ODM backend" << std::endl;
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    nixlAgent agent(kAgentName, cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    nixl_status_t ret = agent.getPluginParams("MARVELL_ODM", mems, params);
    if (ret != NIXL_SUCCESS) {
        std::cerr
            << "Error: MARVELL_ODM plugin not available (build with -Denable_plugins=MARVELL_ODM)"
            << std::endl;
        return 1;
    }

    params["dmadev"] = dev_name;
    params["odm_qid"] = qid_str;
    params["odm_qid_start"] = qid_start_str;
    params["odm_qid_end"] = qid_end_str;
    if (use_io_uring) {
        params["odm_use_io_uring"] = "1";
    }

    nixlBackendH *backend = nullptr;
    ret = agent.createBackend("MARVELL_ODM", params, backend);
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

    if (host_dram) {
        nixl_opt_args_t extra;
        extra.backends.push_back(backend);

        std::cout << "Phase 2: Register host DRAM and device DRAM (auto IOVA)" << std::endl;
        nixl_reg_dlist_t host_list(DRAM_SEG);
        nixl_reg_dlist_t dev_list(DRAM_SEG);
        nixlBlobDesc blob_host;
        blob_host.addr = reinterpret_cast<uintptr_t>(host_buf);
        blob_host.len = transfer_size;
        blob_host.devId = 0;
        nixlBlobDesc blob_dev;
        blob_dev.addr = 0;
        blob_dev.len = transfer_size;
        blob_dev.devId = 0;
        host_list.addDesc(blob_host);
        dev_list.addDesc(blob_dev);

        ret = agent.registerMem(host_list, &extra);
        nixl_exit_on_failure(ret, "registerMem host DRAM", kAgentName);
        ret = agent.registerMem(dev_list, &extra);
        nixl_exit_on_failure(ret, "registerMem device DRAM", kAgentName);

        std::cout << "Phase 3: host DRAM -> device DRAM write" << std::endl;
        {
            nixl_xfer_dlist_t src_list = host_list.trim();
            nixl_xfer_dlist_t dst_list = dev_list.trim();
            ret =
                agent.createXferReq(NIXL_WRITE, src_list, dst_list, kAgentName, write_req, &extra);
            nixl_exit_on_failure(ret, "createXferReq host->device", kAgentName);
            ret = waitForXfer(agent, write_req, false);
            nixl_exit_on_failure(ret >= NIXL_SUCCESS, "host->device transfer", kAgentName);
            agent.releaseXferReq(write_req);
            write_req = nullptr;
        }

        std::cout << "Phase 4: device DRAM -> host DRAM read and validate" << std::endl;
        std::memset(host_buf, 0, transfer_size);
        {
            nixl_xfer_dlist_t src_list = host_list.trim();
            nixl_xfer_dlist_t dst_list = dev_list.trim();
            ret = agent.createXferReq(NIXL_READ, src_list, dst_list, kAgentName, read_req, &extra);
            nixl_exit_on_failure(ret, "createXferReq device->host", kAgentName);
            ret = waitForXfer(agent, read_req, false);
            nixl_exit_on_failure(ret >= NIXL_SUCCESS, "device->host transfer", kAgentName);
            agent.releaseXferReq(read_req);
            read_req = nullptr;
        }

        if (!validatePattern(host_buf, transfer_size, test_pattern)) {
            std::cerr << "ODM host DRAM round-trip validation FAILED" << std::endl;
            result = 1;
        } else {
            std::cout << "ODM host DRAM round-trip validation PASSED" << std::endl;
        }

        agent.deregisterMem(host_list, &extra);
        agent.deregisterMem(dev_list, &extra);
        free(host_buf);
        std::cout << (result == 0 ? "ODM test PASSED" : "ODM test FAILED") << std::endl;
        return result;
    }

    cudaError_t cuerr = cudaMalloc(&gpu_buf, transfer_size);
    if (cuerr != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cuerr) << std::endl;
        free(host_buf);
        return 1;
    }

    unsigned int sync_memops = 1;
    if (cuPointerSetAttribute(&sync_memops,
                              CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                              reinterpret_cast<CUdeviceptr>(gpu_buf)) != CUDA_SUCCESS) {
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

    std::cout << "Phase 2: Register VRAM and device memory (auto IOVA)" << std::endl;
    nixl_reg_dlist_t vram_list(VRAM_SEG);
    nixl_reg_dlist_t odm_list(DRAM_SEG);
    nixlBlobDesc blob_vram;
    blob_vram.addr = reinterpret_cast<uintptr_t>(gpu_buf);
    blob_vram.len = transfer_size;
    blob_vram.devId = 0;
    nixlBlobDesc blob_odm;
    blob_odm.addr = 0;
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
    if (gpu_buf != nullptr) {
        cudaFree(gpu_buf);
    }
    free(host_buf);

    std::cout << (result == 0 ? "ODM test PASSED" : "ODM test FAILED") << std::endl;
    return result;
}
