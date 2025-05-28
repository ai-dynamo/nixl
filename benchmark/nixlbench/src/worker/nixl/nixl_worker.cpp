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

#include "worker/nixl/nixl_worker.h"
#include <cstring>
#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include "utils/utils.h"
#include <unistd.h>
#include <utility>
#include <sys/time.h>
#include <utils/serdes/serdes.h>
#include <omp.h>

#define USE_VMM 0
#define ROUND_UP(value, granularity) ((((value) + (granularity) - 1) / (granularity)) * (granularity))

static uintptr_t gds_running_ptr = 0x0;
static std::vector<std::vector<xferBenchIOV>> gds_remote_iovs;
static std::vector<std::vector<xferBenchIOV>> storage_remote_iovs;

#if HAVE_CUDA
static size_t __attribute__((unused)) padded_size = 0;
static CUmemGenericAllocationHandle __attribute__((unused)) handle;
#endif

#define CHECK_NIXL_ERROR(result, message)                                         \
    do {                                                                          \
        if (0 != result) {                                                        \
            std::cerr << "NIXL: " << message << " (Error code: " << result        \
                      << ")" << std::endl;                                        \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while(0)

#if HAVE_CUDA
    #define HANDLE_VRAM_SEGMENT(_seg_type)                                        \
        _seg_type = VRAM_SEG;
#else
    #define HANDLE_VRAM_SEGMENT(_seg_type)                                        \
        std::cerr << "VRAM segment type not supported without CUDA" << std::endl; \
        std::exit(EXIT_FAILURE);
#endif

#define GET_SEG_TYPE(is_initiator)                                                \
    ({                                                                            \
        std::string _seg_type_str = ((is_initiator) ?                             \
                                     xferBenchConfig::initiator_seg_type :        \
                                     xferBenchConfig::target_seg_type);           \
        nixl_mem_t _seg_type;                                                     \
        if (0 == _seg_type_str.compare("DRAM")) {                                 \
            _seg_type = DRAM_SEG;                                                 \
        } else if (0 == _seg_type_str.compare("VRAM")) {                          \
            HANDLE_VRAM_SEGMENT(_seg_type);                                       \
        } else {                                                                  \
            std::cerr << "Invalid segment type: "                                 \
                        << _seg_type_str << std::endl;                            \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
        _seg_type;                                                                \
    })

xferBenchNixlWorker::xferBenchNixlWorker(int *argc, char ***argv, std::vector<std::string> devices) : xferBenchWorker(argc, argv) {
    seg_type = GET_SEG_TYPE(isInitiator());

    int rank;
    std::string backend_name;
    nixl_b_params_t backend_params;
    bool enable_pt = xferBenchConfig::enable_pt;
    char hostname[256];
    nixl_mem_list_t mems;
    std::vector<nixl_backend_t> plugins;

    rank = rt->getRank();

    nixlAgentConfig dev_meta(enable_pt);

    agent = new nixlAgent(name, dev_meta);

    agent->getAvailPlugins(plugins);

    if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX_MO) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GDS) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_POSIX)){
        backend_name = xferBenchConfig::backend;
    } else {
        std::cerr << "Unsupported backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    agent->getPluginParams(backend_name, mems, backend_params);

    if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX_MO)){
        // No need to set device_list if all is specified
        // fallback to backend preference
        if (devices[0] != "all" && devices.size() >= 1) {
            if (isInitiator()) {
                backend_params["device_list"] = devices[rank];
                if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX_MO)) {
                    backend_params["num_ucx_engines"] = xferBenchConfig::num_initiator_dev;
                }
            } else {
                backend_params["device_list"] = devices[rank - xferBenchConfig::num_initiator_dev];
                if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX_MO)) {
                    backend_params["num_ucx_engines"] = xferBenchConfig::num_target_dev;
                }
            }
        }

        if (gethostname(hostname, 256)) {
           std::cerr << "Failed to get hostname" << std::endl;
           exit(EXIT_FAILURE);
        }

        std::cout << "Init nixl worker, dev " << (("all" == devices[0]) ? "all" : backend_params["device_list"])
                  << " rank " << rank << ", type " << name << ", hostname "
                  << hostname << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GDS)) {
        // Using default param values for GDS backend
        std::cout << "GDS backend" << std::endl;
        backend_params["batch_pool_size"] = std::to_string(xferBenchConfig::gds_batch_pool_size);
        backend_params["batch_limit"] = std::to_string(xferBenchConfig::gds_batch_limit);
        std::cout << "GDS batch pool size: " << xferBenchConfig::gds_batch_pool_size << std::endl;
        std::cout << "GDS batch limit: " << xferBenchConfig::gds_batch_limit << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_POSIX)) {
        // Set API type parameter for POSIX backend
        if (xferBenchConfig::posix_api_type == XFERBENCH_POSIX_API_AIO) {
            backend_params["use_aio"] = true;
            backend_params["use_uring"] = false;
        } else if (xferBenchConfig::posix_api_type == XFERBENCH_POSIX_API_URING) {
            backend_params["use_aio"] = false;
            backend_params["use_uring"] = true;
        }
        std::cout << "POSIX backend with API type: " << xferBenchConfig::posix_api_type << std::endl;
    } else {
        std::cerr << "Unsupported backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    agent->createBackend(backend_name, backend_params, backend_engine);
}

xferBenchNixlWorker::~xferBenchNixlWorker() {
    if (agent) {
        delete agent;
        agent = nullptr;
    }
}

// Convert vector of xferBenchIOV to nixl_reg_dlist_t
static void iovListToNixlRegDlist(const std::vector<xferBenchIOV> &iov_list,
                                 nixl_reg_dlist_t &dlist) {
    nixlBlobDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        dlist.addDesc(desc);
    }
}

// Convert nixl_xfer_dlist_t to vector of xferBenchIOV
static std::vector<xferBenchIOV> nixlXferDlistToIOVList(const nixl_xfer_dlist_t &dlist) {
    std::vector<xferBenchIOV> iov_list;
    for (const auto &desc : dlist) {
        iov_list.emplace_back(desc.addr, desc.len, desc.devId);
    }
    return iov_list;
}

// Convert vector of xferBenchIOV to nixl_xfer_dlist_t
static void iovListToNixlXferDlist(const std::vector<xferBenchIOV> &iov_list,
                                  nixl_xfer_dlist_t &dlist) {
    nixlBasicDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        dlist.addDesc(desc);
    }
}

std::optional<xferBenchIOV> xferBenchNixlWorker::initBasicDescDram(size_t buffer_size, int mem_dev_id) {
    void *addr;

    addr = calloc(1, buffer_size);
    if (!addr) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of DRAM memory" << std::endl;
        return std::nullopt;
    }

    if (isInitiator()) {
        memset(addr, XFERBENCH_INITIATOR_BUFFER_ELEMENT, buffer_size);
    } else if (isTarget()) {
        memset(addr, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);
    }

    // TODO: Does device id need to be set for DRAM?
    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, mem_dev_id);
}

#if HAVE_CUDA
static std::optional<xferBenchIOV> getVramDesc(int devid, size_t buffer_size,
                                 bool isInit)
{
    void *addr;

    CHECK_CUDA_ERROR(cudaSetDevice(devid), "Failed to set device");
#if !USE_VMM
    CHECK_CUDA_ERROR(cudaMalloc(&addr, buffer_size), "Failed to allocate CUDA buffer");
    if (isInit) {
        CHECK_CUDA_ERROR(cudaMemset(addr, XFERBENCH_INITIATOR_BUFFER_ELEMENT, buffer_size), "Failed to set device");

    } else {
        CHECK_CUDA_ERROR(cudaMemset(addr, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size), "Failed to set device");
    }
#else
    CUdeviceptr addr = 0;
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access = {};

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.location.id = devid;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;

    // Get the allocation granularity
    CHECK_CUDA_DRIVER_ERROR(cuMemGetAllocationGranularity(&granularity,
                         &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                         "Failed to get allocation granularity");
    std::cout << "Granularity: " << granularity << std::endl;

    padded_size = ROUND_UP(buffer_size, granularity);
    CHECK_CUDA_DRIVER_ERROR(cuMemCreate(&handle, padded_size, &prop, 0),
                         "Failed to create allocation");

    // Reserve the memory address
    CHECK_CUDA_DRIVER_ERROR(cuMemAddressReserve(&addr, padded_size,
                         granularity, 0, 0), "Failed to reserve address");

    // Map the memory
    CHECK_CUDA_DRIVER_ERROR(cuMemMap(addr, padded_size, 0, handle, 0),
                         "Failed to map memory");

    std::cout << "Address: " << std::hex << std::showbase << addr
              << " Buffer size: " << std::dec << buffer_size
              << " Padded size: " << std::dec << padded_size << std::endl;
    // Set the memory access rights
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = devid;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA_DRIVER_ERROR(cuMemSetAccess(addr, buffer_size, &access, 1),
        "Failed to set access");

    // Set memory content based on role
    if (isInit) {
        CHECK_CUDA_DRIVER_ERROR(cuMemsetD8(addr, XFERBENCH_INITIATOR_BUFFER_ELEMENT, buffer_size),
            "Failed to set device memory to XFERBENCH_INITIATOR_BUFFER_ELEMENT");
    } else {
        CHECK_CUDA_DRIVER_ERROR(cuMemsetD8(addr, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size),
            "Failed to set device memory to XFERBENCH_TARGET_BUFFER_ELEMENT");
    }
#endif /* !USE_VMM */

    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, devid);
}

std::optional<xferBenchIOV> xferBenchNixlWorker::initBasicDescVram(size_t buffer_size, int mem_dev_id) {
    if (IS_PAIRWISE_AND_SG()) {
        int devid = rt->getRank();

        if (isTarget()) {
            devid -= xferBenchConfig::num_initiator_dev;
        }

        if (devid != mem_dev_id) {
            return std::nullopt;
        }
    }

    return getVramDesc(mem_dev_id, buffer_size, isInitiator());
}
#endif /* HAVE_CUDA */

static std::vector<int> createFileFds(std::string name, bool is_gds) {
    std::vector<int> fds;
    int flags = O_RDWR | O_CREAT;
    int num_files = xferBenchConfig::num_files;
    std::string file_path, file_name_prefix;

    if (xferBenchConfig::storage_enable_direct) {
        flags |= O_DIRECT;
    }
    if (is_gds) {
        file_path = xferBenchConfig::gds_filepath != "" ?
                    xferBenchConfig::gds_filepath :
                    std::filesystem::current_path().string();
        file_name_prefix = "/nixlbench_gds_test_file_";
    } else {  // POSIX
        file_path = xferBenchConfig::posix_filepath != "" ?
                    xferBenchConfig::posix_filepath :
                    std::filesystem::current_path().string();
        file_name_prefix = "/nixlbench_posix_test_file_";
    }

    for (int i = 0; i < num_files; i++) {
        std::string file_name = file_path + file_name_prefix + name + "_" + std::to_string(i);
        std::cout << "Creating " << (is_gds ? "GDS" : "POSIX") << " file: " << file_name << std::endl;
        int fd = open(file_name.c_str(), flags, 0744);
        if (fd < 0) {
            std::cerr << "Failed to open file: " << file_name << " with error: "
                      << strerror(errno) << std::endl;
            for (int j = 0; j < i; j++) {
                close(fds[j]);
            }
            return {};
        }
        fds.push_back(fd);
    }
    return fds;
}

std::optional<xferBenchIOV> xferBenchNixlWorker::initBasicDescFile(size_t buffer_size, int fd, int mem_dev_id) {
    auto ret = std::optional<xferBenchIOV>(std::in_place, (uintptr_t)gds_running_ptr, buffer_size, fd);
    // Fill up with data
    void *buf = (void *)malloc(buffer_size);
    if (!buf) {
        std::cerr << "Failed to allocate " << buffer_size
                  << " bytes of memory" << std::endl;
        return std::nullopt;
    }
    // File is always initialized with XFERBENCH_TARGET_BUFFER_ELEMENT
    memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);
    int rc = pwrite(fd, buf, buffer_size, gds_running_ptr);
    if (rc < 0) {
        std::cerr << "Failed to write to file: " << fd
                  << " with error: " << strerror(errno) << std::endl;
        return std::nullopt;
    }
    free(buf);

    gds_running_ptr += (buffer_size * mem_dev_id);

    return ret;
}

void xferBenchNixlWorker::cleanupBasicDescDram(xferBenchIOV &iov) {
    free((void *)iov.addr);
}

#if HAVE_CUDA
void xferBenchNixlWorker::cleanupBasicDescVram(xferBenchIOV &iov) {
    CHECK_CUDA_ERROR(cudaSetDevice(iov.devId), "Failed to set device");
#if !USE_VMM
    CHECK_CUDA_ERROR(cudaFree((void *)iov.addr), "Failed to deallocate CUDA buffer");
#else
    CHECK_CUDA_DRIVER_ERROR(cuMemUnmap(iov.addr, iov.len),
                         "Failed to unmap memory");
    CHECK_CUDA_DRIVER_ERROR(cuMemRelease(handle),
                         "Failed to release memory");
    CHECK_CUDA_DRIVER_ERROR(cuMemAddressFree(iov.addr, padded_size), "Failed to free reserved address");
#endif
}
#endif /* HAVE_CUDA */

void xferBenchNixlWorker::cleanupBasicDescFile(xferBenchIOV &iov) {
    close(iov.devId);
}

std::vector<std::vector<xferBenchIOV>> xferBenchNixlWorker::allocateMemory(int num_lists) {
    std::vector<std::vector<xferBenchIOV>> iov_lists;
    size_t i, buffer_size, num_devices = 0;
    nixl_opt_args_t opt_args;

    if (isInitiator()) {
        num_devices = xferBenchConfig::num_initiator_dev;
    } else if (isTarget()) {
        num_devices = xferBenchConfig::num_target_dev;
    }
    buffer_size = xferBenchConfig::total_buffer_size / (num_devices * num_lists);

    opt_args.backends.push_back(backend_engine);

    if (XFERBENCH_BACKEND_GDS == xferBenchConfig::backend ||
        XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend) {
        bool is_gds = XFERBENCH_BACKEND_GDS == xferBenchConfig::backend;
        remote_fds = createFileFds(getName(), is_gds);
        if (remote_fds.empty()) {
            std::cerr << "Failed to create " << ((is_gds) ? "GDS" : "POSIX") << " file" << std::endl;
            exit(EXIT_FAILURE);
        }
        for (int list_idx = 0; list_idx < num_lists; list_idx++) {
            std::vector<xferBenchIOV> iov_list;
            for (i = 0; i < num_devices; i++) {
                std::optional<xferBenchIOV> basic_desc;
                basic_desc = initBasicDescFile(buffer_size, remote_fds[0], i);
                if (basic_desc) {
                    iov_list.push_back(basic_desc.value());
                }
            }
            nixl_reg_dlist_t desc_list(FILE_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args),
                        "registerMem failed");
            remote_iovs.push_back(iov_list);
        }
        // Reset the running pointer to 0
        gds_running_ptr = 0x0;
    }

    for (int list_idx = 0; list_idx < num_lists; list_idx++) {
        std::vector<xferBenchIOV> iov_list;
        for (i = 0; i < num_devices; i++) {
            std::optional<xferBenchIOV> basic_desc;

            switch (seg_type) {
            case DRAM_SEG:
                basic_desc = initBasicDescDram(buffer_size, i);
                break;
#if HAVE_CUDA
            case VRAM_SEG:
                basic_desc = initBasicDescVram(buffer_size, i);
                break;
#endif
            default:
                std::cerr << "Unsupported mem type: " << seg_type << std::endl;
                exit(EXIT_FAILURE);
            }

            if (basic_desc) {
                iov_list.push_back(basic_desc.value());
            }
        }

        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args),
                       "registerMem failed");
        iov_lists.push_back(iov_list);
    }

    return iov_lists;
}

void xferBenchNixlWorker::deallocateMemory(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    nixl_opt_args_t opt_args;

    opt_args.backends.push_back(backend_engine);
    for (auto &iov_list: iov_lists) {
        for (auto &iov: iov_list) {
            switch (seg_type) {
            case DRAM_SEG:
                cleanupBasicDescDram(iov);
                break;
#if HAVE_CUDA
            case VRAM_SEG:
                cleanupBasicDescVram(iov);
                break;
#endif
            default:
                std::cerr << "Unsupported mem type: " << seg_type << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args),
                         "deregisterMem failed");
    }

    if (XFERBENCH_BACKEND_GDS == xferBenchConfig::backend ||
        XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend) {
        for (auto &iov_list: remote_iovs) {
            for (auto &iov: iov_list) {
                cleanupBasicDescFile(iov);
            }
            nixl_reg_dlist_t desc_list(FILE_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args),
                             "deregisterMem failed");
        }
    }
}

int xferBenchNixlWorker::exchangeMetadata() {
    int meta_sz, ret = 0;

    if (XFERBENCH_BACKEND_GDS == xferBenchConfig::backend ||
        XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend) {
        return 0;
    }

    if (isTarget()) {
        std::string local_metadata;
        const char *buffer;
        int destrank;

        agent->getLocalMD(local_metadata);

        buffer = local_metadata.data();
        meta_sz = local_metadata.size();

        if (IS_PAIRWISE_AND_SG()) {
            destrank = rt->getRank() - xferBenchConfig::num_target_dev;
            //XXX: Fix up the rank, depends on processes distributed on hosts
            //assumes placement is adjacent ranks to same node
        } else {
            destrank = 0;
        }
        rt->sendInt(&meta_sz, destrank);
        rt->sendChar((char *)buffer, meta_sz, destrank);
    } else if (isInitiator()) {
        char * buffer;
        std::string remote_agent;
        int srcrank;

        if (IS_PAIRWISE_AND_SG()) {
            srcrank = rt->getRank() + xferBenchConfig::num_initiator_dev;
            //XXX: Fix up the rank, depends on processes distributed on hosts
            //assumes placement is adjacent ranks to same node
        } else {
            srcrank = 1;
        }
        rt->recvInt(&meta_sz, srcrank);
        buffer = (char *)calloc(meta_sz, sizeof(*buffer));
        rt->recvChar((char *)buffer, meta_sz, srcrank);

        std::string remote_metadata(buffer, meta_sz);
        agent->loadRemoteMD(remote_metadata, remote_agent);
        if("" == remote_agent) {
            std::cerr << "NIXL: loadMetadata failed" << std::endl;
        }
        free(buffer);
    }
    return ret;
}

std::vector<std::vector<xferBenchIOV>>
xferBenchNixlWorker::exchangeIOV(const std::vector<std::vector<xferBenchIOV>> &local_iovs) {
    std::vector<std::vector<xferBenchIOV>> res;
    int desc_str_sz;

    // Special case for GDS
    if (XFERBENCH_BACKEND_GDS == xferBenchConfig::backend ||
        XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend) {
        for (auto &iov_list: local_iovs) {
            std::vector<xferBenchIOV> remote_iov_list;
            for (auto &iov: iov_list) {
                std::optional<xferBenchIOV> basic_desc;
                basic_desc = initBasicDescFile(iov.len, remote_fds[0], iov.devId);
                if (basic_desc) {
                    remote_iov_list.push_back(basic_desc.value());
                }
            }
            res.push_back(remote_iov_list);
        }
    } else {
        for (const auto &local_iov: local_iovs) {
            nixlSerDes ser_des;
            nixl_xfer_dlist_t local_desc(seg_type);

            iovListToNixlXferDlist(local_iov, local_desc);

            if (isTarget()) {
                const char *buffer;
                int destrank;

                local_desc.serialize(&ser_des);
                std::string desc_str = ser_des.exportStr();
                buffer = desc_str.data();
                desc_str_sz = desc_str.size();

                if (IS_PAIRWISE_AND_SG()) {
                    destrank = rt->getRank() - xferBenchConfig::num_target_dev;
                    //XXX: Fix up the rank, depends on processes distributed on hosts
                    //assumes placement is adjacent ranks to same node
                } else {
                    destrank = 0;
                }
                rt->sendInt(&desc_str_sz, destrank);
                rt->sendChar((char *)buffer, desc_str_sz, destrank);
            } else if (isInitiator()) {
                char *buffer;
                int srcrank;

                if (IS_PAIRWISE_AND_SG()) {
                    srcrank = rt->getRank() + xferBenchConfig::num_initiator_dev;
                    //XXX: Fix up the rank, depends on processes distributed on hosts
                    //assumes placement is adjacent ranks to same node
                } else {
                    srcrank = 1;
                }
                rt->recvInt(&desc_str_sz, srcrank);
                buffer = (char *)calloc(desc_str_sz, sizeof(*buffer));
                rt->recvChar((char *)buffer, desc_str_sz, srcrank);

                std::string desc_str(buffer, desc_str_sz);
                ser_des.importStr(desc_str);

                nixl_xfer_dlist_t remote_desc(&ser_des);
                res.emplace_back(nixlXferDlistToIOVList(remote_desc));
            }
        }
    }
    // Ensure all processes have completed the exchange with a barrier/sync
    synchronize();
    return res;
}

static int execTransfer(nixlAgent *agent,
                        const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                        const std::vector<std::vector<xferBenchIOV>> &remote_iovs,
                        const nixl_xfer_op_t op,
                        const int num_iter,
                        const int num_threads)
{
    int ret = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const auto &local_iov = local_iovs[tid];
        const auto &remote_iov = remote_iovs[tid];

        // TODO: fetch local_desc and remote_desc directly from config
        nixl_xfer_dlist_t local_desc(GET_SEG_TYPE(true));
        nixl_xfer_dlist_t remote_desc(GET_SEG_TYPE(false));

        if ((XFERBENCH_BACKEND_GDS == xferBenchConfig::backend) ||
            (XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend)) {
            remote_desc = nixl_xfer_dlist_t(FILE_SEG);
        }

        iovListToNixlXferDlist(local_iov, local_desc);
        iovListToNixlXferDlist(remote_iov, remote_desc);

        nixl_opt_args_t params;
        nixl_b_params_t b_params;
        bool error = false;
        nixlXferReqH *req;
        nixl_status_t rc;
        std::string target;

        if (XFERBENCH_BACKEND_GDS == xferBenchConfig::backend) {
            target = "initiator";
        } else if (XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend) {
            target = "initiator";
        } else {
            params.notifMsg = "0xBEEF";
            params.hasNotif = true;
            target = "target";
        }

        CHECK_NIXL_ERROR(agent->createXferReq(op, local_desc, remote_desc, target,
                                            req, &params), "createTransferReq failed");

        for (int i = 0; i < num_iter && !error; i++) {
            rc = agent->postXferReq(req);
            if (NIXL_ERR_BACKEND == rc) {
                std::cout << "NIXL postRequest failed" << std::endl;
                error = true;
            } else {
                do {
                    /* XXX agent isn't const because the getXferStatus() is not const  */
                    rc = agent->getXferStatus(req);
                    if (NIXL_ERR_BACKEND == rc) {
                        std::cout << "NIXL getStatus failed" << std::endl;
                        error = true;
                        break;
                    }
                } while (NIXL_SUCCESS != rc);
            }
        }

        agent->releaseXferReq(req);
        if (error) {
            std::cout << "NIXL releaseXferReq failed" << std::endl;
            ret = -1;
        }
    }

    return ret;
}

std::variant<double, int> xferBenchNixlWorker::transfer(size_t block_size,
                                               const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                                               const std::vector<std::vector<xferBenchIOV>> &remote_iovs) {
    int num_iter = xferBenchConfig::num_iter / xferBenchConfig::num_threads;
    int skip = xferBenchConfig::warmup_iter / xferBenchConfig::num_threads;
    struct timeval t_start, t_end;
    double total_duration = 0.0;
    int ret = 0;
    nixl_xfer_op_t xfer_op = XFERBENCH_OP_READ == xferBenchConfig::op_type ? NIXL_READ : NIXL_WRITE;
    // int completion_flag = 1;

    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= LARGE_BLOCK_SIZE_ITER_FACTOR;
        num_iter /= LARGE_BLOCK_SIZE_ITER_FACTOR;
    }

    ret = execTransfer(agent, local_iovs, remote_iovs, xfer_op, skip, xferBenchConfig::num_threads);
    if (ret < 0) {
        return std::variant<double, int>(ret);
    }

    // Synchronize to ensure all processes have completed the warmup (iter and polling)
    synchronize();

    gettimeofday(&t_start, nullptr);

    ret = execTransfer(agent, local_iovs, remote_iovs, xfer_op, num_iter, xferBenchConfig::num_threads);

    gettimeofday(&t_end, nullptr);
    total_duration += (((t_end.tv_sec - t_start.tv_sec) * 1e6) +
                       (t_end.tv_usec - t_start.tv_usec)); // In us

    return ret < 0 ? std::variant<double, int>(ret) : std::variant<double, int>(total_duration);
}

void xferBenchNixlWorker::poll(size_t block_size) {
    nixl_notifs_t notifs;
    int skip = 0, num_iter = 0, total_iter = 0;

    skip = xferBenchConfig::warmup_iter;
    num_iter = xferBenchConfig::num_iter;
    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= LARGE_BLOCK_SIZE_ITER_FACTOR;
        num_iter /= LARGE_BLOCK_SIZE_ITER_FACTOR;
    }
    total_iter = skip + num_iter;

    /* Ensure warmup is done*/
    while (skip != int(notifs["initiator"].size())) {
        agent->getNotifs(notifs);
    }
    synchronize();

    /* Polling for actual iterations*/
    while (total_iter != int(notifs["initiator"].size())) {
        agent->getNotifs(notifs);
    }
}

int xferBenchNixlWorker::synchronizeStart() {
    if (IS_PAIRWISE_AND_SG()) {
    	std::cout << "Waiting for all processes to start... (expecting "
    	          << rt->getSize() << " total: "
		  << xferBenchConfig::num_initiator_dev << " initiators and "
    	          << xferBenchConfig::num_target_dev << " targets)" << std::endl;
    } else {
    	std::cout << "Waiting for all processes to start... (expecting "
    	          << rt->getSize() << " total" << std::endl;
    }
    if (rt) {
        int ret = rt->barrier("start_barrier");
        if (ret != 0) {
            std::cerr << "Failed to synchronize at start barrier" << std::endl;
            return -1;
        }
        std::cout << "All processes are ready to proceed" << std::endl;
        return 0;
    }
    return -1;
}
