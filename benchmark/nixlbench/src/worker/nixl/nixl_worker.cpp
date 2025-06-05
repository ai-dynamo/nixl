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
#include <time.h>
#include <utils/serdes/serdes.h>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

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

thread_local double xferBenchNixlWorker::min_latency = 0.0;
thread_local double xferBenchNixlWorker::max_latency = 0.0;
thread_local double xferBenchNixlWorker::median_latency = 0.0;
thread_local double xferBenchNixlWorker::p95_latency = 0.0;
thread_local double xferBenchNixlWorker::p99_latency = 0.0;
thread_local double xferBenchNixlWorker::avg_latency = 0.0;
thread_local double xferBenchNixlWorker::total_duration = 0.0;
thread_local size_t xferBenchNixlWorker::num_operations = 0;

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
            backend_params["use_aio"] = "true";
            backend_params["use_uring"] = "false";
        } else if (xferBenchConfig::posix_api_type == XFERBENCH_POSIX_API_URING) {
            backend_params["use_aio"] = "false";
            backend_params["use_uring"] = "true";
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
    void *addr = nullptr;

    if (xferBenchConfig::storage_enable_direct && xferBenchConfig::backend == XFERBENCH_BACKEND_POSIX) {
        // Allocate 4KB-aligned memory for direct I/O
        int ret = posix_memalign(&addr, 4096, buffer_size);
        if (ret != 0 || addr == nullptr) {
            std::cerr << "Failed to allocate " << buffer_size << " bytes of aligned DRAM memory for direct I/O" << std::endl;
            return std::nullopt;
        }
    } else {
        addr = calloc(1, buffer_size);
        if (!addr) {
            std::cerr << "Failed to allocate " << buffer_size << " bytes of DRAM memory" << std::endl;
            return std::nullopt;
        }
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

    void *buf = (void *)malloc(buffer_size);
    if (!buf) {
        std::cerr << "Failed to allocate " << buffer_size
                  << " bytes of memory" << std::endl;
        return std::nullopt;
    }
    if (!buf) {
        std::cerr << "Failed to allocate " << buffer_size
                  << " bytes of memory" << std::endl;
        return std::nullopt;
    }

    // File is always initialized with XFERBENCH_TARGET_BUFFER_ELEMENT
    memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);
    
    // Write the initialization data to the file
    int rc = pwrite(fd, buf, buffer_size, gds_running_ptr);
    if (rc < 0) {
        std::cerr << "Failed to write to file: " << fd
                  << " with error: " << strerror(errno) << std::endl;
        free(buf);
        return std::nullopt;
    }
        
    // Free the temporary buffer - no longer needed
    // free(buf);
    
    // Update the running pointer for the next operation
    gds_running_ptr += buffer_size;
    
    // return std::optional<xferBenchIOV>(std::in_place, addr, buffer_size, fd);
    return ret;
}

std::optional<xferBenchIOV> xferBenchNixlWorker::initDirectIODescFile(size_t buffer_size, int fd, int mem_dev_id) {
    long page_size = sysconf(_SC_PAGESIZE);

    if (page_size == 0) {
        std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
        return std::nullopt;
    }

    if (buffer_size % page_size != 0) {
        buffer_size = ((buffer_size + page_size - 1) / page_size) * page_size;
        std::cout << "Adjusted transfer size to " << buffer_size << " bytes for O_DIRECT alignment" << std::endl;
    }

    void* buf;
    int result = posix_memalign(&buf, (int)page_size, buffer_size);
    if (result != 0) {
        std::cerr << "Error: " << strerror(result) << std::endl;
        std::cerr << "Failed to allocate " << buffer_size
                  << " bytes of memory" << std::endl;
        return std::nullopt;
    }
    if (!buf) {
        std::cerr << "Failed to allocate " << buffer_size
                  << " bytes of memory" << std::endl;
        return std::nullopt;
    }

    // File is always initialized with XFERBENCH_TARGET_BUFFER_ELEMENT
    memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);
    
    // Make sure running pointer is also aligned for direct I/O
    gds_running_ptr = ((gds_running_ptr + int (page_size) - 1 ) / int (page_size)) * int (page_size);
    // Write the initialization data to the file
    int rc = pwrite(fd, buf, buffer_size, gds_running_ptr);
    if (rc < 0) {
        std::cerr << "Failed to write to file: " << fd
                  << " with error: " << strerror(errno) << std::endl;
        free(buf);
        return std::nullopt;
    }

    uintptr_t addr = static_cast<uintptr_t>(gds_running_ptr);

    gds_running_ptr += buffer_size;
    
    return std::optional<xferBenchIOV>(std::in_place, addr, buffer_size, fd);
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

bool xferBenchNixlWorker::initializeFileSeg(int num_lists, std::vector<std::vector<xferBenchIOV>> &remote_iovs) {
    nixl_opt_args_t opt_args;
    opt_args.backends.push_back(backend_engine);
    size_t buffer_size, num_devices = 0;
    bool is_gds = XFERBENCH_BACKEND_GDS == xferBenchConfig::backend;
    
    // Calculate buffer size
    if (isInitiator()) {
        num_devices = xferBenchConfig::num_initiator_dev;
    } else if (isTarget()) {
        num_devices = xferBenchConfig::num_target_dev;
    }
    buffer_size = xferBenchConfig::total_buffer_size / (num_devices * num_lists);
    
    // Print the backend type
    std::cout << "Using " << (is_gds ? "GDS" : "POSIX") << " backend" 
              << (xferBenchConfig::storage_enable_direct ? " with direct I/O" : "") << std::endl;
    
    // Create file descriptors
    remote_fds = createFileFds(getName(), is_gds);
    if (remote_fds.empty()) {
        std::cerr << "Failed to create " << ((is_gds) ? "GDS" : "POSIX") << " file" << std::endl;
        return false;
    }
    
    // For each thread/list, create a set of descriptors
    for (int list_idx = 0; list_idx < num_lists; list_idx++) {
        std::vector<xferBenchIOV> iov_list;
        for (size_t i = 0; i < num_devices; i++) {
            std::optional<xferBenchIOV> basic_desc;
            
            // Use the appropriate function based on direct I/O setting
            if (xferBenchConfig::storage_enable_direct) {
                std::cout << "Using direct I/O descriptor for device " << i << std::endl;
                basic_desc = initDirectIODescFile(buffer_size, remote_fds[0], i);
            } else {
                std::cout << "Using standard descriptor for device " << i << std::endl;
                basic_desc = initBasicDescFile(buffer_size, remote_fds[0], i);
            }
            
            if (basic_desc) {
                iov_list.push_back(basic_desc.value());
            } else {
                std::cerr << "Failed to create descriptor for device " << i << std::endl;
                return false;
            }
        }
        
        // Register with NIXL
        nixl_reg_dlist_t desc_list(FILE_SEG);
        iovListToNixlRegDlist(iov_list, desc_list);
        nixl_status_t status = agent->registerMem(desc_list, &opt_args);
        if (status != NIXL_SUCCESS) {
            std::cerr << "registerMem failed with status: " << status << std::endl;
            return false;
        }
        
        remote_iovs.push_back(iov_list);
    }
    
    // Reset the running pointer to 0
    gds_running_ptr = 0x0;
    return true;
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

    // For direct I/O, ensure buffer size is a multiple of page size
    if (xferBenchConfig::storage_enable_direct) {
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size > 0 && buffer_size % page_size != 0) {
            size_t aligned_size = ((buffer_size + page_size - 1) / page_size) * page_size;
            std::cout << "Adjusting buffer size from " << buffer_size << " to " << aligned_size 
                      << " for direct I/O alignment" << std::endl;
            buffer_size = aligned_size;
        }
    }

    opt_args.backends.push_back(backend_engine);

    if (XFERBENCH_BACKEND_GDS == xferBenchConfig::backend ||
        XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend) {
        // Initialize file segment
        if (!initializeFileSeg(num_lists, remote_iovs)) {
            exit(EXIT_FAILURE);
        }
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

    // // Clean up direct I/O buffers if they were used
    // if (xferBenchConfig::storage_enable_direct) {
    //     size_t buffers_freed = 0;
    //     for (auto& buffer_list : direct_io_buffers) {
    //         for (void* buf : buffer_list) {
    //             if (buf) {
    //                 buffers_freed++;
    //                 free(buf);
    //             }
    //         }
    //     }
    //     direct_io_buffers.clear();
    // }
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

    if (XFERBENCH_BACKEND_GDS == xferBenchConfig::backend ||
        XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend) {
        for (auto &iov_list: local_iovs) {
            std::vector<xferBenchIOV> remote_iov_list;
            for (auto &iov: iov_list) {
                std::optional<xferBenchIOV> basic_desc;
                if (xferBenchConfig::storage_enable_direct) {
                    basic_desc = initDirectIODescFile(iov.len, remote_fds[0], iov.devId);
                } else {
                    basic_desc = initBasicDescFile(iov.len, remote_fds[0], iov.devId);
                }
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
                        const int num_threads,
                        std::vector<double> &latencies)
{
    int ret = 0;
    std::vector<std::vector<double>> thread_latencies(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const auto &local_iov = local_iovs[tid];
        const auto &remote_iov = remote_iovs[tid];

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


        std::vector<double> local_latencies;

        local_latencies.reserve(num_iter);
        for (int i = 0; i < num_iter && !error; i++) {

            CHECK_NIXL_ERROR(agent->createXferReq(op, local_desc, remote_desc, target,
                                                    req, &params), "createTransferReq failed");
            auto start_time = std::chrono::high_resolution_clock::now();
            rc = agent->postXferReq(req);
            if (NIXL_ERR_BACKEND == rc) {
                std::cout << "NIXL postRequest failed" << std::endl;
                error = true;
            } else {
                do {
                    rc = agent->getXferStatus(req);
                    if (NIXL_ERR_BACKEND == rc) {
                        std::cout << "NIXL getStatus failed" << std::endl;
                        error = true;
                        break;
                    }
                } while (NIXL_SUCCESS != rc);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            double latency_us = duration / 1000.0; // Convert to microseconds
            local_latencies.push_back(latency_us);
            nixl_status_t release_rc = agent->releaseXferReq(req);
            if (NIXL_SUCCESS != release_rc) {
                std::cerr << "Warning: Failed to release transfer request" << std::endl;
            }
        }


        if (error) {
            std::cout << "NIXL transfer operations failed, cleaning up" << std::endl;
            ret = -1;
        }
        thread_latencies[tid] = std::move(local_latencies);
    }
    
    for (const auto& thread_lat : thread_latencies) {
        latencies.insert(latencies.end(), thread_lat.begin(), thread_lat.end());
    }
    return ret;
}

struct LatencyStats {
    double min;
    double max;
    double avg;
    double median;
    double p90;
    double p95;
    double p99;
    double std_dev;
    double total_duration;
};

static LatencyStats calculateLatencyStats(const std::vector<double>& latencies) {
    LatencyStats stats = {0};
    
    if (latencies.empty()) {
        return stats;
    }
    
    std::vector<double> sorted_latencies = latencies;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    
    double sum = 0.0;
    int count = 0;
    for (double lat : latencies) {
        sum += lat;
        count++;
    }

    size_t median_idx = sorted_latencies.size() / 2;
    size_t p90_idx = sorted_latencies.size() * 90 / 100;
    size_t p95_idx = sorted_latencies.size() * 95 / 100;
    size_t p99_idx = sorted_latencies.size() * 99 / 100;
    
    median_idx = std::min(median_idx, sorted_latencies.size() - 1);
    p90_idx = std::min(p90_idx, sorted_latencies.size() - 1);
    p95_idx = std::min(p95_idx, sorted_latencies.size() - 1);
    p99_idx = std::min(p99_idx, sorted_latencies.size() - 1);
    
    stats.min = sorted_latencies.front();
    stats.max = sorted_latencies.back();
    stats.total_duration = sum;
    stats.avg = sum / count;
    stats.median = sorted_latencies[median_idx];
    stats.p90 = sorted_latencies[p90_idx];
    stats.p95 = sorted_latencies[p95_idx];
    stats.p99 = sorted_latencies[p99_idx];
    
    double variance = 0.0;
    for (double lat : sorted_latencies) {
        variance += (lat - stats.avg) * (lat - stats.avg);
    }
    stats.std_dev = std::sqrt(variance / sorted_latencies.size());
    
    return stats;
}

std::variant<double, int> xferBenchNixlWorker::transfer(size_t block_size,
                                               const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                                               const std::vector<std::vector<xferBenchIOV>> &remote_iovs) {
    int num_iter = xferBenchConfig::num_iter / xferBenchConfig::num_threads;
    int skip = xferBenchConfig::warmup_iter / xferBenchConfig::num_threads;
    int ret = 0;
    nixl_xfer_op_t xfer_op = XFERBENCH_OP_READ == xferBenchConfig::op_type ? NIXL_READ : NIXL_WRITE;
    std::vector<double> latencies;

    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= LARGE_BLOCK_SIZE_ITER_FACTOR;
        num_iter /= LARGE_BLOCK_SIZE_ITER_FACTOR;
    }

    std::vector<double> warmup_latencies;
    ret = execTransfer(agent, local_iovs, remote_iovs, xfer_op, skip, xferBenchConfig::num_threads, warmup_latencies);
    if (ret < 0) {
        return std::variant<double, int>(ret);
    }

    // Synchronize to ensure all processes have completed the warmup (iter and polling)
    synchronize();
    
    latencies.reserve(num_iter * xferBenchConfig::num_threads);

    ret = execTransfer(agent, local_iovs, remote_iovs, xfer_op, num_iter, xferBenchConfig::num_threads, latencies);
    if (ret < 0) {
        return std::variant<double, int>(ret);
    }
    LatencyStats stats = calculateLatencyStats(latencies);

    if (ret < 0) {
        return std::variant<double, int>(ret);
    } else {
        min_latency = stats.min;
        max_latency = stats.max;
        median_latency = stats.median;
        p95_latency = stats.p95;
        p99_latency = stats.p99;
        avg_latency = stats.avg;
        total_duration = stats.total_duration;
        num_operations = latencies.size();
        return std::variant<double, int>(stats.total_duration);
    }
}

void xferBenchNixlWorker::getLastLatencyStats(double& min_lat, double& med_lat, double& max_lat, double& p95_lat, 
                                             double& p99_lat, double& avg_lat, 
                                             double& total_dur, size_t& num_ops) {
    min_lat = min_latency;
    med_lat = median_latency;
    max_lat = max_latency;
    p95_lat = p95_latency;
    p99_lat = p99_latency;
    avg_lat = avg_latency;
    total_dur = total_duration;
    num_ops = num_operations;
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

