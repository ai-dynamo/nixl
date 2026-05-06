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

#include "worker/nixl/nixl_worker.h"
#include "benchmark_config.h"
#include "worker/nixl/nixl_backend_params.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include "utils/neuron.h"
#include "utils/utils.h"
#include <unistd.h>
#include <utility>
#include <sys/time.h>
#include <sys/stat.h>
#include <utils/serdes/serdes.h>
#include <omp.h>

#define ROUND_UP(value, granularity) \
    ((((value) + (granularity) - 1) / (granularity)) * (granularity))

#define CHECK_NIXL_ERROR(result, message)                                                       \
    do {                                                                                        \
        if (0 != result) {                                                                      \
            std::cerr << "NIXL: " << message << " (Error code: " << result << ")" << std::endl; \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

#if HAVE_CUDA
#define HANDLE_VRAM_SEGMENT(_seg_type) _seg_type = VRAM_SEG;
#else
#define HANDLE_VRAM_SEGMENT(_seg_type)                                        \
    std::cerr << "VRAM segment type not supported without CUDA" << std::endl; \
    std::exit(EXIT_FAILURE);
#endif

namespace {

nixl_mem_t
getSegType(const nixlbench::benchmarkConfig &config, bool is_initiator) {
    const std::string &seg_type_str =
        is_initiator ? config.transfer.initiator_seg_type : config.transfer.target_seg_type;
    if (seg_type_str == XFERBENCH_SEG_TYPE_DRAM) {
        return DRAM_SEG;
    }
    if (seg_type_str == XFERBENCH_SEG_TYPE_VRAM) {
        nixl_mem_t seg_type;
        HANDLE_VRAM_SEGMENT(seg_type);
        return seg_type;
    }

    std::cerr << "Invalid segment type: " << seg_type_str << std::endl;
    exit(EXIT_FAILURE);
}

nixl_mem_t
getLegacySegType(const xferBenchConfig &config, bool is_initiator) {
    const std::string &seg_type_str =
        is_initiator ? config.initiator_seg_type : config.target_seg_type;
    if (seg_type_str == XFERBENCH_SEG_TYPE_DRAM) {
        return DRAM_SEG;
    }
    if (seg_type_str == XFERBENCH_SEG_TYPE_VRAM) {
        nixl_mem_t seg_type;
        HANDLE_VRAM_SEGMENT(seg_type);
        return seg_type;
    }

    std::cerr << "Invalid segment type: " << seg_type_str << std::endl;
    exit(EXIT_FAILURE);
}

void
printNixlBackendParams(const nixlbench::benchmarkConfig &config,
                       const nixl_b_params_t &backend_params,
                       const std::vector<std::string> &devices,
                       int rank,
                       const std::string &worker_name,
                       const std::vector<GusliDeviceConfig> &gusli_devices) {
    char hostname[256];
    auto paramValue = [&backend_params](const std::string &name) -> std::string {
        const auto iter = backend_params.find(name);
        return iter == backend_params.end() ? "" : iter->second;
    };

    if (config.backend.name == XFERBENCH_BACKEND_UCX) {
        if (gethostname(hostname, 256)) {
            std::cerr << "Failed to get hostname" << std::endl;
            exit(EXIT_FAILURE);
        }
        const auto device = backend_params.find("device_list");
        std::cout << "Init nixl worker, dev "
                  << ((devices[0] == "all" || device == backend_params.end()) ? "all" :
                                                                                device->second)
                  << " rank " << rank << ", type " << worker_name << ", hostname " << hostname
                  << std::endl;
    } else if (config.backend.name == XFERBENCH_BACKEND_LIBFABRIC) {
        if (gethostname(hostname, 256)) {
            std::cerr << "Failed to get hostname" << std::endl;
            exit(EXIT_FAILURE);
        }

        // We need to make sure the Neuron runtime is initialized before initializing libfabric,
        // otherwise the FI_HMEM_NEURON backend will not be created. This issue has been fixed
        // upstream: https://github.com/ofiwg/libfabric/pull/11804
        int nc_count = neuronCoreCount();

        std::cout << "Init nixl worker, dev " << ((devices[0] == "all") ? "all" : devices[rank])
                  << " rank " << rank << ", type " << worker_name << ", hostname " << hostname
                  << ", nc_count " << nc_count << std::endl;
    } else if (config.backend.name == XFERBENCH_BACKEND_GDS) {
        std::cout << "GDS backend" << std::endl;
        std::cout << "GDS batch pool size: " << paramValue("batch_pool_size") << std::endl;
        std::cout << "GDS batch limit: " << paramValue("batch_limit") << std::endl;
    } else if (config.backend.name == XFERBENCH_BACKEND_GDS_MT) {
        std::cout << "GDS_MT backend" << std::endl;
        std::cout << "GDS MT Num threads: " << paramValue("thread_count") << std::endl;
    } else if (config.backend.name == XFERBENCH_BACKEND_POSIX) {
        std::cout << "POSIX backend" << std::endl;
    } else if (config.backend.name == XFERBENCH_BACKEND_GPUNETIO) {
        std::cout << "GPUNETIO backend, network device " << paramValue("network_devices")
                  << " GPU device " << paramValue("gpu_devices") << " OOB interface "
                  << paramValue("oob_interface") << std::endl;
    } else if (config.backend.name == XFERBENCH_BACKEND_MOONCAKE) {
        std::cout << "Mooncake backend" << std::endl;
    } else if (config.backend.name == XFERBENCH_BACKEND_HF3FS) {
        std::cout << "HF3FS backend iopool_size " << paramValue("iopool_size") << std::endl;
    } else if (config.backend.name == XFERBENCH_BACKEND_OBJ) {
        if (backend_params.count("crtMinLimit") && std::stoull(paramValue("crtMinLimit")) > 0) {
            std::cout << "OBJ backend with S3 CRT client enabled for objects >= "
                      << paramValue("crtMinLimit") << " bytes" << std::endl;
        } else if (backend_params.count("accelerated") && paramValue("accelerated") == "true") {
            std::cout << "OBJ backend with S3 Accelerated client enabled";
            if (backend_params.count("type") && !paramValue("type").empty()) {
                std::cout << " (type: " << paramValue("type") << ")";
            }
            std::cout << std::endl;
        } else {
            std::cout << "OBJ backend with standard S3 enabled" << std::endl;
        }
    } else if (config.backend.name == XFERBENCH_BACKEND_GUSLI) {
        std::cout << "GUSLI backend initialized:" << std::endl;
        std::cout << "  Client name: " << paramValue("client_name") << std::endl;
        std::cout << "  Max simultaneous requests: " << paramValue("max_num_simultaneous_requests")
                  << std::endl;
        std::cout << "  Direct I/O: Enabled (required)" << std::endl;
        std::cout << "  Configured devices: " << gusli_devices.size() << std::endl;
        for (const auto &dev : gusli_devices) {
            std::cout << "    Device " << dev.device_id << " [" << dev.device_type
                      << "]: " << dev.device_path << " (" << dev.security_flags << ")"
                      << ", offset = " << dev.dev_offset << std::endl;
        }
    } else if (config.backend.name == XFERBENCH_BACKEND_UCCL) {
        std::cout << "UCCL backend" << std::endl;
    } else if (config.backend.name == XFERBENCH_BACKEND_AZURE_BLOB) {
        std::cout << "AZURE_BLOB backend" << std::endl;
    }
}

} // namespace

xferBenchNixlWorker::xferBenchNixlWorker(const nixlbench::benchmarkConfig &benchmark_config,
                                         std::vector<std::string> devices)
    : xferBenchWorker(benchmark_config) {
    seg_type = getSegType(benchmark_config, isInitiator());

    int rank;
    nixl_b_params_t backend_params;
    nixl_thread_sync_t sync_mode = benchmark_config.transfer.num_threads > 1 ?
        nixl_thread_sync_t::NIXL_THREAD_SYNC_RW :
        nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT;
    nixl_mem_list_t mems;

    rank = rt->getRank();

    nixlAgentConfig dev_meta;
    dev_meta.useProgThread = benchmark_config.worker.enable_progress_thread;
    dev_meta.syncMode = sync_mode;

    agent = new nixlAgent(name, dev_meta);

    CHECK_NIXL_ERROR(agent->getPluginParams(benchmark_config.backend.name, mems, backend_params),
                     "getPluginParams failed!");
    if (benchmark_config.backend.capabilities.requiresDirectStorage) {
        if (!config.storage_enable_direct) {
            std::cout << benchmark_config.backend.name
                      << " backend: Automatically enabling storage_enable_direct for direct I/O"
                      << std::endl;
            config.storage_enable_direct = true;
        }
    }
    gusli_devices = nixlbench::buildGusliDeviceConfigs(benchmark_config, isInitiator());
    backend_params = nixlbench::buildNixlBackendParams(
        benchmark_config, std::move(backend_params), devices, isInitiator(), rank, gusli_devices);
    printNixlBackendParams(benchmark_config, backend_params, devices, rank, name, gusli_devices);

    CHECK_NIXL_ERROR(agent->createBackend(benchmark_config.backend.name, backend_params, backend_engine),
                     "createBackend failed!");
}

xferBenchNixlWorker::~xferBenchNixlWorker() {
    delete rt;
    rt = nullptr;

    if (agent) {
        delete agent;
        agent = nullptr;
    }
}

// Convert vector of xferBenchIOV to nixl_reg_dlist_t
static void
iovListToNixlRegDlist(const std::vector<xferBenchIOV> &iov_list, nixl_reg_dlist_t &dlist) {
    nixlBlobDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        desc.metaInfo = iov.metaInfo;
        dlist.addDesc(desc);
    }
}

// Convert nixl_xfer_dlist_t to vector of xferBenchIOV
static std::vector<xferBenchIOV>
nixlXferDlistToIOVList(const nixl_xfer_dlist_t &dlist) {
    std::vector<xferBenchIOV> iov_list;
    for (const auto &desc : dlist) {
        iov_list.emplace_back(desc.addr, desc.len, desc.devId);
    }
    return iov_list;
}

// Convert vector of xferBenchIOV to nixl_xfer_dlist_t
static void
iovListToNixlXferDlist(const std::vector<xferBenchIOV> &iov_list, nixl_xfer_dlist_t &dlist) {
    nixlBasicDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        dlist.addDesc(desc);
    }
}

static bool
allocateXferMemory(const size_t page_size, size_t buffer_size, void **addr) {
    if (!addr) {
        std::cerr << "Invalid address" << std::endl;
        return false;
    }
    if (buffer_size == 0) {
        std::cerr << "Invalid buffer size" << std::endl;
        return false;
    }
    if (page_size == 0) {
        std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
        return false;
    }

    int rc = posix_memalign(addr, page_size, buffer_size);
    if (rc != 0 || !*addr) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of page-aligned DRAM memory"
                  << std::endl;
        return false;
    }
    memset(*addr, 0, buffer_size);
    return true;
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescDram(size_t page_size, size_t buffer_size, int mem_dev_id) {
    void *addr;

    if (!allocateXferMemory(page_size, buffer_size, &addr)) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of DRAM memory" << std::endl;
        return std::nullopt;
    }

    // TODO: Does device id need to be set for DRAM?
    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, mem_dev_id);
}


std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescDram(size_t buffer_size, int mem_dev_id) {
    void *addr;

    if (!allocateXferMemory(config.page_size, buffer_size, &addr)) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of DRAM memory" << std::endl;
        return std::nullopt;
    }

    // TODO: Does device id need to be set for DRAM?
    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, mem_dev_id);
}

#if HAVE_CUDA
static std::optional<xferBenchIOV>
getVramDescCuda(int devid, size_t buffer_size, uint8_t memset_value) {
    void *addr;
    CHECK_CUDA_ERROR(cudaMalloc(&addr, buffer_size), "Failed to allocate CUDA buffer");
    CHECK_CUDA_ERROR(cudaMemset(addr, memset_value, buffer_size), "Failed to set device memory");

    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, devid);
}

static std::optional<xferBenchIOV>
getVramDescCudaVmm(int devid, size_t buffer_size, uint8_t memset_value) {
#if HAVE_CUDA_FABRIC
    CUdeviceptr addr = 0;
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access = {};

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.location.id = devid;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    // Get the allocation granularity
    size_t granularity = 0;
    CHECK_CUDA_DRIVER_ERROR(
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "Failed to get allocation granularity");
    std::cout << "Granularity: " << granularity << std::endl;

    size_t padded_size = ROUND_UP(buffer_size, granularity);
    CUmemGenericAllocationHandle handle;
    CHECK_CUDA_DRIVER_ERROR(cuMemCreate(&handle, padded_size, &prop, 0),
                            "Failed to create allocation");

    // Reserve the memory address
    CHECK_CUDA_DRIVER_ERROR(cuMemAddressReserve(&addr, padded_size, granularity, 0, 0),
                            "Failed to reserve address");

    // Map the memory
    CHECK_CUDA_DRIVER_ERROR(cuMemMap(addr, padded_size, 0, handle, 0), "Failed to map memory");

    std::cout << "Address: " << std::hex << std::showbase << addr << " Buffer size: " << std::dec
              << buffer_size << " Padded size: " << std::dec << padded_size << std::endl;

    // Set the memory access rights
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = devid;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA_DRIVER_ERROR(cuMemSetAccess(addr, buffer_size, &access, 1), "Failed to set access");

    // Set memory content based on role
    CHECK_CUDA_DRIVER_ERROR(cuMemsetD8(addr, memset_value, buffer_size),
                            "Failed to set VMM device memory");

    return std::optional<xferBenchIOV>(
        std::in_place, (uintptr_t)addr, buffer_size, devid, padded_size, handle);

#else
    std::cerr << "CUDA_FABRIC is not supported" << std::endl;
    return std::nullopt;
#endif /* HAVE_CUDA_FABRIC */
}

static std::optional<xferBenchIOV>
getVramDescNeuron(int devid, size_t buffer_size, uint8_t memset_value) {
    void *addr;
    CHECK_NEURON_ERROR(neuronMalloc(&addr, buffer_size, devid), "Failed to allocate nrt tensor");
    CHECK_NEURON_ERROR(neuronMemset(addr, memset_value, buffer_size),
                       "Failed to set device memory");

    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, devid);
}

static std::optional<xferBenchIOV>
getVramDesc(const xferBenchConfig &config, int devid, size_t buffer_size, bool isInit) {
    uint8_t memset_value =
        isInit ? XFERBENCH_INITIATOR_BUFFER_ELEMENT : XFERBENCH_TARGET_BUFFER_ELEMENT;

    // Assume no CUDA cores exist if Neuron cores are found.
    // There are no AWS instance types with both NVIDIA GPUs and Neuron accelerators.
    if (neuronCoreCount() > 0) {
        return getVramDescNeuron(devid, buffer_size, memset_value);
    }

    CHECK_CUDA_ERROR(cudaSetDevice(devid), "Failed to set device");
    if (config.enable_vmm) {
        return getVramDescCudaVmm(devid, buffer_size, memset_value);
    } else {
        return getVramDescCuda(devid, buffer_size, memset_value);
    }
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescVram(size_t buffer_size, int mem_dev_id) {
    if (IS_PAIRWISE_AND_SG(config)) {
        int devid = rt->getRank();

        if (isTarget()) {
            devid -= config.num_initiator_dev;
        }

        if (devid != mem_dev_id) {
            return std::nullopt;
        }
    }

    return getVramDesc(config, mem_dev_id, buffer_size, isInitiator());
}
#endif /* HAVE_CUDA */

// Helper to open a single file with appropriate flags
static std::optional<xferFileState>
openFileWithFlags(const std::string &op_type, const std::string &file_name, int flags) {
    uint64_t file_size = 0;
    if (XFERBENCH_OP_READ == op_type) {
        struct stat st;
        if (::stat(file_name.c_str(), &st) == 0) {
            std::cout << "File " << file_name << " exists, size: " << st.st_size << std::endl;
            file_size = st.st_size;
        } else {
            std::cout << "File " << file_name << " does not exist, will be created." << std::endl;
        }
    }

    int fd = open(file_name.c_str(), flags, 0744);
    if (fd < 0) {
        std::cerr << "Failed to open file: " << file_name << " with error: " << strerror(errno)
                  << std::endl;
        return std::nullopt;
    }

    return xferFileState{fd, file_size, 0};
}

// Create file descriptors from explicit filenames or auto-generate
static std::vector<xferFileState>
createFileFds(const nixlbench::storageConfig &storage_config,
              const std::string &op_type,
              std::string name,
              int num_files,
              const std::vector<std::string> &filenames = {}) {
    std::vector<xferFileState> fds;
    int flags = O_RDWR | O_CREAT | O_LARGEFILE;

    if (storage_config.enable_direct) {
        flags |= O_DIRECT;
    }

    // Use provided filenames if available
    if (!filenames.empty()) {
        if (filenames.size() != static_cast<size_t>(num_files)) {
            std::cerr << "Error: Number of filenames (" << filenames.size()
                      << ") doesn't match num_files (" << num_files << ")" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (const auto &file_name : filenames) {
            std::cout << "Opening file: " << file_name << std::endl;
            auto fstate = openFileWithFlags(op_type, file_name, flags);
            if (!fstate) {
                // Cleanup already opened files
                for (auto &fd : fds) {
                    close(fd.fd);
                }
                return {};
            }
            fds.push_back(fstate.value());
        }
        return fds;
    }

    // Auto-generate filenames (backward compatibility)
    const std::string file_path = storage_config.filepath != "" ?
        storage_config.filepath :
        std::filesystem::current_path().string();
    // std::string file_backend = config.backend;
    // std::transform(file_backend.begin(), file_backend.end(), file_backend.begin(), ::tolower);
    const std::string file_name_prefix = "/nixlbench_test_file_";

    for (int i = 0; i < num_files; i++) {
        std::string file_name = file_path + file_name_prefix + name + "_" + std::to_string(i);
        std::cout << "Creating file: " << file_name << std::endl;

        auto fstate = openFileWithFlags(op_type, file_name, flags);
        if (!fstate) {
            // Cleanup already opened files
            for (int j = 0; j < i; j++) {
                close(fds[j].fd);
            }
            return {};
        }
        fds.push_back(fstate.value());
    }
    return fds;
}

// Create file descriptors from explicit filenames or auto-generate
static std::vector<xferFileState>
createFileFds(const xferBenchConfig &config,
              std::string name,
              int num_files,
              const std::vector<std::string> &filenames = {}) {
    std::vector<xferFileState> fds;
    int flags = O_RDWR | O_CREAT | O_LARGEFILE;

    if (!config.isStorageBackend()) {
        std::cerr << "Unknown storage backend: " << config.backend << std::endl;
        exit(EXIT_FAILURE);
    }

    if (config.storage_enable_direct) {
        flags |= O_DIRECT;
    }

    // Use provided filenames if available
    if (!filenames.empty()) {
        if (filenames.size() != static_cast<size_t>(num_files)) {
            std::cerr << "Error: Number of filenames (" << filenames.size()
                      << ") doesn't match num_files (" << num_files << ")" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (const auto &file_name : filenames) {
            std::cout << "Opening file: " << file_name << std::endl;
            auto fstate = openFileWithFlags(config.op_type, file_name, flags);
            if (!fstate) {
                // Cleanup already opened files
                for (auto &fd : fds) {
                    close(fd.fd);
                }
                return {};
            }
            fds.push_back(fstate.value());
        }
        return fds;
    }

    // Auto-generate filenames (backward compatibility)
    const std::string file_path =
        config.filepath != "" ? config.filepath : std::filesystem::current_path().string();
    std::string file_backend = config.backend;
    std::transform(file_backend.begin(), file_backend.end(), file_backend.begin(), ::tolower);
    const std::string file_name_prefix = "/nixlbench_" + file_backend + "_test_file_";

    for (int i = 0; i < num_files; i++) {
        std::string file_name = file_path + file_name_prefix + name + "_" + std::to_string(i);
        std::cout << "Creating file: " << file_name << std::endl;

        auto fstate = openFileWithFlags(config.op_type, file_name, flags);
        if (!fstate) {
            // Cleanup already opened files
            for (int j = 0; j < i; j++) {
                close(fds[j].fd);
            }
            return {};
        }
        fds.push_back(fstate.value());
    }
    return fds;
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescFile(std::string op_type,size_t page_size,size_t buffer_size, xferFileState &fstate, int mem_dev_id) {
    int fd = fstate.fd;
    uint64_t start_offset = fstate.offset;
    uint64_t end_offset = fstate.offset + buffer_size;
    auto ret = std::optional<xferBenchIOV>(std::in_place, fstate.offset, buffer_size, fd);

    fstate.offset = end_offset;

    // If in READ mode, only write if the region is not already present in the file
    if (XFERBENCH_OP_READ == op_type && end_offset <= fstate.file_size) {
        return ret;
    }

    // Fill up with data
    void *buf;
    if (!allocateXferMemory(page_size, buffer_size, &buf)) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of memory" << std::endl;
        return std::nullopt;
    }

    // File is always initialized with XFERBENCH_TARGET_BUFFER_ELEMENT
    memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);

    size_t offset = start_offset;
    char *write_ptr = static_cast<char *>(buf);
    while (buffer_size > 0) {
        ssize_t rc = pwrite(fd, write_ptr, buffer_size, offset);
        if (rc < 0) {
            std::cerr << "Failed to write to file: " << fd << " with error: " << strerror(errno)
                      << std::endl;
            return std::nullopt;
        }

        buffer_size -= rc;
        offset += rc;
        write_ptr += rc;
    }

    free(buf);

    if (end_offset > fstate.file_size) fstate.file_size = end_offset;

    return ret;
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescObj(size_t buffer_size, int mem_dev_id, std::string name) {
    return std::optional<xferBenchIOV>(std::in_place, 0, buffer_size, mem_dev_id, name);
}

void
xferBenchNixlWorker::cleanupBasicDescDram(xferBenchIOV &iov) {
    free((void *)iov.addr);
}

#if HAVE_CUDA
void
xferBenchNixlWorker::cleanupBasicDescVram(xferBenchIOV &iov) {
    // Assume no CUDA cores exist if Neuron cores are found.
    // There are no AWS instance types with both NVIDIA GPUs and Neuron accelerators.
    if (neuronCoreCount() > 0) {
        CHECK_NEURON_ERROR(neuronFree((void *)iov.addr), "Failed to free nrt tensor");
        return;
    }

    CHECK_CUDA_ERROR(cudaSetDevice(iov.devId), "Failed to set device");
    if (config.enable_vmm) {
        CHECK_CUDA_DRIVER_ERROR(cuMemUnmap(iov.addr, iov.len), "Failed to unmap memory");
        CHECK_CUDA_DRIVER_ERROR(cuMemRelease(iov.handle), "Failed to release memory");
        CHECK_CUDA_DRIVER_ERROR(cuMemAddressFree(iov.addr, iov.padded_size),
                                "Failed to free reserved address");
    } else {
        /*
         * CUDA streams allow for concurrent execution of kernels and memory operations. However,
         * memory management functions like cudaFree are implicitly synchronized with all streams to
         * guarantee safety. This means cudaFree will wait for all kernels (in any stream) that
         * might use the memory to finish before actually freeing it.
         * If the application hangs on cudaFree due to kernels running in other streams, switching
         * to cudaFreeAsync can allow the host to proceed without waiting for the entire device
         * synchronization.
         */
        CHECK_CUDA_ERROR(cudaFreeAsync((void *)iov.addr, 0), "Failed to deallocate CUDA buffer");
        CHECK_CUDA_ERROR(cudaStreamSynchronize(0), "Failed to synchronize stream 0");
    }
}
#endif /* HAVE_CUDA */

void
xferBenchNixlWorker::cleanupBasicDescFile(xferBenchIOV &iov) {
    close(iov.devId);
}

void
xferBenchNixlWorker::cleanupBasicDescObj(xferBenchIOV &iov) {
    if (!xferBenchUtils::rmObj(config, iov.metaInfo)) {
        std::cerr << "Failed to remove object: " << iov.metaInfo << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescBlk(size_t buffer_size, int mem_dev_id, size_t dev_offset) {
    // The dev_offset represents the LBA (Logical Block Address) offset in the block device

    // Create IOV with LBA offset as address, buffer size, and device ID
    // The device ID corresponds to the block device UUID (e.g., 11 for local file, 14 for
    // /dev/zero)
    return std::optional<xferBenchIOV>(std::in_place, dev_offset, buffer_size, mem_dev_id);
}

void
xferBenchNixlWorker::cleanupBasicDescBlk(xferBenchIOV &iov) {
    // No cleanup needed for block device descriptors
    // The block device backend handles the device lifecycle
}

bool
xferBenchNixlWorker::ensureFileHasConsistencyData(const GusliDeviceConfig &device, size_t size) {
    int flags = O_RDWR | O_CREAT | O_LARGEFILE;
    if (config.storage_enable_direct) flags |= O_DIRECT;

    int fd = open(device.device_path.c_str(), flags, 0744);
    if (fd < 0) {
        std::cerr << "Failed to open GUSLI file: " << device.device_path << ": " << strerror(errno)
                  << std::endl;
        return false;
    }

    // Sample one page at the offset GUSLI will read from
    void *check_buf;
    bool needs_write = true;
    if (allocateXferMemory(config.page_size, config.page_size, &check_buf)) {
        ssize_t rd = pread(fd, check_buf, config.page_size, device.dev_offset);
        if (rd == (ssize_t)config.page_size) {
            needs_write = false;
            uint8_t *bytes = static_cast<uint8_t *>(check_buf);
            for (ssize_t i = 0; i < rd; i++) {
                if (bytes[i] != XFERBENCH_TARGET_BUFFER_ELEMENT) {
                    needs_write = true;
                    break;
                }
            }
        }
        free(check_buf);
    }

    if (needs_write) {
        std::cout << "Warning: GUSLI file '" << device.device_path << "' at offset "
                  << device.dev_offset << " does not contain expected pattern (0x" << std::hex
                  << (int)XFERBENCH_TARGET_BUFFER_ELEMENT << std::dec << "). Overwriting."
                  << std::endl;

        void *buf;
        if (!allocateXferMemory(config.page_size, size, &buf)) {
            close(fd);
            return false;
        }
        memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, size);

        size_t remaining = size;
        size_t offset = device.dev_offset;
        char *write_ptr = static_cast<char *>(buf);
        while (remaining > 0) {
            ssize_t rc = pwrite(fd, write_ptr, remaining, offset);
            if (rc < 0) {
                std::cerr << "Failed to write to " << device.device_path << " at offset " << offset
                          << ": " << strerror(errno) << std::endl;
                free(buf);
                close(fd);
                return false;
            }
            remaining -= rc;
            offset += rc;
            write_ptr += rc;
        }
        free(buf);
    } else {
        std::cout << "GUSLI file '" << device.device_path << "' at offset " << device.dev_offset
                  << " already contains expected pattern (0x" << std::hex
                  << (int)XFERBENCH_TARGET_BUFFER_ELEMENT << std::dec
                  << "). Skipping initialization." << std::endl;
    }

    close(fd);
    return true;
}

std::vector<std::vector<xferBenchIOV>>
xferBenchNixlWorker::allocateMemory(int num_threads,
                                    nixlAgent *agent,
                                    const nixlbench::storageConfig &storage_config,
                                    const nixlbench::transferConfig &transfer_config,
                                    const nixlbench::backendConfig &backend) {
    std::vector<std::vector<xferBenchIOV>> iov_lists;
    size_t buffer_size;
    nixl_opt_args_t opt_args;
    std::vector<std::vector<xferBenchIOV>> remote_iovs;

    size_t page_size = sysconf(_SC_PAGESIZE);
    buffer_size = transfer_config.total_buffer_size / (num_threads);

    if (storage_config.enable_direct) {
        // need to handle hugepages
        if (page_size == 0) {
            std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
            exit(EXIT_FAILURE);
        }

        buffer_size = ((buffer_size + page_size - 1) / page_size) * page_size;
    }

    // if (backend.capabilities.canUseAsStorage && !backend.capabilities.canReadWriteFiles) {
    //     struct timeval tv;
    //     gettimeofday(&tv, nullptr);
    //     uint64_t timestamp = tv.tv_sec * 1000000ULL + tv.tv_usec;

    //     for (int list_idx = 0; list_idx < num_threads; list_idx++) {
    //         std::vector<xferBenchIOV> iov_list;
    //         std::optional<xferBenchIOV> basic_desc;
    //         std::string unique_name = "nixlbench_obj" + std::to_string(list_idx) + "_" +
    //             std::to_string(i) + "_" + std::to_string(timestamp);

    //         if (transfer_config.op_type == XFERBENCH_OP_READ) {
    //             if (!xferBenchUtils::putObj(config, buffer_size, unique_name)) {
    //                 std::cerr << "Failed to put object: " << unique_name << std::endl;
    //                 continue;
    //             }
    //         }

    //         basic_desc = initBasicDescObj(buffer_size, i, unique_name);
    //         if (basic_desc) {
    //             std::cout << "Creating obj: " << unique_name << std::endl;
    //             iov_list.push_back(basic_desc.value());
    //         }

    //         nixl_reg_dlist_t desc_list(OBJ_SEG);
    //         iovListToNixlRegDlist(iov_list, desc_list);
    //         CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
    //         remote_iovs.push_back(iov_list);
    //     }
    // } else if (XFERBENCH_BACKEND_GUSLI == config.backend) {
    //     // GUSLI backend uses block device descriptors
    //     if (gusli_devices.empty()) {
    //         std::cerr << "No GUSLI devices configured" << std::endl;
    //         exit(EXIT_FAILURE);
    //     }

    //     if (config.op_type == XFERBENCH_OP_READ) {
    //         for (auto &device : gusli_devices) {
    //             if (device.device_type == 'F' &&
    //                 !ensureFileHasConsistencyData(device, buffer_size)) {
    //                 exit(EXIT_FAILURE);
    //             }
    //         }
    //     }

    //     for (int list_idx = 0; list_idx < num_threads; list_idx++) {
    //         std::vector<xferBenchIOV> iov_list;
    //         for (i = 0; i < num_devices; i++) {
    //             std::optional<xferBenchIOV> basic_desc;
    //             // Use device IDs from parsed configuration (num_devices == gusli_devices.size())
    //             basic_desc = initBasicDescBlk(
    //                 buffer_size, gusli_devices[i].device_id, gusli_devices[i].dev_offset);
    //             if (basic_desc) {
    //                 iov_list.push_back(basic_desc.value());
    //             }
    //         }
    //         nixl_reg_dlist_t desc_list(BLK_SEG);
    //         iovListToNixlRegDlist(iov_list, desc_list);
    //         CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
    //         remote_iovs.push_back(iov_list);
    //     }
    if (backend.capabilities.canReadWriteFiles) {
        // int num_buffers = num_threads;
        int num_files = storage_config.num_files;
        // int remainder_buffers = num_buffers % num_files;

        std::vector<std::string> filenames;
        if (!storage_config.filenames.empty()) {
            std::string filename;
            std::stringstream ss(storage_config.filenames);
            while (std::getline(ss, filename, ',')) {
                filenames.push_back(filename);
            }
        }
        // assumes storage mode so just use the backend name
        std::vector<xferFileState> remote_file_descriptors = createFileFds(
            storage_config, transfer_config.op_type, backend.name, num_files, filenames);
        if (remote_file_descriptors.empty()) {
            std::cerr << "Failed to create files for " << backend.name << std::endl;
            exit(EXIT_FAILURE);
        }

        int file_idx = 0;
        for (int list_idx = 0; list_idx < num_threads; list_idx++) {
            std::vector<xferBenchIOV> iov_list;
            std::optional<xferBenchIOV> basic_desc;
            basic_desc = initBasicDescFile(transfer_config.op_type, page_size, buffer_size, remote_file_descriptors[file_idx], 0);
            if (basic_desc) {
                iov_list.push_back(basic_desc.value());
            }
            file_idx += 1;
            if (file_idx >= num_files) file_idx = 0;
            nixl_reg_dlist_t desc_list(FILE_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
            remote_iovs.push_back(iov_list);
        }
    }

    for (int list_idx = 0; list_idx < num_threads; list_idx++) {
        std::vector<xferBenchIOV> iov_list;
        std::optional<xferBenchIOV> basic_desc;

        nixl_mem_t seg_type = transfer_config.initiator_seg_type == XFERBENCH_SEG_TYPE_DRAM ? DRAM_SEG : VRAM_SEG;
        switch (seg_type) {
        case DRAM_SEG: {
            // TODD, make the gusli exceptions here in a smart way
            int mem_dev_id = 0;
            basic_desc = initBasicDescDram(page_size, buffer_size, mem_dev_id);
            break;
        }
#if HAVE_CUDA
        // case VRAM_SEG:
        //     basic_desc = initBasicDescVram(buffer_size, i);
        //     break;
#endif
        default:
            std::cerr << "Unsupported mem type: " << seg_type << std::endl;
            exit(EXIT_FAILURE);
        }

        if (basic_desc) {
            if (!remote_iovs.empty()) {
                basic_desc.value().metaInfo = remote_iovs[list_idx][0].metaInfo;
            }
            iov_list.push_back(basic_desc.value());
        }

        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
        iov_lists.push_back(iov_list);
    }

    return iov_lists;
}

std::vector<std::vector<xferBenchIOV>>
xferBenchNixlWorker::allocateMemory(int num_threads) {
    std::vector<std::vector<xferBenchIOV>> iov_lists;
    size_t i, buffer_size, num_devices = 0;
    nixl_opt_args_t opt_args;

    if (isInitiator()) {
        num_devices = config.num_initiator_dev;
    } else if (isTarget()) {
        num_devices = config.num_target_dev;
    }
    buffer_size = config.total_buffer_size / (num_devices * num_threads);

    if (config.storage_enable_direct) {
        if (config.page_size == 0) {
            std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
            exit(EXIT_FAILURE);
        }
        buffer_size = ((buffer_size + config.page_size - 1) / config.page_size) * config.page_size;
    }

    opt_args.backends.push_back(backend_engine);

    if (config.isObjStorageBackend()) {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        uint64_t timestamp = tv.tv_sec * 1000000ULL + tv.tv_usec;

        for (int list_idx = 0; list_idx < num_threads; list_idx++) {
            std::vector<xferBenchIOV> iov_list;
            for (i = 0; i < num_devices; i++) {
                std::optional<xferBenchIOV> basic_desc;
                std::string unique_name = "nixlbench_obj" + std::to_string(list_idx) + "_" +
                    std::to_string(i) + "_" + std::to_string(timestamp);

                if (config.op_type == XFERBENCH_OP_READ) {
                    if (!xferBenchUtils::putObj(config, buffer_size, unique_name)) {
                        std::cerr << "Failed to put object: " << unique_name << std::endl;
                        continue;
                    }
                }

                basic_desc = initBasicDescObj(buffer_size, i, unique_name);
                if (basic_desc) {
                    std::cout << "Creating obj: " << unique_name << std::endl;
                    iov_list.push_back(basic_desc.value());
                }
            }
            nixl_reg_dlist_t desc_list(OBJ_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
            remote_iovs.push_back(iov_list);
        }
    } else if (XFERBENCH_BACKEND_GUSLI == config.backend) {
        // GUSLI backend uses block device descriptors
        if (gusli_devices.empty()) {
            std::cerr << "No GUSLI devices configured" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (config.op_type == XFERBENCH_OP_READ) {
            for (auto &device : gusli_devices) {
                if (device.device_type == 'F' &&
                    !ensureFileHasConsistencyData(device, buffer_size)) {
                    exit(EXIT_FAILURE);
                }
            }
        }

        for (int list_idx = 0; list_idx < num_threads; list_idx++) {
            std::vector<xferBenchIOV> iov_list;
            for (i = 0; i < num_devices; i++) {
                std::optional<xferBenchIOV> basic_desc;
                // Use device IDs from parsed configuration (num_devices == gusli_devices.size())
                basic_desc = initBasicDescBlk(
                    buffer_size, gusli_devices[i].device_id, gusli_devices[i].dev_offset);
                if (basic_desc) {
                    iov_list.push_back(basic_desc.value());
                }
            }
            nixl_reg_dlist_t desc_list(BLK_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
            remote_iovs.push_back(iov_list);
        }
    } else if (config.isStorageBackend()) {
        int num_buffers = num_threads * num_devices;
        int num_files = config.num_files;
        int remainder_buffers = num_buffers % num_files;

        if (num_files > num_buffers) {
            std::cerr << "Error: number of buffers (" << num_buffers
                      << ") needs to be bigger or equal to the number of files (" << num_files
                      << "). Try adjusting num_files." << std::endl;
            exit(EXIT_FAILURE);
        }

        if (remainder_buffers != 0) {
            std::cerr << "Error: number of buffers (" << num_buffers
                      << ") needs to be divisible by the number of files (" << num_files
                      << "). Try adjusting num_files." << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<std::string> filenames;
        if (!config.filenames.empty()) {
            std::string filename;
            std::stringstream ss(config.filenames);
            while (std::getline(ss, filename, ',')) {
                filenames.push_back(filename);
            }
        }
        remote_fds = createFileFds(config, getName(), num_files, filenames);
        if (remote_fds.empty()) {
            std::cerr << "Failed to create " << config.backend << " file" << std::endl;
            exit(EXIT_FAILURE);
        }

        int file_idx = 0;
        for (int list_idx = 0; list_idx < num_threads; list_idx++) {
            std::vector<xferBenchIOV> iov_list;
            for (i = 0; i < num_devices; i++) {
                std::optional<xferBenchIOV> basic_desc;
                basic_desc = initBasicDescFile(config.op_type, config.page_size,buffer_size, remote_fds[file_idx], i);
                if (basic_desc) {
                    iov_list.push_back(basic_desc.value());
                }
                file_idx += 1;
                if (file_idx >= num_files) file_idx = 0;
            }
            nixl_reg_dlist_t desc_list(FILE_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
            remote_iovs.push_back(iov_list);
        }
    }

    for (int list_idx = 0; list_idx < num_threads; list_idx++) {
        std::vector<xferBenchIOV> iov_list;
        for (i = 0; i < num_devices; i++) {
            std::optional<xferBenchIOV> basic_desc;

            switch (seg_type) {
            case DRAM_SEG: {
                // For GUSLI backend, use device ID from parsed configuration
                int mem_dev_id =
                    (XFERBENCH_BACKEND_GUSLI == config.backend && !gusli_devices.empty()) ?
                    gusli_devices[i].device_id :
                    i;
                basic_desc = initBasicDescDram(buffer_size, mem_dev_id);
                break;
            }
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
                if (!remote_iovs.empty()) {
                    basic_desc.value().metaInfo = remote_iovs[list_idx][i].metaInfo;
                }
                iov_list.push_back(basic_desc.value());
            }
        }

        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
        iov_lists.push_back(iov_list);

        /*
         * Workaround for a GUSLI registration bug which resets memory to 0, this initialization
         * is only needed when validating data. It was moved from the initBasicDescDram function to
         * here to avoid memsetting the memory again.
         */
        if (seg_type == DRAM_SEG && config.check_consistency) {
            for (auto &iov : iov_list) {
                if (isInitiator()) {
                    memset((void *)iov.addr, XFERBENCH_INITIATOR_BUFFER_ELEMENT, buffer_size);
                } else if (isTarget()) {
                    memset((void *)iov.addr, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);
                }
            }
        }
    }

    return iov_lists;
}

void
xferBenchNixlWorker::deallocateMemory(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    nixl_opt_args_t opt_args;

    opt_args.backends.push_back(backend_engine);
    for (auto &iov_list : iov_lists) {
        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");

        for (auto &iov : iov_list) {
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
    }

    if (config.isObjStorageBackend()) {
        for (auto &iov_list : remote_iovs) {
            for (auto &iov : iov_list) {
                cleanupBasicDescObj(iov);
            }
            nixl_reg_dlist_t desc_list(OBJ_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");
        }
    } else if (config.backend == XFERBENCH_BACKEND_GUSLI) {
        for (auto &iov_list : remote_iovs) {
            for (auto &iov : iov_list) {
                cleanupBasicDescBlk(iov);
            }
            nixl_reg_dlist_t desc_list(BLK_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");
        }
    } else if (config.isStorageBackend()) {
        for (auto &iov_list : remote_iovs) {
            for (auto &iov : iov_list) {
                cleanupBasicDescFile(iov);
            }
            nixl_reg_dlist_t desc_list(FILE_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");
        }
    }
}

int
xferBenchNixlWorker::exchangeMetadata() {
    int meta_sz, ret = 0;

    // Skip metadata exchange for storage backends or when ETCD is not available
    if (config.isStorageBackend()) {
        return 0;
    }

    if (isTarget()) {
        std::string local_metadata;
        const char *buffer;
        int destrank;

        agent->getLocalMD(local_metadata);

        buffer = local_metadata.data();
        meta_sz = local_metadata.size();

        if (IS_PAIRWISE_AND_SG(config)) {
            destrank = rt->getRank() - config.num_target_dev;
            // XXX: Fix up the rank, depends on processes distributed on hosts
            // assumes placement is adjacent ranks to same node
        } else {
            destrank = 0;
        }
        rt->sendInt(&meta_sz, destrank);
        rt->sendChar((char *)buffer, meta_sz, destrank);
    } else if (isInitiator()) {
        std::string remote_agent;
        int srcrank;

        if (IS_PAIRWISE_AND_SG(config)) {
            srcrank = rt->getRank() + config.num_initiator_dev;
            // XXX: Fix up the rank, depends on processes distributed on hosts
            // assumes placement is adjacent ranks to same node
        } else {
            srcrank = 1;
        }

        ret = rt->recvInt(&meta_sz, srcrank);
        if (ret < 0) {
            std::cerr << "NIXL: failed to receive metadata size" << std::endl;
            return ret;
        }

        std::string remote_metadata(meta_sz, '\0');
        ret = rt->recvChar(remote_metadata.data(), meta_sz, srcrank);
        if (ret < 0) {
            std::cerr << "NIXL: failed to receive metadata" << std::endl;
            return ret;
        }

        nixl_status_t status = agent->loadRemoteMD(remote_metadata, remote_agent);
        if (status != NIXL_SUCCESS) {
            std::cerr << "NIXL: loadRemoteMD failed: " << nixlEnumStrings::statusStr(status)
                      << std::endl;
            return -1;
        }
    }

    return ret;
}

std::vector<std::vector<xferBenchIOV>>
xferBenchNixlWorker::exchangeIOV(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                                 size_t block_size) {
    std::vector<std::vector<xferBenchIOV>> res;
    int desc_str_sz;

    if (config.isStorageBackend()) {
        size_t fd_idx = 0;
        uint64_t file_offset = 0;
        for (auto &iov_list : local_iovs) {
            std::vector<xferBenchIOV> remote_iov_list;
            int devidx = 0;
            for (auto &iov : iov_list) {
                if (config.isObjStorageBackend()) {
                    std::optional<xferBenchIOV> basic_desc;
                    basic_desc = initBasicDescObj(iov.len, iov.devId, iov.metaInfo);
                    if (basic_desc) {
                        remote_iov_list.push_back(basic_desc.value());
                    }
                } else if (XFERBENCH_BACKEND_GUSLI == config.backend) {
                    xferBenchIOV iov_remote(iov);
                    iov_remote.addr = gusli_devices[devidx++].dev_offset + file_offset;
                    iov_remote.len = block_size;
                    iov_remote.devId = iov.devId;
                    remote_iov_list.push_back(iov_remote);
                } else {
                    xferBenchIOV iov_remote(iov);
                    iov_remote.addr = file_offset;
                    iov_remote.len = block_size;
                    iov_remote.devId = remote_fds[fd_idx].fd;
                    remote_iov_list.push_back(iov_remote);
                    fd_idx++;
                    if (fd_idx >= remote_fds.size()) {
                        file_offset += block_size;
                        fd_idx = 0;
                    }
                }
            }
            res.push_back(remote_iov_list);
            file_offset += block_size;
        }
    } else {
        for (const auto &local_iov : local_iovs) {
            nixlSerDes ser_des;
            nixl_xfer_dlist_t local_desc(seg_type);

            iovListToNixlXferDlist(local_iov, local_desc);

            if (isTarget()) {
                int destrank;
                if (IS_PAIRWISE_AND_SG(config)) {
                    destrank = rt->getRank() - config.num_target_dev;
                    // XXX: Fix up the rank, depends on processes distributed on hosts
                    // assumes placement is adjacent ranks to same node
                } else {
                    destrank = 0;
                }

                local_desc.serialize(&ser_des);
                std::string desc_str = ser_des.exportStr();
                desc_str_sz = desc_str.size();
                rt->sendInt(&desc_str_sz, destrank);
                rt->sendChar(desc_str.data(), desc_str.size(), destrank);
            } else if (isInitiator()) {
                int srcrank;
                if (IS_PAIRWISE_AND_SG(config)) {
                    srcrank = rt->getRank() + config.num_initiator_dev;
                    // XXX: Fix up the rank, depends on processes distributed on hosts
                    // assumes placement is adjacent ranks to same node
                } else {
                    srcrank = 1;
                }

                if (rt->recvInt(&desc_str_sz, srcrank) != 0) {
                    std::cerr << "NIXL: failed to receive metadata size" << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                std::string desc_str;
                desc_str.resize(desc_str_sz, '\0');
                if (rt->recvChar(desc_str.data(), desc_str.size(), srcrank) != 0) {
                    std::cerr << "NIXL: failed to receive metadata" << std::endl;
                    std::exit(EXIT_FAILURE);
                }

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

// Helper to execute a single transfer iteration
static inline nixl_status_t
execSingleTransfer(nixlAgent *agent,
                   nixlXferReqH *req,
                   xferBenchTimer &timer,
                   xferBenchStats &thread_stats) {
    nixl_status_t rc = agent->postXferReq(req);
    thread_stats.post_duration.add(timer.lap());
    while (NIXL_IN_PROG == rc) {
        rc = agent->getXferStatus(req);
    }
    return rc;
}

// Helper to prepare transfer descriptors based on backend type
static void
prepareTransferDescriptors(const xferBenchConfig &config,
                           nixl_xfer_dlist_t &local_desc,
                           nixl_xfer_dlist_t &remote_desc,
                           const std::vector<xferBenchIOV> &local_iov,
                           const std::vector<xferBenchIOV> &remote_iov) {
    // Set remote descriptor type based on backend
    if (config.isObjStorageBackend()) {
        remote_desc = nixl_xfer_dlist_t(OBJ_SEG);
    } else if (XFERBENCH_BACKEND_GUSLI == config.backend) {
        remote_desc = nixl_xfer_dlist_t(BLK_SEG);
    } else if (config.isStorageBackend()) {
        remote_desc = nixl_xfer_dlist_t(FILE_SEG);
    }

    iovListToNixlXferDlist(local_iov, local_desc);
    iovListToNixlXferDlist(remote_iov, remote_desc);
}

// Execute transfers with configurable request lifecycle behavior
// recreate_per_iteration: true for GUSLI (bug workaround), false for standard backends
static int
execTransferIterations(const xferBenchConfig &config,
                       nixlAgent *agent,
                       const nixl_xfer_op_t op,
                       nixl_xfer_dlist_t &local_desc,
                       nixl_xfer_dlist_t &remote_desc,
                       const std::string &target,
                       nixl_opt_args_t &params,
                       const int num_iter,
                       xferBenchTimer &timer,
                       xferBenchStats &thread_stats,
                       const bool recreate_per_iteration) {
    nixlXferReqH *req = nullptr;
    nixlTime::us_t total_prepare_duration = 0;

    // Create request once if not recreating per iteration
    if (!recreate_per_iteration) {
        nixl_status_t create_rc =
            agent->createXferReq(op, local_desc, remote_desc, target, req, &params);
        if (NIXL_SUCCESS != create_rc) {
            std::cerr << "createXferReq failed: " << nixlEnumStrings::statusStr(create_rc)
                      << std::endl;
            return -1;
        }
        thread_stats.prepare_duration.add(timer.lap());
    }

    // Execute transfer iterations
    // Branch prediction hint: most backends don't recreate per iteration
    if (__builtin_expect(recreate_per_iteration, 0)) {
        // GUSLI path: Create/execute/release per iteration
        for (int i = 0; i < num_iter; ++i) {
            nixl_status_t create_rc =
                agent->createXferReq(op, local_desc, remote_desc, target, req, &params);
            if (__builtin_expect(create_rc != NIXL_SUCCESS, 0)) {
                std::cerr << "createXferReq failed: " << nixlEnumStrings::statusStr(create_rc)
                          << std::endl;
                return -1;
            }
            total_prepare_duration += timer.lap();

            nixl_status_t rc = execSingleTransfer(agent, req, timer, thread_stats);

            if (__builtin_expect(rc != NIXL_SUCCESS, 0)) {
                std::cout << "NIXL Xfer failed with status: " << nixlEnumStrings::statusStr(rc)
                          << std::endl;
                agent->releaseXferReq(req);
                return -1;
            }
            thread_stats.transfer_duration.add(timer.lap());

            if (__builtin_expect(agent->releaseXferReq(req) != NIXL_SUCCESS, 0)) {
                std::cout << "NIXL releaseXferReq failed" << std::endl;
                return -1;
            }
        }
        // Average prepare duration across iterations
        thread_stats.prepare_duration.add(total_prepare_duration / num_iter);
    } else {
        // Standard path: Single request for all iterations
        for (int i = 0; i < num_iter; ++i) {
            nixl_status_t rc = execSingleTransfer(agent, req, timer, thread_stats);

            if (__builtin_expect(rc != NIXL_SUCCESS, 0)) {
                std::cout << "NIXL Xfer failed with status: " << nixlEnumStrings::statusStr(rc)
                          << std::endl;
                agent->releaseXferReq(req);
                return -1;
            }
            thread_stats.transfer_duration.add(timer.lap());
        }

        // Release request once after all iterations
        if (__builtin_expect(agent->releaseXferReq(req) != NIXL_SUCCESS, 0)) {
            std::cout << "NIXL releaseXferReq failed" << std::endl;
            return -1;
        }
    }

    return 0;
}

static int
execTransfer(const xferBenchConfig &config,
             nixlAgent *agent,
             const std::vector<std::vector<xferBenchIOV>> &local_iovs,
             const std::vector<std::vector<xferBenchIOV>> &remote_iovs,
             const nixl_xfer_op_t op,
             const int num_iter,
             const int num_threads,
             xferBenchStats &stats) {
    int ret = 0;
    stats.clear();

    xferBenchTimer total_timer;
#pragma omp parallel num_threads(num_threads)
    {
        xferBenchStats thread_stats;
        thread_stats.reserve(num_iter);
        xferBenchTimer timer;
        const int tid = omp_get_thread_num();
        const auto &local_iov = local_iovs[tid];
        const auto &remote_iov = remote_iovs[tid];

        // Prepare transfer descriptors
        nixl_xfer_dlist_t local_desc(getLegacySegType(config, true));
        nixl_xfer_dlist_t remote_desc(getLegacySegType(config, false));
        prepareTransferDescriptors(config, local_desc, remote_desc, local_iov, remote_iov);

        // Setup transfer parameters
        nixl_opt_args_t params;
        std::string target = config.isStorageBackend() ? "initiator" : "target";
        if (!config.isStorageBackend()) {
            params.notif = "0xBEEF";
        }

        // Execute transfers
        const int result = execTransferIterations(config,
                                                  agent,
                                                  op,
                                                  local_desc,
                                                  remote_desc,
                                                  target,
                                                  params,
                                                  num_iter,
                                                  timer,
                                                  thread_stats,
                                                  config.recreate_xfer);

        if (__builtin_expect(result != 0, 0)) {
            ret = result;
        }

#pragma omp critical
        { stats.add(thread_stats); }
    }

    const nixlTime::us_t total_duration = total_timer.lap();
    stats.total_duration.add(total_duration);
    return ret;
}

std::variant<xferBenchStats, int>
xferBenchNixlWorker::transfer(size_t block_size,
                              const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                              const std::vector<std::vector<xferBenchIOV>> &remote_iovs) {
    int num_iter = config.num_iter / config.num_threads;
    int skip = config.warmup_iter / config.num_threads;
    xferBenchStats stats;
    int ret = 0;
    nixl_xfer_op_t xfer_op = XFERBENCH_OP_READ == config.op_type ? NIXL_READ : NIXL_WRITE;
    // int completion_flag = 1;

    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= config.large_blk_iter_ftr;
        num_iter /= config.large_blk_iter_ftr;
    }

    if (skip > 0) {
        ret = execTransfer(
            config, agent, local_iovs, remote_iovs, xfer_op, skip, config.num_threads, stats);
        if (ret < 0) {
            return std::variant<xferBenchStats, int>(ret);
        }
    }

    // Synchronize to ensure all processes have completed the warmup (iter and polling)
    synchronize();

    stats.clear();

    ret = execTransfer(
        config, agent, local_iovs, remote_iovs, xfer_op, num_iter, config.num_threads, stats);
    if (ret < 0) {
        return std::variant<xferBenchStats, int>(ret);
    }

    synchronize();
    return std::variant<xferBenchStats, int>(stats);
}

void
xferBenchNixlWorker::poll(size_t block_size) {
    nixl_notifs_t notifs;
    nixl_status_t status;
    int skip = 0, num_iter = 0, total_iter = 0;

    skip = config.warmup_iter;
    num_iter = config.num_iter;
    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= config.large_blk_iter_ftr;
        num_iter /= config.large_blk_iter_ftr;
    }
    total_iter = skip + num_iter;

    /* Ensure warmup is done*/
    do {
        status = agent->getNotifs(notifs);
    } while (status == NIXL_SUCCESS && skip != int(notifs["initiator"].size()));
    synchronize();

    /* Polling for actual iterations*/
    do {
        status = agent->getNotifs(notifs);
    } while (status == NIXL_SUCCESS && total_iter != int(notifs["initiator"].size()));
    synchronize();
}

int
xferBenchNixlWorker::synchronizeStart() {
    // For storage backends without ETCD, no synchronization needed
    if (config.isStorageBackend() && config.etcd_endpoints.empty()) {
        std::cout << "Single instance storage backend - no synchronization needed" << std::endl;
        return 0;
    }

    if (IS_PAIRWISE_AND_SG(config)) {
        std::cout << "Waiting for all processes to start... (expecting " << rt->getSize()
                  << " total: " << config.num_initiator_dev << " initiators and "
                  << config.num_target_dev << " targets)" << std::endl;
    } else {
        std::cout << "Waiting for all processes to start... (expecting " << rt->getSize()
                  << " total" << std::endl;
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
