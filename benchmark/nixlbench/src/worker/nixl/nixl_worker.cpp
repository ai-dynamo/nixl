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
#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cstring>
#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif HAVE_ROCM
#include <hip/hip_runtime.h>
#endif
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include "utils/neuron.h"
#include "utils/utils.h"
#include <unistd.h>
#include <utility>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <utils/serdes/serdes.h>
#include <omp.h>

// MAP_HUGE_2MB may not be defined on all systems, define it if needed
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26) // 2MB hugepage size encoding
#endif

#define CHECK_NIXL_ERROR(result, message)                                                     \
    do {                                                                                      \
        const nixl_status_t _r = (result);                                                    \
        if (0 != _r) {                                                                        \
            std::cerr << "NIXL: " << message << " (" << nixlEnumStrings::statusStr(_r) << ")" \
                      << std::endl;                                                           \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

static nixl_mem_t
resolveVramSegment() {
#if HAVE_CUDA
    return VRAM_SEG;
#elif HAVE_ROCM
    return VRAM_SEG;
#else
    if (neuronCoreCount() > 0) return VRAM_SEG;
    std::cerr << "VRAM not supported without CUDA, ROCm or Neuron" << std::endl;
    std::exit(EXIT_FAILURE);
#endif
}

#define GET_SEG_TYPE(is_initiator)                                                          \
    ({                                                                                      \
        std::string _seg_type_str = ((is_initiator) ? xferBenchConfig::initiator_seg_type : \
                                                      xferBenchConfig::target_seg_type);    \
        nixl_mem_t _seg_type;                                                               \
        if (0 == _seg_type_str.compare("DRAM")) {                                           \
            _seg_type = DRAM_SEG;                                                           \
        } else if (0 == _seg_type_str.compare("VRAM")) {                                    \
            _seg_type = resolveVramSegment();                                               \
        } else {                                                                            \
            std::cerr << "Invalid segment type: " << _seg_type_str << std::endl;            \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
        _seg_type;                                                                          \
    })

// Reuse parser from utils

// Generate GUSLI config file from device configurations
static std::string
generateGusliConfigFile(const std::vector<GusliDeviceConfig> &devices) {
    std::stringstream config;
    config << "# Config file\nversion=1\n";

    for (const auto &dev : devices) {
        // Format: "id type access_mode direct_io path security_flags"
        // Example: "11 F W D ./store0.bin sec=0x3"
        config << dev.device_id << " " << dev.device_type << " "
               << "W D " // Write mode, Direct I/O
               << dev.device_path << " " << dev.security_flags << "\n";
    }

    std::cout << "GUSLI Device Config: " << config.str() << std::endl;

    return config.str();
}

xferBenchNixlWorker::xferBenchNixlWorker(const std::vector<std::string> &devices)
    : xferBenchWorker() {
    seg_type = GET_SEG_TYPE(isInitiator());

    int rank;
    std::string backend_name;
    nixl_b_params_t backend_params;
    bool enable_pt = xferBenchConfig::enable_pt;
    nixl_thread_sync_t sync_mode = xferBenchConfig::num_threads > 1 ?
        nixl_thread_sync_t::NIXL_THREAD_SYNC_RW :
        nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT;
    char hostname[256];
    nixl_mem_list_t mems;
    std::vector<nixl_backend_t> plugins;

    rank = rt->getRank();

    nixlAgentConfig dev_meta;
    dev_meta.useProgThread = enable_pt;
    dev_meta.syncMode = sync_mode;

    agent = new nixlAgent(name, dev_meta);

    agent->getAvailPlugins(plugins);

    if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_LIBFABRIC) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GPUNETIO) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_MOONCAKE) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCCL) ||
        xferBenchConfig::isStorageBackend()) {
        backend_name = xferBenchConfig::backend;
    } else {
        std::cerr << "Unsupported NIXLBench backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    agent->getPluginParams(backend_name, mems, backend_params);

    if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX)) {
        backend_params["num_threads"] = std::to_string(xferBenchConfig::progress_threads);

        // No need to set device_list if all is specified
        // fallback to backend preference
        if (devices[0] != "all" && devices.size() >= 1) {
            if (isInitiator()) {
                backend_params["device_list"] = devices[rank];
            } else {
                backend_params["device_list"] = devices[rank - xferBenchConfig::num_initiator_dev];
            }
        }

        if (gethostname(hostname, 256)) {
            std::cerr << "Failed to get hostname" << std::endl;
            exit(EXIT_FAILURE);
        }

        backend_params["num_workers"] = std::to_string(xferBenchConfig::num_threads + 1);

        std::cout << "Init nixl worker, dev "
                  << (("all" == devices[0]) ? "all" : backend_params["device_list"]) << " rank "
                  << rank << ", type " << name << ", hostname " << hostname << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_LIBFABRIC)) {
        if (gethostname(hostname, 256)) {
            std::cerr << "Failed to get hostname" << std::endl;
            exit(EXIT_FAILURE);
        }

        // We need to make sure the Neuron runtime is initialized before initializing libfabric,
        // otherwise the FI_HMEM_NEURON backend will not be created. This issue has been fixed
        // upstream: https://github.com/ofiwg/libfabric/pull/11804
        int nc_count = neuronCoreCount();

        std::cout << "Init nixl worker, dev " << (("all" == devices[0]) ? "all" : devices[rank])
                  << " rank " << rank << ", type " << name << ", hostname " << hostname
                  << ", nc_count " << nc_count << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GDS)) {
        // Using default param values for GDS backend
        std::cout << "GDS backend" << std::endl;
        backend_params["batch_pool_size"] = std::to_string(xferBenchConfig::gds_batch_pool_size);
        backend_params["batch_limit"] = std::to_string(xferBenchConfig::gds_batch_limit);
        std::cout << "GDS batch pool size: " << xferBenchConfig::gds_batch_pool_size << std::endl;
        std::cout << "GDS batch limit: " << xferBenchConfig::gds_batch_limit << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GDS_MT)) {
        std::cout << "GDS_MT backend" << std::endl;
        backend_params["thread_count"] = std::to_string(xferBenchConfig::gds_mt_num_threads);
        std::cout << "GDS MT Num threads: " << xferBenchConfig::gds_mt_num_threads << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_POSIX)) {
        // Set API type parameter for POSIX backend
        if (xferBenchConfig::posix_api_type == XFERBENCH_POSIX_API_AIO) {
            backend_params["use_aio"] = "true";
            backend_params["use_uring"] = "false";
            backend_params["use_posix_aio"] = "false";
        } else if (xferBenchConfig::posix_api_type == XFERBENCH_POSIX_API_URING) {
            backend_params["use_aio"] = "false";
            backend_params["use_uring"] = "true";
            backend_params["use_posix_aio"] = "false";
        } else if (xferBenchConfig::posix_api_type == XFERBENCH_POSIX_API_POSIXAIO) {
            backend_params["use_aio"] = "false";
            backend_params["use_uring"] = "false";
            backend_params["use_posix_aio"] = "true";
        }
        std::cout << "POSIX backend with API type: " << xferBenchConfig::posix_api_type
                  << std::endl;
        backend_params["ios_pool_size"] = std::to_string(xferBenchConfig::posix_ios_pool_size);
        backend_params["kernel_queue_size"] =
            std::to_string(xferBenchConfig::posix_kernel_queue_size);
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GPUNETIO)) {
        std::cout << "GPUNETIO backend, network device " << devices[0] << " GPU device "
                  << xferBenchConfig::gpunetio_device_list << " OOB interface "
                  << xferBenchConfig::gpunetio_oob_list << std::endl;
        backend_params["network_devices"] = devices[0];
        backend_params["gpu_devices"] = xferBenchConfig::gpunetio_device_list;
        backend_params["oob_interface"] = xferBenchConfig::gpunetio_oob_list;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_MOONCAKE)) {
        std::cout << "Mooncake backend" << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_HF3FS)) {
        // Using default param values for HF3FS backend
        std::cout << "HF3FS backend iopool_size " << xferBenchConfig::hf3fs_iopool_size
                  << std::endl;
        backend_params["iopool_size"] = std::to_string(xferBenchConfig::hf3fs_iopool_size);
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_OBJ)) {
        // Using default param values for OBJ backend
        backend_params["access_key"] = xferBenchConfig::obj_access_key;
        backend_params["secret_key"] = xferBenchConfig::obj_secret_key;
        backend_params["session_token"] = xferBenchConfig::obj_session_token;
        backend_params["bucket"] = xferBenchConfig::obj_bucket_name;
        backend_params["scheme"] = xferBenchConfig::obj_scheme;
        backend_params["region"] = xferBenchConfig::obj_region;
        backend_params["use_virtual_addressing"] =
            xferBenchConfig::obj_use_virtual_addressing ? "true" : "false";
        backend_params["req_checksum"] = xferBenchConfig::obj_req_checksum;

        if (xferBenchConfig::obj_ca_bundle != "") {
            backend_params["ca_bundle"] = xferBenchConfig::obj_ca_bundle;
        }

        if (xferBenchConfig::obj_endpoint_override != "") {
            backend_params["endpoint_override"] = xferBenchConfig::obj_endpoint_override;
        }

        if (xferBenchConfig::obj_crt_min_limit > 0) {
            // Warn if both CRT and accelerated options are set - CRT takes precedence
            if (xferBenchConfig::obj_accelerated_enable) {
                std::cerr << "Warning: Both obj_crt_min_limit and obj_accelerated_enable are set. "
                          << "CRT client will be used (takes precedence over accelerated)."
                          << std::endl;
            }
            backend_params["crtMinLimit"] = std::to_string(xferBenchConfig::obj_crt_min_limit);
            std::cout << "OBJ backend with S3 CRT client enabled for objects >= "
                      << xferBenchConfig::obj_crt_min_limit << " bytes" << std::endl;
        } else if (xferBenchConfig::obj_accelerated_enable) {
            backend_params["accelerated"] = "true";
            std::cout << "OBJ backend with S3 Accelerated client enabled";
            if (!xferBenchConfig::obj_accelerated_type.empty()) {
                backend_params["type"] = xferBenchConfig::obj_accelerated_type;
                std::cout << " (type: " << xferBenchConfig::obj_accelerated_type << ")";
            }
            std::cout << std::endl;
        } else {
            std::cout << "OBJ backend with standard S3 enabled" << std::endl;
        }
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GUSLI)) {
        // GUSLI backend requires direct I/O - enable it automatically
        if (!xferBenchConfig::storage_enable_direct) {
            std::cout
                << "GUSLI backend: Automatically enabling storage_enable_direct for direct I/O"
                << std::endl;
            xferBenchConfig::storage_enable_direct = true;
        }

        // Parse and configure GUSLI devices from general device_list parameter
        int expected_num_devices =
            isInitiator() ? xferBenchConfig::num_initiator_dev : xferBenchConfig::num_target_dev;
        gusli_devices = parseGusliDeviceList(xferBenchConfig::device_list,
                                             xferBenchConfig::gusli_device_security,
                                             xferBenchConfig::gusli_device_byte_offsets,
                                             expected_num_devices);

        // Set GUSLI backend parameters
        backend_params["client_name"] = xferBenchConfig::gusli_client_name;
        backend_params["max_num_simultaneous_requests"] =
            std::to_string(xferBenchConfig::gusli_max_simultaneous_requests);

        // Generate config file if not explicitly provided
        if (xferBenchConfig::gusli_config_file.empty()) {
            backend_params["config_file"] = generateGusliConfigFile(gusli_devices);
        } else {
            backend_params["config_file"] = xferBenchConfig::gusli_config_file;
        }

        std::cout << "GUSLI backend initialized:" << std::endl;
        std::cout << "  Client name: " << xferBenchConfig::gusli_client_name << std::endl;
        std::cout << "  Max simultaneous requests: "
                  << xferBenchConfig::gusli_max_simultaneous_requests << std::endl;
        std::cout << "  Direct I/O: Enabled (required)" << std::endl;
        std::cout << "  Configured devices: " << gusli_devices.size() << std::endl;
        for (const auto &dev : gusli_devices) {
            std::cout << "    Device " << dev.device_id << " [" << dev.device_type
                      << "]: " << dev.device_path << " (" << dev.security_flags << ")"
                      << ", offset = " << dev.dev_offset << std::endl;
        }
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCCL)) {
        std::cout << "UCCL backend" << std::endl;
        backend_params["in_python"] = "0";
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_AZURE_BLOB)) {
        // Using default param values for AZURE_BLOB backend
        backend_params["account_url"] = xferBenchConfig::azure_blob_account_url;
        backend_params["container_name"] = xferBenchConfig::azure_blob_container_name;
        backend_params["connection_string"] = xferBenchConfig::azure_blob_connection_string;
        std::cout << "AZURE_BLOB backend" << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_INFINIA)) {
        // INFINIA backend - configuration via config file
        if (!xferBenchConfig::infinia_config_file.empty()) {
            backend_params["config_file"] = xferBenchConfig::infinia_config_file;
            std::cout << "INFINIA backend with config file: "
                      << xferBenchConfig::infinia_config_file << std::endl;
        } else {
            std::cout << "INFINIA backend (plugin will use environment variables or defaults)"
                      << std::endl;
            std::cout << "  Tip: Use --infinia_config_file to specify a config file" << std::endl;
        }
    } else {
        std::cerr << "Unsupported NIXLBench backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    CHECK_NIXL_ERROR(agent->createBackend(backend_name, backend_params, backend_engine),
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

namespace {

// RAII wrapper for an xfer-memory buffer. The factory make() selects heap or
// hugepages based on xferBenchConfig::use_hugepages; subclass destructors
// handle the matching deallocation. Code paths that store the raw address in
// xferBenchIOV use release() to hand off ownership and adopt() to take it
// back at cleanup time.
class nixlAlloc {
public:
    static std::unique_ptr<nixlAlloc>
    make(size_t size);
    static std::unique_ptr<nixlAlloc>
    adopt(void *addr, size_t size);

    virtual ~nixlAlloc() = default;

    nixlAlloc(const nixlAlloc &) = delete;
    nixlAlloc &
    operator=(const nixlAlloc &) = delete;

    void *
    addr() const noexcept {
        return addr_;
    }

    size_t
    size() const noexcept {
        return size_;
    }

    void *
    release() noexcept {
        void *p = addr_;
        addr_ = nullptr;
        return p;
    }

protected:
    nixlAlloc(void *addr, size_t size) : addr_(addr), size_(size) {}

    void *addr_;
    size_t size_;
};

class nixlHeapAlloc final : public nixlAlloc {
public:
    nixlHeapAlloc(void *addr, size_t size) : nixlAlloc(addr, size) {}

    ~nixlHeapAlloc() override {
        if (addr_) {
            free(addr_);
        }
    }
};

class nixlHugepagesAlloc final : public nixlAlloc {
public:
    nixlHugepagesAlloc(void *addr, size_t size) : nixlAlloc(addr, size) {}

    ~nixlHugepagesAlloc() override {
        if (!addr_) {
            return;
        }
        const size_t aligned = ROUND_UP(size_, HUGEPAGE_SIZE);
        if (munmap(addr_, aligned) != 0) {
            std::cerr << "Warning: Failed to unmap hugepage memory: " << strerror(errno)
                      << std::endl;
        }
    }
};

static std::unique_ptr<nixlAlloc>
makeHugepagesAlloc(size_t size) {
    const size_t aligned = ROUND_UP(size, HUGEPAGE_SIZE);
    void *addr = mmap(nullptr,
                      aligned,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB | MAP_POPULATE,
                      -1,
                      0);
    if (addr == MAP_FAILED) {
        std::cerr << "Error: Hugepage allocation failed (" << strerror(errno) << ")" << std::endl;
        std::cerr << "Hugepages may not be available. Check /proc/sys/vm/nr_hugepages and "
                  << "ensure sufficient hugepages are configured, or run without "
                  << "--use_hugepages" << std::endl;
        return nullptr;
    }

    assert(reinterpret_cast<uintptr_t>(addr) % HUGEPAGE_SIZE == 0);

    std::cout << "Allocated hugepage buffer: addr=0x" << std::hex
              << reinterpret_cast<uintptr_t>(addr) << std::dec << ", requested_size=" << size
              << ", allocated_size=" << aligned << std::endl;

    return std::make_unique<nixlHugepagesAlloc>(addr, size);
}

static std::unique_ptr<nixlAlloc>
makeHeapAlloc(size_t size) {
    void *addr = nullptr;
    int rc = posix_memalign(&addr, xferBenchConfig::page_size, size);
    if (rc != 0 || !addr) {
        std::cerr << "Failed to allocate " << size << " bytes of page-aligned DRAM memory"
                  << std::endl;
        return nullptr;
    }
    memset(addr, 0, size);
    return std::make_unique<nixlHeapAlloc>(addr, size);
}

std::unique_ptr<nixlAlloc>
nixlAlloc::make(size_t size) {
    if (size == 0) {
        std::cerr << "Invalid buffer size" << std::endl;
        return nullptr;
    }
    if (xferBenchConfig::page_size == 0) {
        std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
        return nullptr;
    }

    if (xferBenchConfig::use_hugepages) {
        return makeHugepagesAlloc(size);
    }
    return makeHeapAlloc(size);
}

std::unique_ptr<nixlAlloc>
nixlAlloc::adopt(void *addr, size_t size) {
    if (xferBenchConfig::use_hugepages) {
        return std::make_unique<nixlHugepagesAlloc>(addr, size);
    }
    return std::make_unique<nixlHeapAlloc>(addr, size);
}

} // namespace

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescDram(size_t buffer_size, int mem_dev_id) {
    auto alloc = nixlAlloc::make(buffer_size);
    if (!alloc) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of DRAM memory" << std::endl;
        return std::nullopt;
    }

    // Ownership of the underlying buffer is handed off to the iov; the matching
    // cleanupBasicDescDram() reclaims it via nixlAlloc::adopt().
    // TODO: Does device id need to be set for DRAM?
    return std::optional<xferBenchIOV>(
        std::in_place, reinterpret_cast<uintptr_t>(alloc->release()), buffer_size, mem_dev_id);
}

static std::optional<xferBenchIOV>
getVramDescNeuron(int devid, size_t buffer_size, uint8_t memset_value) {
    void *addr;
    CHECK_NEURON_ERROR(neuronMalloc(&addr, buffer_size, devid), "Failed to allocate nrt tensor");
    CHECK_NEURON_ERROR(neuronMemset(addr, memset_value, buffer_size),
                       "Failed to set device memory");

    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, devid);
}

static void
cleanupVramNeuron(xferBenchIOV &iov) {
    CHECK_NEURON_ERROR(neuronFree((void *)iov.addr), "Failed to free nrt tensor");
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

static void
cleanupVramCuda(xferBenchIOV &iov) {
    CHECK_CUDA_ERROR(cudaSetDevice(iov.devId), "Failed to set device");
    if (xferBenchConfig::enable_vmm) {
        CHECK_CUDA_DRIVER_ERROR(cuMemUnmap(iov.addr, iov.len), "Failed to unmap memory");
        CHECK_CUDA_DRIVER_ERROR(cuMemRelease(iov.handle), "Failed to release memory");
        CHECK_CUDA_DRIVER_ERROR(cuMemAddressFree(iov.addr, iov.padded_size),
                                "Failed to free reserved address");
    } else {
        CHECK_CUDA_ERROR(cudaFreeAsync((void *)iov.addr, 0), "Failed to deallocate CUDA buffer");
        CHECK_CUDA_ERROR(cudaStreamSynchronize(0), "Failed to synchronize stream 0");
    }
}

#endif /* HAVE_CUDA */

#if HAVE_ROCM
static std::optional<xferBenchIOV>
getVramDescRocm(int devid, size_t buffer_size, uint8_t memset_value) {
    void *addr;
    CHECK_CUDA_ERROR(hipMalloc(&addr, buffer_size), "Failed to allocate ROCm buffer");
    CHECK_CUDA_ERROR(hipMemset(addr, memset_value, buffer_size), "Failed to set device memory");
    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, devid);
}

static void
cleanupVramRocm(xferBenchIOV &iov) {
    CHECK_CUDA_ERROR(hipSetDevice(iov.devId), "Failed to set device");
    CHECK_CUDA_ERROR(hipFree((void *)iov.addr), "Failed to deallocate ROCm buffer");
}
#endif /* HAVE_ROCM */

static std::optional<xferBenchIOV>
getVramDesc(int devid, size_t buffer_size, bool isInit) {
    uint8_t memset_value =
        isInit ? XFERBENCH_INITIATOR_BUFFER_ELEMENT : XFERBENCH_TARGET_BUFFER_ELEMENT;

    if (neuronCoreCount() > 0) {
        return getVramDescNeuron(devid, buffer_size, memset_value);
    }

#if HAVE_CUDA
    CHECK_CUDA_ERROR(cudaSetDevice(devid), "Failed to set device");
    if (xferBenchConfig::enable_vmm) {
        return getVramDescCudaVmm(devid, buffer_size, memset_value);
    } else {
        return getVramDescCuda(devid, buffer_size, memset_value);
    }
#elif HAVE_ROCM
    CHECK_CUDA_ERROR(hipSetDevice(devid), "Failed to set device");
    return getVramDescRocm(devid, buffer_size, memset_value);
#else
    std::cerr << "VRAM not supported without CUDA, ROCm or Neuron" << std::endl;
    return std::nullopt;
#endif
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescVram(size_t buffer_size, int mem_dev_id) {
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

// Helper to open a single file with appropriate flags
static std::optional<xferFileState>
openFileWithFlags(const std::string &file_name, int flags) {
    uint64_t file_size = 0;
    if (XFERBENCH_OP_READ == xferBenchConfig::op_type) {
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
createFileFds(std::string name, int num_files, const std::vector<std::string> &filenames = {}) {
    std::vector<xferFileState> fds;
    int flags = O_RDWR | O_CREAT | O_LARGEFILE;

    if (!xferBenchConfig::isStorageBackend()) {
        std::cerr << "Unknown storage backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    if (xferBenchConfig::storage_enable_direct) {
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
            auto fstate = openFileWithFlags(file_name, flags);
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
    const std::string file_path = xferBenchConfig::filepath != "" ?
        xferBenchConfig::filepath :
        std::filesystem::current_path().string();
    std::string file_backend = xferBenchConfig::backend;
    std::transform(file_backend.begin(), file_backend.end(), file_backend.begin(), ::tolower);
    const std::string file_name_prefix = "/nixlbench_" + file_backend + "_test_file_";

    for (int i = 0; i < num_files; i++) {
        std::string file_name = file_path + file_name_prefix + name + "_" + std::to_string(i);
        std::cout << "Creating file: " << file_name << std::endl;

        auto fstate = openFileWithFlags(file_name, flags);
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
xferBenchNixlWorker::initBasicDescFile(size_t buffer_size, xferFileState &fstate, int mem_dev_id) {
    int fd = fstate.fd;
    uint64_t start_offset = fstate.offset;
    uint64_t end_offset = fstate.offset + buffer_size;
    auto ret = std::optional<xferBenchIOV>(std::in_place, fstate.offset, buffer_size, fd);

    fstate.offset = end_offset;

    // If in READ mode, only write if the region is not already present in the file
    if (XFERBENCH_OP_READ == xferBenchConfig::op_type && end_offset <= fstate.file_size) {
        return ret;
    }

    // Fill up with data
    auto alloc = nixlAlloc::make(buffer_size);
    if (!alloc) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of memory" << std::endl;
        return std::nullopt;
    }
    void *buf = alloc->addr();

    // File is always initialized with XFERBENCH_TARGET_BUFFER_ELEMENT
    memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);

    size_t offset = start_offset;
    char *write_ptr = static_cast<char *>(buf);
    size_t remaining = buffer_size;
    while (remaining > 0) {
        ssize_t rc = pwrite(fd, write_ptr, remaining, offset);
        if (rc < 0) {
            std::cerr << "Failed to write to file: " << fd << " with error: " << strerror(errno)
                      << std::endl;
            return std::nullopt;
        }

        remaining -= rc;
        offset += rc;
        write_ptr += rc;
    }

    if (end_offset > fstate.file_size) fstate.file_size = end_offset;

    return ret;
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescObj(size_t buffer_size, int mem_dev_id, std::string name) {
    return std::optional<xferBenchIOV>(std::in_place, 0, buffer_size, mem_dev_id, name);
}

void
xferBenchNixlWorker::cleanupBasicDescDram(xferBenchIOV &iov) {
    // Reclaim ownership of the buffer handed out by initBasicDescDram(); the
    // returned wrapper goes out of scope here and frees the buffer.
    nixlAlloc::adopt(reinterpret_cast<void *>(iov.addr), iov.len);
}

void
xferBenchNixlWorker::cleanupBasicDescVram(xferBenchIOV &iov) {
    if (neuronCoreCount() > 0) {
        cleanupVramNeuron(iov);
        return;
    }

#if HAVE_CUDA
    cleanupVramCuda(iov);
#elif HAVE_ROCM
    cleanupVramRocm(iov);
#else
    std::cerr << "VRAM not supported without CUDA, ROCm or Neuron" << std::endl;
#endif
}

void
xferBenchNixlWorker::cleanupBasicDescFile(xferBenchIOV &iov) {
    close(iov.devId);
}

void
xferBenchNixlWorker::cleanupBasicDescObj(xferBenchIOV &iov) {
    if (!xferBenchUtils::rmObj(iov.metaInfo)) {
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
    if (xferBenchConfig::storage_enable_direct) flags |= O_DIRECT;

    int fd = open(device.device_path.c_str(), flags, 0744);
    if (fd < 0) {
        std::cerr << "Failed to open GUSLI file: " << device.device_path << ": " << strerror(errno)
                  << std::endl;
        return false;
    }

    // Sample one page at the offset GUSLI will read from
    bool needs_write = true;
    if (auto check_alloc = nixlAlloc::make(xferBenchConfig::page_size)) {
        void *check_buf = check_alloc->addr();
        ssize_t rd = pread(fd, check_buf, xferBenchConfig::page_size, device.dev_offset);
        if (rd == (ssize_t)xferBenchConfig::page_size) {
            needs_write = false;
            uint8_t *bytes = static_cast<uint8_t *>(check_buf);
            for (ssize_t i = 0; i < rd; i++) {
                if (bytes[i] != XFERBENCH_TARGET_BUFFER_ELEMENT) {
                    needs_write = true;
                    break;
                }
            }
        }
    }

    if (needs_write) {
        std::cout << "Warning: GUSLI file '" << device.device_path << "' at offset "
                  << device.dev_offset << " does not contain expected pattern (0x" << std::hex
                  << (int)XFERBENCH_TARGET_BUFFER_ELEMENT << std::dec << "). Overwriting."
                  << std::endl;

        auto alloc = nixlAlloc::make(size);
        if (!alloc) {
            close(fd);
            return false;
        }
        void *buf = alloc->addr();
        memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, size);

        size_t remaining = size;
        size_t offset = device.dev_offset;
        char *write_ptr = static_cast<char *>(buf);
        while (remaining > 0) {
            ssize_t rc = pwrite(fd, write_ptr, remaining, offset);
            if (rc < 0) {
                std::cerr << "Failed to write to " << device.device_path << " at offset " << offset
                          << ": " << strerror(errno) << std::endl;
                close(fd);
                return false;
            }
            remaining -= rc;
            offset += rc;
            write_ptr += rc;
        }
    } else {
        std::cout << "GUSLI file '" << device.device_path << "' at offset " << device.dev_offset
                  << " already contains expected pattern (0x" << std::hex
                  << (int)XFERBENCH_TARGET_BUFFER_ELEMENT << std::dec
                  << "). Skipping initialization." << std::endl;
    }

    close(fd);
    return true;
}

/** Allocate and register memory descriptors for each thread. */
std::vector<std::vector<xferBenchIOV>>
xferBenchNixlWorker::allocateMemory(int num_threads) {
    std::vector<std::vector<xferBenchIOV>> iov_lists;
    size_t i, buffer_size, num_devices = 0;
    nixl_opt_args_t opt_args;

    if (isInitiator()) {
        num_devices = xferBenchConfig::num_initiator_dev;
    } else if (isTarget()) {
        num_devices = xferBenchConfig::num_target_dev;
    }
    buffer_size = xferBenchConfig::total_buffer_size / (num_devices * num_threads);

    if (xferBenchConfig::storage_enable_direct) {
        if (xferBenchConfig::page_size == 0) {
            std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
            exit(EXIT_FAILURE);
        }
        buffer_size =
            ((buffer_size + xferBenchConfig::page_size - 1) / xferBenchConfig::page_size) *
            xferBenchConfig::page_size;
    }

    opt_args.backends.push_back(backend_engine);

    if (xferBenchConfig::isObjStorageBackend()) {
        buffer_size = xferBenchConfig::max_block_size;

        struct timeval tv;
        gettimeofday(&tv, nullptr);
        uint64_t timestamp = tv.tv_sec * 1000000ULL + tv.tv_usec;

        for (int list_idx = 0; list_idx < num_threads; list_idx++) {
            std::vector<xferBenchIOV> iov_list;
            for (i = 0; i < num_devices; i++) {
                std::optional<xferBenchIOV> basic_desc;
                std::string unique_name = "nixlbench_obj" + std::to_string(list_idx) + "_" +
                    std::to_string(i) + "_" + std::to_string(timestamp);

                if (xferBenchConfig::op_type == XFERBENCH_OP_READ) {
                    if (!xferBenchUtils::putObj(buffer_size, unique_name)) {
                        std::cerr << "Failed to put object: " << unique_name << std::endl;
                        continue;
                    }
                }

                int obj_dev_id = list_idx * num_devices + i;
                basic_desc = initBasicDescObj(buffer_size, obj_dev_id, unique_name);
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
    } else if (XFERBENCH_BACKEND_GUSLI == xferBenchConfig::backend) {
        // GUSLI backend uses block device descriptors
        if (gusli_devices.empty()) {
            std::cerr << "No GUSLI devices configured" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (xferBenchConfig::op_type == XFERBENCH_OP_READ) {
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
    } else if (xferBenchConfig::isStorageBackend()) {
        int num_buffers = num_threads * num_devices;
        int num_files = xferBenchConfig::num_files;
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
        if (!xferBenchConfig::filenames.empty()) {
            std::string filename;
            std::stringstream ss(xferBenchConfig::filenames);
            while (std::getline(ss, filename, ',')) {
                filenames.push_back(filename);
            }
        }
        remote_fds = createFileFds(getName(), num_files, filenames);
        if (remote_fds.empty()) {
            std::cerr << "Failed to create " << xferBenchConfig::backend << " file" << std::endl;
            exit(EXIT_FAILURE);
        }

        int file_idx = 0;
        for (int list_idx = 0; list_idx < num_threads; list_idx++) {
            std::vector<xferBenchIOV> iov_list;
            for (i = 0; i < num_devices; i++) {
                std::optional<xferBenchIOV> basic_desc;
                basic_desc = initBasicDescFile(buffer_size, remote_fds[file_idx], i);
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
                int mem_dev_id = (XFERBENCH_BACKEND_GUSLI == xferBenchConfig::backend &&
                                  !gusli_devices.empty()) ?
                    gusli_devices[i].device_id :
                    i;
                basic_desc = initBasicDescDram(buffer_size, mem_dev_id);
                break;
            }
            case VRAM_SEG:
                basic_desc = initBasicDescVram(buffer_size, i);
                break;
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
        if (seg_type == DRAM_SEG && xferBenchConfig::check_consistency) {
            for (auto &iov : iov_list) {
                if (isInitiator()) {
                    memset((void *)iov.addr, XFERBENCH_INITIATOR_BUFFER_ELEMENT, iov.len);
                } else if (isTarget()) {
                    memset((void *)iov.addr, XFERBENCH_TARGET_BUFFER_ELEMENT, iov.len);
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

    // Ordering invariants:
    // 1. Deregister remote IOVs before local IOVs
    //    (remote registrations may reference local buffers).
    // 2. Call deregisterMem() before each IOV cleanup
    //    (cleanup destroys resources that backends need).
    if (xferBenchConfig::isObjStorageBackend()) {
        for (auto &iov_list : remote_iovs) {
            nixl_reg_dlist_t desc_list(OBJ_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");
            for (auto &iov : iov_list) {
                cleanupBasicDescObj(iov);
            }
        }
    } else if (xferBenchConfig::backend == XFERBENCH_BACKEND_GUSLI) {
        for (auto &iov_list : remote_iovs) {
            nixl_reg_dlist_t desc_list(BLK_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");
            for (auto &iov : iov_list) {
                cleanupBasicDescBlk(iov);
            }
        }
    } else if (xferBenchConfig::isStorageBackend()) {
        for (auto &iov_list : remote_iovs) {
            nixl_reg_dlist_t desc_list(FILE_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");
        }
        // Close each backing fd exactly once, after all deregistrations complete.
        std::unordered_set<int> closed_fds;
        for (auto &iov_list : remote_iovs) {
            for (auto &iov : iov_list) {
                if (closed_fds.insert(iov.devId).second) {
                    cleanupBasicDescFile(iov);
                }
            }
        }
    }

    for (auto &iov_list : iov_lists) {
        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");

        for (auto &iov : iov_list) {
            switch (seg_type) {
            case DRAM_SEG:
                cleanupBasicDescDram(iov);
                break;
            case VRAM_SEG:
                cleanupBasicDescVram(iov);
                break;
            default:
                std::cerr << "Unsupported mem type: " << seg_type << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}

int
xferBenchNixlWorker::exchangeMetadata() {
    int meta_sz, ret = 0;

    // Skip metadata exchange for storage backends or when ETCD is not available
    if (xferBenchConfig::isStorageBackend()) {
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

        if (IS_PAIRWISE_AND_SG()) {
            srcrank = rt->getRank() + xferBenchConfig::num_initiator_dev;
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

    if (xferBenchConfig::isStorageBackend()) {
        size_t fd_idx = 0;
        uint64_t file_offset = 0;
        for (size_t list_idx = 0; list_idx < local_iovs.size(); list_idx++) {
            const auto &iov_list = local_iovs[list_idx];
            std::vector<xferBenchIOV> remote_iov_list;
            size_t num_devices = iov_list.size();
            for (size_t devidx = 0; devidx < num_devices; devidx++) {
                const auto &iov = iov_list[devidx];
                if (xferBenchConfig::isObjStorageBackend()) {
                    std::optional<xferBenchIOV> basic_desc;
                    int obj_dev_id = list_idx * num_devices + devidx;
                    basic_desc = initBasicDescObj(iov.len, obj_dev_id, iov.metaInfo);
                    if (basic_desc) {
                        remote_iov_list.push_back(basic_desc.value());
                    }
                } else if (XFERBENCH_BACKEND_GUSLI == xferBenchConfig::backend) {
                    xferBenchIOV iov_remote(iov);
                    iov_remote.addr = gusli_devices[devidx].dev_offset + file_offset;
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
            if (XFERBENCH_BACKEND_GUSLI == xferBenchConfig::backend) {
                file_offset += block_size;
            }
        }
    } else {
        for (const auto &local_iov : local_iovs) {
            nixlSerDes ser_des;
            nixl_xfer_dlist_t local_desc(seg_type);

            iovListToNixlXferDlist(local_iov, local_desc);

            if (isTarget()) {
                int destrank;
                if (IS_PAIRWISE_AND_SG()) {
                    destrank = rt->getRank() - xferBenchConfig::num_target_dev;
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
                if (IS_PAIRWISE_AND_SG()) {
                    srcrank = rt->getRank() + xferBenchConfig::num_initiator_dev;
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

// Helper to prepare transfer descriptors based on backend type
static void
prepareTransferDescriptors(nixl_xfer_dlist_t &local_desc,
                           nixl_xfer_dlist_t &remote_desc,
                           const std::vector<xferBenchIOV> &local_iov,
                           const std::vector<xferBenchIOV> &remote_iov) {
    // Set remote descriptor type based on backend
    if (xferBenchConfig::isObjStorageBackend()) {
        remote_desc = nixl_xfer_dlist_t(OBJ_SEG);
    } else if (XFERBENCH_BACKEND_GUSLI == xferBenchConfig::backend) {
        remote_desc = nixl_xfer_dlist_t(BLK_SEG);
    } else if (xferBenchConfig::isStorageBackend()) {
        remote_desc = nixl_xfer_dlist_t(FILE_SEG);
    }

    iovListToNixlXferDlist(local_iov, local_desc);
    iovListToNixlXferDlist(remote_iov, remote_desc);
}

static nixl_mem_t
getRemoteSegType() {
    if (xferBenchConfig::isObjStorageBackend()) {
        return OBJ_SEG;
    } else if (XFERBENCH_BACKEND_GUSLI == xferBenchConfig::backend) {
        return BLK_SEG;
    } else if (xferBenchConfig::isStorageBackend()) {
        return FILE_SEG;
    }
    return GET_SEG_TYPE(false);
}

// Fill a transfer buffer with a single byte. For VRAM the fill targets the
// device that owns the buffer, so multi-GPU runs poison the right device.
static void
fillTransferBuffer(nixl_mem_t seg_type, uintptr_t addr, size_t len, uint8_t value, int dev_id) {
    if (seg_type != VRAM_SEG) {
        std::memset(reinterpret_cast<void *>(addr), value, len);
        return;
    }
    if (neuronCoreCount() > 0) {
        std::vector<uint8_t> src(len, value);
        CHECK_NEURON_ERROR(
            neuronMemcpy(reinterpret_cast<void *>(addr), src.data(), len, neuronMemcpyHostToDevice),
            "fillTransferBuffer: neuronMemcpy failed");
        return;
    }
#if HAVE_CUDA
    CHECK_CUDA_ERROR(cudaSetDevice(dev_id), "fillTransferBuffer: cudaSetDevice failed");
    CHECK_CUDA_ERROR(cudaMemset(reinterpret_cast<void *>(addr), value, len),
                     "fillTransferBuffer: cudaMemset failed");
#elif HAVE_ROCM
    CHECK_CUDA_ERROR(hipSetDevice(dev_id), "fillTransferBuffer: hipSetDevice failed");
    CHECK_CUDA_ERROR(hipMemset(reinterpret_cast<void *>(addr), value, len),
                     "fillTransferBuffer: hipMemset failed");
#else
    (void)dev_id;
    std::cerr << "VRAM consistency check needs CUDA, ROCm or Neuron" << std::endl;
    exit(EXIT_FAILURE);
#endif
}

// Return a host-readable view of a transfer buffer. DRAM is returned in place;
// VRAM is copied from its owning device into host_scratch (never dereferenced
// as host memory directly).
static const uint8_t *
hostView(nixl_mem_t seg_type,
         uintptr_t addr,
         size_t len,
         std::vector<uint8_t> &host_scratch,
         int dev_id) {
    if (seg_type != VRAM_SEG) {
        return reinterpret_cast<const uint8_t *>(addr);
    }
    host_scratch.resize(len);
    if (neuronCoreCount() > 0) {
        CHECK_NEURON_ERROR(
            neuronMemcpy(
                host_scratch.data(), reinterpret_cast<void *>(addr), len, neuronMemcpyDeviceToHost),
            "hostView: neuronMemcpy failed");
        return host_scratch.data();
    }
#if HAVE_CUDA
    CHECK_CUDA_ERROR(cudaSetDevice(dev_id), "hostView: cudaSetDevice failed");
    CHECK_CUDA_ERROR(
        cudaMemcpy(
            host_scratch.data(), reinterpret_cast<void *>(addr), len, cudaMemcpyDeviceToHost),
        "hostView: cudaMemcpy failed");
#elif HAVE_ROCM
    CHECK_CUDA_ERROR(hipSetDevice(dev_id), "hostView: hipSetDevice failed");
    CHECK_CUDA_ERROR(
        hipMemcpy(host_scratch.data(), reinterpret_cast<void *>(addr), len, hipMemcpyDeviceToHost),
        "hostView: hipMemcpy failed");
#else
    (void)dev_id;
    std::cerr << "VRAM consistency check needs CUDA, ROCm or Neuron" << std::endl;
    exit(EXIT_FAILURE);
#endif
    return host_scratch.data();
}

// Verify a storage WRITE without touching local disk.
//
// The data has just been written to the remote backend. Rather than download
// the object to a local file and re-read it (the file/device path other storage
// backends use), read the objects back over the same transport into the
// initiator buffers and compare them to the byte the initiator wrote
// (XFERBENCH_INITIATOR_BUFFER_ELEMENT). This verifies the round-trip end to end
// and needs no local copy, so it also covers backends that materialize none.
//
// The buffers are poisoned with a different byte first, so a read that moves
// nothing is caught instead of passing by coincidence. `local_lists` are the
// initiator buffers; `remote_lists` the matching object descriptors.
bool
xferBenchNixlWorker::verifyWriteByReadback(std::vector<std::vector<xferBenchIOV>> &local_lists,
                                           std::vector<std::vector<xferBenchIOV>> &remote_lists) {
    constexpr uint8_t poison = 0x00;
    static_assert(poison != XFERBENCH_INITIATOR_BUFFER_ELEMENT,
                  "poison value must differ from the initiator sentinel");

    if (local_lists.size() != remote_lists.size()) {
        std::cerr << "verifyWriteByReadback: list count mismatch (" << local_lists.size()
                  << " local vs " << remote_lists.size() << " remote)" << std::endl;
        return false;
    }

    bool pass = true;
    for (size_t list_idx = 0; list_idx < local_lists.size(); list_idx++) {
        if (local_lists[list_idx].size() != remote_lists[list_idx].size()) {
            std::cerr << "verifyWriteByReadback: descriptor count mismatch in list " << list_idx
                      << std::endl;
            return false;
        }

        for (const auto &iov : local_lists[list_idx]) {
            fillTransferBuffer(seg_type, iov.addr, iov.len, poison, iov.devId);
        }

        nixl_xfer_dlist_t local_desc(seg_type);
        nixl_xfer_dlist_t remote_desc(getRemoteSegType());
        prepareTransferDescriptors(
            local_desc, remote_desc, local_lists[list_idx], remote_lists[list_idx]);

        std::string target = xferBenchConfig::isStorageBackend() ? "initiator" : "target";
        nixl_opt_args_t params;
        nixlXferReqH *req = nullptr;
        nixl_status_t rc =
            agent->createXferReq(NIXL_READ, local_desc, remote_desc, target, req, &params);
        if (NIXL_SUCCESS != rc) {
            std::cerr << "verifyWriteByReadback: createXferReq failed for list " << list_idx
                      << std::endl;
            return false;
        }
        rc = agent->postXferReq(req);
        if (NIXL_SUCCESS != rc && NIXL_IN_PROG != rc) {
            std::cerr << "verifyWriteByReadback: postXferReq failed for list " << list_idx
                      << std::endl;
            agent->releaseXferReq(req);
            return false;
        }
        while (NIXL_IN_PROG == agent->getXferStatus(req)) {}
        agent->releaseXferReq(req);

        size_t iov_idx = 0;
        for (const auto &iov : local_lists[list_idx]) {
            std::vector<uint8_t> host_scratch;
            const uint8_t *bytes = hostView(seg_type, iov.addr, iov.len, host_scratch, iov.devId);
            for (size_t b = 0; b < iov.len; b++) {
                if (bytes[b] != XFERBENCH_INITIATOR_BUFFER_ELEMENT) {
                    std::cerr << "Consistency check failed for iov " << list_idx << ":" << iov_idx
                              << " at byte " << b << std::endl;
                    pass = false;
                    break;
                }
            }
            iov_idx++;
        }
    }

    if (!pass) {
        std::cerr << "Consistency check failed" << std::endl;
    }
    return pass;
}

// Register local and remote memory with the agent.
static nixl_status_t
registerIterationMem(nixlAgent *agent,
                     const std::vector<xferBenchIOV> &local_iov,
                     const std::vector<xferBenchIOV> &remote_iov,
                     nixlBackendH *backend_engine) {
    nixl_opt_args_t reg_args;
    reg_args.backends.push_back(backend_engine);

    nixl_reg_dlist_t local_reg(GET_SEG_TYPE(true));
    iovListToNixlRegDlist(local_iov, local_reg);
    nixl_status_t rc = agent->registerMem(local_reg, &reg_args);
    if (rc != NIXL_SUCCESS) {
        return rc;
    }

    nixl_reg_dlist_t remote_reg(getRemoteSegType());
    iovListToNixlRegDlist(remote_iov, remote_reg);
    rc = agent->registerMem(remote_reg, &reg_args);
    if (rc != NIXL_SUCCESS) {
        return rc;
    }

    return NIXL_SUCCESS;
}

// Deregister local and remote memory from the agent.
static nixl_status_t
deregisterIterationMem(nixlAgent *agent,
                       const std::vector<xferBenchIOV> &local_iov,
                       const std::vector<xferBenchIOV> &remote_iov,
                       nixlBackendH *backend_engine) {
    nixl_opt_args_t reg_args;
    reg_args.backends.push_back(backend_engine);

    nixl_reg_dlist_t local_reg(GET_SEG_TYPE(true));
    iovListToNixlRegDlist(local_iov, local_reg);
    nixl_status_t rc = agent->deregisterMem(local_reg, &reg_args);
    if (rc != NIXL_SUCCESS) {
        return rc;
    }

    nixl_reg_dlist_t remote_reg(getRemoteSegType());
    iovListToNixlRegDlist(remote_iov, remote_reg);
    rc = agent->deregisterMem(remote_reg, &reg_args);
    if (rc != NIXL_SUCCESS) {
        return rc;
    }

    return NIXL_SUCCESS;
}

// Per-slot state for execTransferLoop. A slot owns its slice of the IOV
// vector for the lifetime of the run; req/registered track the current
// nixlXferReqH and registration state so the prepare/post/recycle helpers
// can be called idempotently.
struct slotState {
    std::vector<xferBenchIOV> local_iov;
    std::vector<xferBenchIOV> remote_iov;
    nixlXferReqH *req = nullptr;
    bool in_flight = false;
    bool registered = false;
    nixlTime::us_t post_ts = 0;
    nixlDlistH *prep_local_dlist = nullptr;
    nixlDlistH *prep_remote_dlist = nullptr;
    std::vector<int> indices;
};

// Register memory (if --reregister_mem) and create the XferReq for a slot
// that doesn't already have one. Records the wall-clock time as
// prepare_duration.
static nixl_status_t
prepareSlot(nixlAgent *agent,
            nixlBackendH *backend_engine,
            const nixl_xfer_op_t op,
            const std::string &target,
            nixl_opt_args_t &params,
            xferBenchStats &thread_stats,
            slotState &slot) {
    const bool reregister = xferBenchConfig::reregister_mem;
    const nixlTime::us_t prep_start = nixlTime::getUs();

    if (reregister && !slot.registered) {
        nixl_status_t rc =
            registerIterationMem(agent, slot.local_iov, slot.remote_iov, backend_engine);
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
        slot.registered = true;
    }

    if (!slot.req) {
        nixl_status_t rc;
        if (xferBenchConfig::prepared_xfer) {
            if (!slot.prep_local_dlist) {
                nixl_xfer_dlist_t ld(GET_SEG_TYPE(true));
                nixl_xfer_dlist_t rd(GET_SEG_TYPE(false));
                prepareTransferDescriptors(ld, rd, slot.local_iov, slot.remote_iov);
                rc = agent->prepXferDlist(NIXL_INIT_AGENT, ld, slot.prep_local_dlist, &params);
                if (rc != NIXL_SUCCESS) {
                    return rc;
                }
                rc = agent->prepXferDlist(target, rd, slot.prep_remote_dlist, &params);
                if (rc != NIXL_SUCCESS) {
                    return rc;
                }
                slot.indices.resize(ld.descCount());
                std::iota(slot.indices.begin(), slot.indices.end(), 0);
            }
            rc = agent->makeXferReq(op,
                                    slot.prep_local_dlist,
                                    slot.indices,
                                    slot.prep_remote_dlist,
                                    slot.indices,
                                    slot.req,
                                    &params);
        } else {
            nixl_xfer_dlist_t ld(GET_SEG_TYPE(true));
            nixl_xfer_dlist_t rd(GET_SEG_TYPE(false));
            prepareTransferDescriptors(ld, rd, slot.local_iov, slot.remote_iov);
            rc = agent->createXferReq(op, ld, rd, target, slot.req, &params);
        }
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
    }

    thread_stats.prepare_duration.add(nixlTime::getUs() - prep_start);
    return NIXL_SUCCESS;
}

// Post the slot's request and record post_duration. Marks the slot
// in-flight; the caller drives completion via getXferStatus.
static nixl_status_t
postSlot(nixlAgent *agent, xferBenchStats &thread_stats, slotState &slot) {
    const nixlTime::us_t post_start = nixlTime::getUs();
    nixl_status_t rc = agent->postXferReq(slot.req);
    if (rc != NIXL_SUCCESS && rc != NIXL_IN_PROG) {
        return rc;
    }
    slot.post_ts = nixlTime::getUs();
    thread_stats.post_duration.add(slot.post_ts - post_start);
    slot.in_flight = true;
    return NIXL_SUCCESS;
}

// Tear down the request and (if --reregister_mem) the registration so the
// next prepareSlot exercises the full lifecycle.
static nixl_status_t
recycleSlot(nixlAgent *agent, nixlBackendH *backend_engine, slotState &slot) {
    if (slot.req) {
        agent->releaseXferReq(slot.req);
        slot.req = nullptr;
    }
    if (xferBenchConfig::prepared_xfer) {
        if (slot.prep_local_dlist) {
            agent->releasedDlistH(slot.prep_local_dlist);
            slot.prep_local_dlist = nullptr;
        }
        if (slot.prep_remote_dlist) {
            agent->releasedDlistH(slot.prep_remote_dlist);
            slot.prep_remote_dlist = nullptr;
        }
        slot.indices.clear();
    }
    if (xferBenchConfig::reregister_mem && slot.registered) {
        nixl_status_t rc =
            deregisterIterationMem(agent, slot.local_iov, slot.remote_iov, backend_engine);
        slot.registered = false;
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
    }
    return NIXL_SUCCESS;
}

// Best-effort teardown for early-exit / error paths.
static void
cleanupSlots(nixlAgent *agent, nixlBackendH *backend_engine, std::vector<slotState> &slots) {
    for (auto &slot : slots) {
        if (slot.req) {
            agent->releaseXferReq(slot.req);
            slot.req = nullptr;
        }
        if (slot.prep_local_dlist) {
            agent->releasedDlistH(slot.prep_local_dlist);
            slot.prep_local_dlist = nullptr;
        }
        if (slot.prep_remote_dlist) {
            agent->releasedDlistH(slot.prep_remote_dlist);
            slot.prep_remote_dlist = nullptr;
        }
        if (xferBenchConfig::reregister_mem && slot.registered) {
            deregisterIterationMem(agent, slot.local_iov, slot.remote_iov, backend_engine);
            slot.registered = false;
        }
    }
}

// Run num_iter transfers using a sliding window of pipeline_depth in-flight
// requests. Depth=1 collapses to the original "one create, N posts, one
// release" baseline (the previous execTransferIterations); --recreate_xfer
// tears down and rebuilds the request between iterations, --reregister_mem
// adds the matching registerMem/deregisterMem cycle.
static int
execTransferLoop(nixlAgent *agent,
                 nixlBackendH *backend_engine,
                 const nixl_xfer_op_t op,
                 const std::string &target,
                 nixl_opt_args_t &params,
                 const int num_iter,
                 xferBenchStats &thread_stats,
                 const std::vector<xferBenchIOV> &local_iov,
                 const std::vector<xferBenchIOV> &remote_iov,
                 const std::atomic<int> *terminate_ptr = nullptr) {
    const int depth = std::min(xferBenchConfig::pipeline_depth, num_iter);
    if (depth < xferBenchConfig::pipeline_depth) {
        std::cout << "Warning: pipeline_depth (" << xferBenchConfig::pipeline_depth
                  << ") exceeds num_iter (" << num_iter << "), capping to " << depth << std::endl;
    }
    const bool recreate = xferBenchConfig::recreate_xfer;

    if (local_iov.size() % depth != 0) {
        std::cerr << "Error: descriptor count (" << local_iov.size()
                  << ") is not evenly divisible by pipeline depth (" << depth << ")" << std::endl;
        return -1;
    }
    const size_t entries_per_slot = local_iov.size() / depth;

    std::vector<slotState> slots(depth);
    for (int s = 0; s < depth; s++) {
        auto lb = local_iov.begin() + s * entries_per_slot;
        auto rb = remote_iov.begin() + s * entries_per_slot;
        slots[s].local_iov.assign(lb, lb + entries_per_slot);
        slots[s].remote_iov.assign(rb, rb + entries_per_slot);
    }

    int issued = 0;
    int completed = 0;

    for (int s = 0; s < depth; s++) {
        if (terminate_ptr && terminate_ptr->load()) [[unlikely]] {
            cleanupSlots(agent, backend_engine, slots);
            return -1;
        }
        nixl_status_t rc =
            prepareSlot(agent, backend_engine, op, target, params, thread_stats, slots[s]);
        if (rc != NIXL_SUCCESS) [[unlikely]] {
            std::cerr << "prepareSlot failed for slot " << s << ": "
                      << nixlEnumStrings::statusStr(rc) << std::endl;
            cleanupSlots(agent, backend_engine, slots);
            return -1;
        }
        rc = postSlot(agent, thread_stats, slots[s]);
        if (rc != NIXL_SUCCESS) [[unlikely]] {
            std::cerr << "postSlot failed for slot " << s << ": " << nixlEnumStrings::statusStr(rc)
                      << std::endl;
            cleanupSlots(agent, backend_engine, slots);
            return -1;
        }
        issued++;
    }

    while (completed < num_iter) {
        if (terminate_ptr && terminate_ptr->load()) [[unlikely]] {
            cleanupSlots(agent, backend_engine, slots);
            return -1;
        }
        for (int s = 0; s < depth; s++) {
            if (!slots[s].in_flight) {
                continue;
            }

            nixl_status_t rc = agent->getXferStatus(slots[s].req);
            if (rc == NIXL_IN_PROG) {
                continue;
            }

            if (rc != NIXL_SUCCESS) [[unlikely]] {
                std::cerr << "Transfer failed on slot " << s << ": "
                          << nixlEnumStrings::statusStr(rc) << std::endl;
                cleanupSlots(agent, backend_engine, slots);
                return -1;
            }

            completed++;
            thread_stats.transfer_duration.add(nixlTime::getUs() - slots[s].post_ts);
            slots[s].in_flight = false;

            if (issued >= num_iter) {
                continue;
            }

            if (terminate_ptr && terminate_ptr->load()) [[unlikely]] {
                cleanupSlots(agent, backend_engine, slots);
                return -1;
            }

            if (recreate) {
                rc = recycleSlot(agent, backend_engine, slots[s]);
                if (rc != NIXL_SUCCESS) [[unlikely]] {
                    std::cerr << "recycleSlot failed for slot " << s << ": "
                              << nixlEnumStrings::statusStr(rc) << std::endl;
                    cleanupSlots(agent, backend_engine, slots);
                    return -1;
                }
                rc = prepareSlot(agent, backend_engine, op, target, params, thread_stats, slots[s]);
                if (rc != NIXL_SUCCESS) [[unlikely]] {
                    std::cerr << "prepareSlot failed on resubmit for slot " << s << ": "
                              << nixlEnumStrings::statusStr(rc) << std::endl;
                    cleanupSlots(agent, backend_engine, slots);
                    return -1;
                }
            }

            rc = postSlot(agent, thread_stats, slots[s]);
            if (rc != NIXL_SUCCESS) [[unlikely]] {
                std::cerr << "postSlot failed on resubmit for slot " << s << ": "
                          << nixlEnumStrings::statusStr(rc) << std::endl;
                cleanupSlots(agent, backend_engine, slots);
                return -1;
            }
            issued++;
        }
    }

    cleanupSlots(agent, backend_engine, slots);
    return 0;
}

static int
execTransfer(nixlAgent *agent,
             nixlBackendH *backend_engine,
             const std::vector<std::vector<xferBenchIOV>> &local_iovs,
             const std::vector<std::vector<xferBenchIOV>> &remote_iovs,
             const nixl_xfer_op_t op,
             const int num_iter,
             const int num_threads,
             xferBenchStats &stats,
             const std::atomic<int> *terminate_ptr = nullptr) {
    int ret = 0;
    stats.clear();

    xferBenchTimer total_timer;
#pragma omp parallel num_threads(num_threads)
    {
        xferBenchStats thread_stats;
        thread_stats.reserve(num_iter);
        const int tid = omp_get_thread_num();
        const auto &local_iov = local_iovs[tid];
        const auto &remote_iov = remote_iovs[tid];

        // Setup transfer parameters
        nixl_opt_args_t params;
        std::string target = xferBenchConfig::isStorageBackend() ? "initiator" : "target";
        if (!xferBenchConfig::isStorageBackend()) {
            params.notif = "0xBEEF";
        }

        int result = execTransferLoop(agent,
                                      backend_engine,
                                      op,
                                      target,
                                      params,
                                      num_iter,
                                      thread_stats,
                                      local_iov,
                                      remote_iov,
                                      terminate_ptr);

        if (result != 0) [[unlikely]] {
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
    int num_iter = xferBenchConfig::num_iter / xferBenchConfig::num_threads;
    int skip = xferBenchConfig::warmup_iter / xferBenchConfig::num_threads;
    xferBenchStats stats;
    int ret = 0;
    nixl_xfer_op_t xfer_op = XFERBENCH_OP_READ == xferBenchConfig::op_type ? NIXL_READ : NIXL_WRITE;

    if (!rt->checkKeepAlive()) { // also refreshes the lease internally.
        std::cerr << "nixlbench: keepalive failed before transfer — aborting" << std::endl;
        return std::variant<xferBenchStats, int>(-1);
    }

    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= xferBenchConfig::large_blk_iter_ftr;
        num_iter /= xferBenchConfig::large_blk_iter_ftr;
    }

    if (skip > 0) {
        ret = execTransfer(agent,
                           backend_engine,
                           local_iovs,
                           remote_iovs,
                           xfer_op,
                           skip,
                           xferBenchConfig::num_threads,
                           stats,
                           &terminate);
        if (ret < 0) {
            return std::variant<xferBenchStats, int>(ret);
        }
    }

    // Synchronize to ensure all processes have completed the warmup (iter and polling)
    synchronize();

    stats.clear();

    ret = execTransfer(agent,
                       backend_engine,
                       local_iovs,
                       remote_iovs,
                       xfer_op,
                       num_iter,
                       xferBenchConfig::num_threads,
                       stats,
                       &terminate);
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

    skip = xferBenchConfig::warmup_iter;
    num_iter = xferBenchConfig::num_iter;
    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= xferBenchConfig::large_blk_iter_ftr;
        num_iter /= xferBenchConfig::large_blk_iter_ftr;
    }
    total_iter = skip + num_iter;

    // Periodically check if all peers are still alive via etcd lease keys.
    // Fires at most once every liveness_check_interval to avoid
    // saturating etcd with get() calls during tight polling loops.
    using namespace std::chrono_literals;
    constexpr auto liveness_check_interval = 5s;
    auto last_liveness_check = std::chrono::steady_clock::time_point::min(); // never done before.
    auto checkLiveness = [&]() {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_liveness_check >= liveness_check_interval) {
            last_liveness_check = now;
            if (rt && !rt->areAllPeersAlive()) {
                std::cerr << "nixlbench: peer liveness check failed — aborting poll" << std::endl;
                terminate.store(1);
            }
        }
    };

    /* Ensure warmup is done*/
    do {
        status = agent->getNotifs(notifs);
        checkLiveness();
    } while (!signaled() && status == NIXL_SUCCESS && skip != int(notifs["initiator"].size()));
    synchronize();

    /* Polling for actual iterations*/
    do {
        status = agent->getNotifs(notifs);
        checkLiveness();
    } while (!signaled() && status == NIXL_SUCCESS &&
             total_iter != int(notifs["initiator"].size()));
    synchronize();
}

int
xferBenchNixlWorker::synchronizeStart() {
    if (IS_PAIRWISE_AND_SG()) {
        std::cout << "Waiting for all processes to start... (expecting " << rt->getSize()
                  << " total: " << xferBenchConfig::num_initiator_dev << " initiators and "
                  << xferBenchConfig::num_target_dev << " targets)" << std::endl;
    } else {
        std::cout << "Waiting for all processes to start... (expecting " << rt->getSize()
                  << " total)" << std::endl;
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
