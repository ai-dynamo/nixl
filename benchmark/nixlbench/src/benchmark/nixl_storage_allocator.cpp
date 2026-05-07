/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/nixl_storage_allocator.h"

#include "config.h"
#include "utils/neuron.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

namespace nixlbench {
namespace {

#define ROUND_UP(value, granularity) \
    ((((value) + (granularity) - 1) / (granularity)) * (granularity))

bool
allocateXferMemory(std::size_t page_size, std::size_t buffer_size, void **addr) {
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
        std::cerr << "Failed to allocate " << buffer_size
                  << " bytes of page-aligned DRAM memory" << std::endl;
        return false;
    }
    std::memset(*addr, 0, buffer_size);
    return true;
}

std::size_t
defaultPageSize() {
    const long page_size = sysconf(_SC_PAGESIZE);
    return page_size <= 0 ? 0 : static_cast<std::size_t>(page_size);
}

void
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

std::vector<std::string>
splitCsv(const std::string &value) {
    std::vector<std::string> result;
    std::string element;
    std::stringstream ss(value);
    while (std::getline(ss, element, ',')) {
        if (!element.empty()) {
            result.push_back(element);
        }
    }
    return result;
}

std::optional<fileRemoteIovStrategy::fileState>
openFileWithFlags(const std::string &op_type, const std::string &file_name, int flags) {
    std::uint64_t file_size = 0;
    if (XFERBENCH_OP_READ == op_type) {
        struct stat st;
        if (::stat(file_name.c_str(), &st) == 0) {
            file_size = static_cast<std::uint64_t>(st.st_size);
        }
    }

    int fd = open(file_name.c_str(), flags, 0744);
    if (fd < 0) {
        std::cerr << "Failed to open file: " << file_name << " with error: " << strerror(errno)
                  << std::endl;
        return std::nullopt;
    }

    return fileRemoteIovStrategy::fileState{fd, file_size, 0};
}

std::optional<xferBenchIOV>
createFileDesc(const std::string &op_type,
               std::size_t page_size,
               std::size_t buffer_size,
               fileRemoteIovStrategy::fileState &file_state) {
    const int fd = file_state.fd;
    const std::uint64_t start_offset = file_state.offset;
    const std::uint64_t end_offset = file_state.offset + buffer_size;
    auto ret = std::optional<xferBenchIOV>(std::in_place, file_state.offset, buffer_size, fd);

    file_state.offset = end_offset;

    if (XFERBENCH_OP_READ == op_type && end_offset <= file_state.file_size) {
        return ret;
    }

    void *buf = nullptr;
    if (!allocateXferMemory(page_size, buffer_size, &buf)) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of memory" << std::endl;
        return std::nullopt;
    }

    std::memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);

    std::size_t remaining = buffer_size;
    std::size_t offset = start_offset;
    char *write_ptr = static_cast<char *>(buf);
    while (remaining > 0) {
        ssize_t rc = pwrite(fd, write_ptr, remaining, offset);
        if (rc < 0) {
            std::cerr << "Failed to write to file: " << fd << " with error: " << strerror(errno)
                      << std::endl;
            free(buf);
            return std::nullopt;
        }

        remaining -= static_cast<std::size_t>(rc);
        offset += static_cast<std::size_t>(rc);
        write_ptr += rc;
    }

    free(buf);

    if (end_offset > file_state.file_size) {
        file_state.file_size = end_offset;
    }

    return ret;
}

std::uint64_t
timestampMicros() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<std::uint64_t>(tv.tv_sec) * 1000000ULL + tv.tv_usec;
}

#if HAVE_CUDA
std::optional<xferBenchIOV>
getVramDescCuda(int devid, std::size_t buffer_size, uint8_t memset_value) {
    void *addr = nullptr;
    cudaError_t alloc_result = cudaMalloc(&addr, buffer_size);
    if (alloc_result != cudaSuccess) {
        std::cerr << "Failed to allocate CUDA buffer: " << cudaGetErrorString(alloc_result)
                  << std::endl;
        return std::nullopt;
    }
    cudaError_t memset_result = cudaMemset(addr, memset_value, buffer_size);
    if (memset_result != cudaSuccess) {
        std::cerr << "Failed to set device memory: " << cudaGetErrorString(memset_result)
                  << std::endl;
        cudaFree(addr);
        return std::nullopt;
    }

    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, devid);
}

std::optional<xferBenchIOV>
getVramDescCudaVmm(int devid, std::size_t buffer_size, uint8_t memset_value) {
#if HAVE_CUDA_FABRIC
    CUdeviceptr addr = 0;
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access = {};

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.location.id = devid;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    std::size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
        CUDA_SUCCESS) {
        std::cerr << "Failed to get VMM allocation granularity" << std::endl;
        return std::nullopt;
    }

    std::size_t padded_size = ROUND_UP(buffer_size, granularity);
    CUmemGenericAllocationHandle handle;
    if (cuMemCreate(&handle, padded_size, &prop, 0) != CUDA_SUCCESS) {
        std::cerr << "Failed to create VMM allocation" << std::endl;
        return std::nullopt;
    }
    if (cuMemAddressReserve(&addr, padded_size, granularity, 0, 0) != CUDA_SUCCESS) {
        std::cerr << "Failed to reserve VMM address" << std::endl;
        cuMemRelease(handle);
        return std::nullopt;
    }
    if (cuMemMap(addr, padded_size, 0, handle, 0) != CUDA_SUCCESS) {
        std::cerr << "Failed to map VMM allocation" << std::endl;
        cuMemAddressFree(addr, padded_size);
        cuMemRelease(handle);
        return std::nullopt;
    }

    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = devid;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(addr, buffer_size, &access, 1) != CUDA_SUCCESS ||
        cuMemsetD8(addr, memset_value, buffer_size) != CUDA_SUCCESS) {
        std::cerr << "Failed to initialize VMM allocation" << std::endl;
        cuMemUnmap(addr, padded_size);
        cuMemAddressFree(addr, padded_size);
        cuMemRelease(handle);
        return std::nullopt;
    }

    return std::optional<xferBenchIOV>(
        std::in_place, (uintptr_t)addr, buffer_size, devid, padded_size, handle);
#else
    std::cerr << "CUDA_FABRIC is not supported" << std::endl;
    return std::nullopt;
#endif
}

std::optional<xferBenchIOV>
getVramDescNeuron(int devid, std::size_t buffer_size, uint8_t memset_value) {
    void *addr = nullptr;
    if (neuronMalloc(&addr, buffer_size, devid) != 0) {
        std::cerr << "Failed to allocate Neuron tensor" << std::endl;
        return std::nullopt;
    }
    if (neuronMemset(addr, memset_value, buffer_size) != 0) {
        std::cerr << "Failed to set Neuron tensor memory" << std::endl;
        neuronFree(addr);
        return std::nullopt;
    }

    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, devid);
}
#endif

} // namespace

dramLocalIovStrategy::dramLocalIovStrategy(std::size_t page_size)
    : page_size_(page_size == 0 ? defaultPageSize() : page_size) {}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
dramLocalIovStrategy::create(int num_threads, std::size_t buffer_size) {
    if (num_threads <= 0 || buffer_size == 0) {
        return EXIT_FAILURE;
    }

    std::vector<std::vector<xferBenchIOV>> iov_lists;
    iov_lists.reserve(static_cast<std::size_t>(num_threads));
    for (int list_idx = 0; list_idx < num_threads; ++list_idx) {
        void *addr = nullptr;
        if (!allocateXferMemory(page_size_, buffer_size, &addr)) {
            cleanup(iov_lists);
            return EXIT_FAILURE;
        }
        iov_lists.push_back({xferBenchIOV((uintptr_t)addr, buffer_size, 0)});
    }

    return iov_lists;
}

void
dramLocalIovStrategy::cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    for (auto &iov_list : iov_lists) {
        for (auto &iov : iov_list) {
            free(reinterpret_cast<void *>(iov.addr));
            iov.addr = 0;
        }
    }
    iov_lists.clear();
}

nixl_mem_t
dramLocalIovStrategy::segmentType() const {
    return DRAM_SEG;
}

vramLocalIovStrategy::vramLocalIovStrategy(bool enable_vmm)
    : enable_vmm_(enable_vmm) {}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
vramLocalIovStrategy::create(int num_threads, std::size_t buffer_size) {
#if HAVE_CUDA
    if (num_threads <= 0 || buffer_size == 0) {
        return EXIT_FAILURE;
    }

    std::vector<std::vector<xferBenchIOV>> iov_lists;
    iov_lists.reserve(static_cast<std::size_t>(num_threads));
    const uint8_t memset_value = XFERBENCH_INITIATOR_BUFFER_ELEMENT;
    for (int list_idx = 0; list_idx < num_threads; ++list_idx) {
        if (neuronCoreCount() == 0) {
            cudaError_t set_device_result = cudaSetDevice(list_idx);
            if (set_device_result != cudaSuccess) {
                std::cerr << "Failed to set CUDA device: "
                          << cudaGetErrorString(set_device_result) << std::endl;
                cleanup(iov_lists);
                return EXIT_FAILURE;
            }
        }

        auto desc = neuronCoreCount() > 0 ?
            getVramDescNeuron(list_idx, buffer_size, memset_value) :
            (enable_vmm_ ? getVramDescCudaVmm(list_idx, buffer_size, memset_value) :
                           getVramDescCuda(list_idx, buffer_size, memset_value));
        if (!desc) {
            cleanup(iov_lists);
            return EXIT_FAILURE;
        }
        iov_lists.push_back({desc.value()});
    }

    return iov_lists;
#else
    (void)num_threads;
    (void)buffer_size;
    std::cerr << "VRAM segment type not supported without CUDA" << std::endl;
    return EXIT_FAILURE;
#endif
}

void
vramLocalIovStrategy::cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
#if HAVE_CUDA
    for (auto &iov_list : iov_lists) {
        for (auto &iov : iov_list) {
            if (neuronCoreCount() > 0) {
                neuronFree(reinterpret_cast<void *>(iov.addr));
                continue;
            }

            cudaSetDevice(iov.devId);
            if (enable_vmm_) {
#if HAVE_CUDA_FABRIC
                cuMemUnmap(iov.addr, iov.padded_size);
                cuMemRelease(iov.handle);
                cuMemAddressFree(iov.addr, iov.padded_size);
#endif
            } else {
                cudaFree(reinterpret_cast<void *>(iov.addr));
            }
            iov.addr = 0;
        }
    }
#endif
    iov_lists.clear();
}

nixl_mem_t
vramLocalIovStrategy::segmentType() const {
    return VRAM_SEG;
}

std::unique_ptr<localIovStrategy>
makeLocalIovStrategy(const std::string &segment_type, std::size_t page_size, bool enable_vmm) {
    if (segment_type == XFERBENCH_SEG_TYPE_DRAM) {
        return std::make_unique<dramLocalIovStrategy>(page_size);
    }
    if (segment_type == XFERBENCH_SEG_TYPE_VRAM) {
        return std::make_unique<vramLocalIovStrategy>(enable_vmm);
    }
    return nullptr;
}

fileRemoteIovStrategy::fileRemoteIovStrategy(storageFileConfig config)
    : config_(std::move(config)) {
    if (config_.page_size == 0) {
        config_.page_size = defaultPageSize();
    }
}

std::variant<std::vector<fileRemoteIovStrategy::fileState>, int>
fileRemoteIovStrategy::openFiles() {
    if (config_.num_files <= 0) {
        std::cerr << "Invalid storage file count" << std::endl;
        return EXIT_FAILURE;
    }

    int flags = O_RDWR | O_CREAT | O_LARGEFILE;
    if (config_.enable_direct) {
        flags |= O_DIRECT;
    }

    std::vector<std::string> filenames = splitCsv(config_.filenames);
    if (!filenames.empty() && filenames.size() != static_cast<std::size_t>(config_.num_files)) {
        std::cerr << "Error: Number of filenames (" << filenames.size()
                  << ") doesn't match num_files (" << config_.num_files << ")" << std::endl;
        return EXIT_FAILURE;
    }

    if (filenames.empty()) {
        const std::string file_path = config_.filepath.empty() ?
            std::filesystem::current_path().string() :
            config_.filepath;
        const std::string file_name_prefix = "/nixlbench_test_file_";
        for (int i = 0; i < config_.num_files; ++i) {
            filenames.push_back(file_path + file_name_prefix + config_.backend_name + "_" +
                                std::to_string(i));
        }
    }

    std::vector<fileState> fds;
    fds.reserve(filenames.size());
    for (const auto &file_name : filenames) {
        auto fstate = openFileWithFlags(config_.op_type, file_name, flags);
        if (!fstate) {
            for (auto &fd : fds) {
                close(fd.fd);
            }
            return EXIT_FAILURE;
        }
        fds.push_back(fstate.value());
    }

    return fds;
}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
fileRemoteIovStrategy::create(int num_threads, std::size_t buffer_size) {
    if (num_threads <= 0 || buffer_size == 0) {
        return EXIT_FAILURE;
    }

    auto fds_result = openFiles();
    // an int result is an error code, return the failure
    if (std::holds_alternative<int>(fds_result)) {
        return std::get<int>(fds_result);
    }
    files_ = std::move(std::get<std::vector<fileState>>(fds_result));

    std::vector<std::vector<xferBenchIOV>> remote_iovs;
    remote_iovs.reserve(static_cast<std::size_t>(num_threads));
    std::size_t file_idx = 0;
    for (int list_idx = 0; list_idx < num_threads; ++list_idx) {
        auto basic_desc = createFileDesc(config_.op_type,
                                         config_.page_size,
                                         buffer_size,
                                         files_[file_idx]);
        if (!basic_desc) {
            cleanup(remote_iovs);
            return EXIT_FAILURE;
        }
        remote_iovs.push_back({basic_desc.value()});
        file_idx = (file_idx + 1) % files_.size();
    }

    return remote_iovs;
}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
fileRemoteIovStrategy::createTransferIovs(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                                          std::size_t block_size) const {
    if (block_size == 0 || files_.empty()) {
        return EXIT_FAILURE;
    }

    std::vector<std::vector<xferBenchIOV>> remote_iovs;
    remote_iovs.reserve(local_iovs.size());
    std::size_t fd_idx = 0;
    std::uint64_t file_offset = 0;
    for (const auto &iov_list : local_iovs) {
        std::vector<xferBenchIOV> remote_iov_list;
        remote_iov_list.reserve(iov_list.size());
        for (const auto &iov : iov_list) {
            xferBenchIOV remote_iov(iov);
            remote_iov.addr = file_offset;
            remote_iov.len = block_size;
            remote_iov.devId = files_[fd_idx].fd;
            remote_iov_list.push_back(remote_iov);
            fd_idx++;
            if (fd_idx >= files_.size()) {
                file_offset += block_size;
                fd_idx = 0;
            }
        }
        remote_iovs.push_back(remote_iov_list);
        file_offset += block_size;
    }

    return remote_iovs;
}

void
fileRemoteIovStrategy::cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    for (auto &file : files_) {
        if (file.fd >= 0) {
            close(file.fd);
            file.fd = -1;
        }
    }
    files_.clear();
    iov_lists.clear();
}

nixl_mem_t
fileRemoteIovStrategy::segmentType() const {
    return FILE_SEG;
}

objectRemoteIovStrategy::objectRemoteIovStrategy(objectStorageConfig config)
    : config_(std::move(config)) {}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
objectRemoteIovStrategy::create(int num_threads, std::size_t buffer_size) {
    if (num_threads <= 0 || config_.num_devices <= 0 || buffer_size == 0) {
        return EXIT_FAILURE;
    }

    std::vector<std::vector<xferBenchIOV>> remote_iovs;
    remote_iovs.reserve(static_cast<std::size_t>(num_threads));
    const std::uint64_t timestamp = timestampMicros();
    for (int list_idx = 0; list_idx < num_threads; ++list_idx) {
        std::vector<xferBenchIOV> iov_list;
        iov_list.reserve(static_cast<std::size_t>(config_.num_devices));
        for (int device_idx = 0; device_idx < config_.num_devices; ++device_idx) {
            std::string name = config_.name_prefix + std::to_string(list_idx) + "_" +
                std::to_string(device_idx) + "_" + std::to_string(timestamp);
            iov_list.emplace_back(0, buffer_size, device_idx, name);
        }
        remote_iovs.push_back(std::move(iov_list));
    }

    return remote_iovs;
}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
objectRemoteIovStrategy::createTransferIovs(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                                            std::size_t block_size) const {
    std::vector<std::vector<xferBenchIOV>> remote_iovs;
    remote_iovs.reserve(local_iovs.size());
    for (const auto &iov_list : local_iovs) {
        std::vector<xferBenchIOV> remote_iov_list;
        remote_iov_list.reserve(iov_list.size());
        for (const auto &iov : iov_list) {
            remote_iov_list.emplace_back(0, block_size, iov.devId, iov.metaInfo);
        }
        remote_iovs.push_back(std::move(remote_iov_list));
    }
    return remote_iovs;
}

void
objectRemoteIovStrategy::cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    iov_lists.clear();
}

nixl_mem_t
objectRemoteIovStrategy::segmentType() const {
    return OBJ_SEG;
}

blockRemoteIovStrategy::blockRemoteIovStrategy(std::vector<blockStorageDevice> devices)
    : devices_(std::move(devices)) {}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
blockRemoteIovStrategy::create(int num_threads, std::size_t buffer_size) {
    if (num_threads <= 0 || devices_.empty() || buffer_size == 0) {
        return EXIT_FAILURE;
    }

    std::vector<std::vector<xferBenchIOV>> remote_iovs;
    remote_iovs.reserve(static_cast<std::size_t>(num_threads));
    for (int list_idx = 0; list_idx < num_threads; ++list_idx) {
        std::vector<xferBenchIOV> iov_list;
        iov_list.reserve(devices_.size());
        for (const auto &device : devices_) {
            iov_list.emplace_back(device.offset, buffer_size, device.device_id);
        }
        remote_iovs.push_back(std::move(iov_list));
    }

    return remote_iovs;
}

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
blockRemoteIovStrategy::createTransferIovs(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                                           std::size_t block_size) const {
    std::vector<std::vector<xferBenchIOV>> remote_iovs;
    remote_iovs.reserve(local_iovs.size());
    for (const auto &iov_list : local_iovs) {
        std::vector<xferBenchIOV> remote_iov_list;
        remote_iov_list.reserve(iov_list.size());
        for (std::size_t i = 0; i < iov_list.size(); ++i) {
            const auto &device = devices_[i % devices_.size()];
            remote_iov_list.emplace_back(device.offset, block_size, device.device_id);
        }
        remote_iovs.push_back(std::move(remote_iov_list));
    }
    return remote_iovs;
}

void
blockRemoteIovStrategy::cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    iov_lists.clear();
}

nixl_mem_t
blockRemoteIovStrategy::segmentType() const {
    return BLK_SEG;
}

nixlStorageAllocator::nixlStorageAllocator(nixlAgent &agent,
                                           nixlBackendH *backend,
                                           int num_threads,
                                           std::size_t total_buffer_size,
                                           bool align_for_direct_io,
                                           localIovStrategy &local_strategy,
                                           remoteIovStrategy *remote_strategy)
    : agent_(agent),
      backend_(backend),
      num_threads_(num_threads),
      total_buffer_size_(total_buffer_size),
      align_for_direct_io_(align_for_direct_io),
      page_size_(defaultPageSize()),
      local_strategy_(local_strategy),
      remote_strategy_(remote_strategy) {}

allocationResult
nixlStorageAllocator::allocate() {
    if (num_threads_ <= 0 || total_buffer_size_ == 0 || page_size_ == 0) {
        return EXIT_FAILURE;
    }

    benchmarkAllocation allocation;
    const std::size_t buffer_size = perThreadBufferSize();

    if (remote_strategy_ != nullptr) {
        auto remote_result = remote_strategy_->create(num_threads_, buffer_size);
        if (std::holds_alternative<int>(remote_result)) {
            return std::get<int>(remote_result);
        }
        allocation.remote_iovs = std::move(std::get<std::vector<std::vector<xferBenchIOV>>>(
            remote_result));
        if (!registerIovLists(allocation.remote_iovs, remote_strategy_->segmentType())) {
            remote_strategy_->cleanup(allocation.remote_iovs);
            return EXIT_FAILURE;
        }
    }

    auto local_result = local_strategy_.create(num_threads_, buffer_size);
    if (std::holds_alternative<int>(local_result)) {
        if (remote_strategy_ != nullptr) {
            deregisterIovLists(allocation.remote_iovs, remote_strategy_->segmentType());
            remote_strategy_->cleanup(allocation.remote_iovs);
        }
        return std::get<int>(local_result);
    }
    allocation.local_iovs = std::move(std::get<std::vector<std::vector<xferBenchIOV>>>(
        local_result));

    if (!allocation.remote_iovs.empty()) {
        for (std::size_t i = 0; i < allocation.local_iovs.size(); ++i) {
            if (!allocation.remote_iovs[i].empty() && !allocation.local_iovs[i].empty()) {
                allocation.local_iovs[i][0].metaInfo = allocation.remote_iovs[i][0].metaInfo;
            }
        }
    }

    if (!registerIovLists(allocation.local_iovs, local_strategy_.segmentType())) {
        local_strategy_.cleanup(allocation.local_iovs);
        if (remote_strategy_ != nullptr) {
            deregisterIovLists(allocation.remote_iovs, remote_strategy_->segmentType());
            remote_strategy_->cleanup(allocation.remote_iovs);
        }
        return EXIT_FAILURE;
    }

    return allocation;
}

void
nixlStorageAllocator::deallocate(benchmarkAllocation &allocation) {
    deregisterIovLists(allocation.local_iovs, local_strategy_.segmentType());
    local_strategy_.cleanup(allocation.local_iovs);

    if (remote_strategy_ != nullptr) {
        deregisterIovLists(allocation.remote_iovs, remote_strategy_->segmentType());
        remote_strategy_->cleanup(allocation.remote_iovs);
    }
}

bool
nixlStorageAllocator::registerIovLists(const std::vector<std::vector<xferBenchIOV>> &iov_lists,
                                       nixl_mem_t segment_type) {
    nixl_opt_args_t opt_args;
    if (backend_ != nullptr) {
        opt_args.backends.push_back(backend_);
    }

    for (std::size_t list_idx = 0; list_idx < iov_lists.size(); ++list_idx) {
        nixl_reg_dlist_t desc_list(segment_type);
        iovListToNixlRegDlist(iov_lists[list_idx], desc_list);
        nixl_status_t status = agent_.registerMem(desc_list, &opt_args);
        if (status != NIXL_SUCCESS) {
            std::cerr << "NIXL: registerMem failed (Error code: " << status << ")" << std::endl;
            for (std::size_t registered_idx = 0; registered_idx < list_idx; ++registered_idx) {
                nixl_reg_dlist_t registered_desc_list(segment_type);
                iovListToNixlRegDlist(iov_lists[registered_idx], registered_desc_list);
                agent_.deregisterMem(registered_desc_list, &opt_args);
            }
            return false;
        }
    }

    return true;
}

void
nixlStorageAllocator::deregisterIovLists(const std::vector<std::vector<xferBenchIOV>> &iov_lists,
                                         nixl_mem_t segment_type) {
    nixl_opt_args_t opt_args;
    if (backend_ != nullptr) {
        opt_args.backends.push_back(backend_);
    }

    for (const auto &iov_list : iov_lists) {
        nixl_reg_dlist_t desc_list(segment_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        nixl_status_t status = agent_.deregisterMem(desc_list, &opt_args);
        if (status != NIXL_SUCCESS) {
            std::cerr << "NIXL: deregisterMem failed (Error code: " << status << ")"
                      << std::endl;
        }
    }
}

std::size_t
nixlStorageAllocator::perThreadBufferSize() const {
    std::size_t buffer_size = total_buffer_size_ / static_cast<std::size_t>(num_threads_);
    if (align_for_direct_io_) {
        buffer_size = ROUND_UP(buffer_size, page_size_);
    }
    return buffer_size;
}

} // namespace nixlbench
