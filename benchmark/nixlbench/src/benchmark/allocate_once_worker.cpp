/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/allocate_once_worker.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace {

constexpr uint64_t per_thread_seed_increment = 0x9e3779b97f4a7c15ULL;

class offsetIterationLifecycle final : public xferBenchNixlIterationLifecycle {
public:
    offsetIterationLifecycle(nixlbench::offsetSequence &offsets,
                             size_t block_size,
                             size_t thread,
                             std::vector<std::vector<xferBenchIOV>> &last_remote_iovs)
        : offsets_(offsets),
          blockSize_(block_size),
          thread_(thread),
          lastRemoteIovs_(last_remote_iovs) {}

    nixl_status_t
    prepare(std::vector<xferBenchIOV> &local_iovs,
            std::vector<xferBenchIOV> &remote_iovs) override {
        (void)local_iovs;
        const auto slots = offsets_.next();
        for (size_t index = 0; index < remote_iovs.size(); ++index) {
            remote_iovs[index].addr = slots[index] * blockSize_;
        }
        return NIXL_SUCCESS;
    }

    nixl_status_t
    complete(const std::vector<xferBenchIOV> &local_iovs,
             const std::vector<xferBenchIOV> &remote_iovs) override {
        (void)local_iovs;
        lastRemoteIovs_[thread_] = remote_iovs;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    release() override {
        return NIXL_SUCCESS;
    }

private:
    nixlbench::offsetSequence &offsets_;
    size_t blockSize_;
    size_t thread_;
    std::vector<std::vector<xferBenchIOV>> &lastRemoteIovs_;
};

} // namespace

xferBenchNixlAllocateOnceWorker::xferBenchNixlAllocateOnceWorker(
    const std::vector<std::string> &devices,
    nixlbench::allocateOnceRequest request)
    : xferBenchNixlWorker(devices),
      request_(std::move(request)) {}

std::vector<std::vector<xferBenchIOV>>
xferBenchNixlAllocateOnceWorker::allocateMemory(int num_threads) {
    if (num_threads != request_.common.threads) {
        std::cerr << "Scenario thread configuration does not match the worker" << std::endl;
        return {};
    }

    std::vector<xferBenchIOV> file_iovs;
    file_iovs.reserve(request_.files.size());
    std::set<std::pair<dev_t, ino_t>> file_identities;
    for (const auto &path : request_.files) {
        const int access_flags = (request_.common.operation == NIXL_WRITE ? O_RDWR : O_RDONLY) |
            O_LARGEFILE | (request_.direct ? O_DIRECT : 0);
        int flags = access_flags;
        if (request_.managedFiles) {
            flags |= O_NOFOLLOW;
        }
        int fd = open(path.c_str(), flags);
        if (fd < 0) {
            std::cerr << "Failed to open scenario file " << path << ": " << strerror(errno)
                      << std::endl;
            return {};
        }
        if (request_.managedFiles) {
            // Pin the verified path, then reopen the same inode without O_NOFOLLOW because some
            // FILE_SEG backends reject that flag in registered descriptors.
            const std::string pinned_path = "/proc/self/fd/" + std::to_string(fd);
            const int pinned_fd = open(pinned_path.c_str(), access_flags);
            close(fd);
            fd = pinned_fd;
            if (fd < 0) {
                std::cerr << "Failed to reopen pinned scenario file " << path << ": "
                          << strerror(errno) << std::endl;
                return {};
            }
        }
        struct stat info{};
        if (fstat(fd, &info) != 0 || (request_.managedFiles && !S_ISREG(info.st_mode))) {
            std::cerr << "Scenario file is no longer a valid regular file: " << path << std::endl;
            close(fd);
            return {};
        }
        const bool invalid_size = request_.managedFiles ?
            static_cast<uint64_t>(info.st_size) != request_.fileSize :
            info.st_size < 0 || static_cast<uint64_t>(info.st_size) < request_.fileSize;
        if (invalid_size) {
            std::cerr << "Scenario file size changed before registration: " << path << std::endl;
            close(fd);
            return {};
        }
        if (!file_identities.emplace(info.st_dev, info.st_ino).second) {
            std::cerr << "Scenario files must still refer to distinct files: " << path << std::endl;
            close(fd);
            return {};
        }
        retainRemoteFile(fd, request_.fileSize);
        file_iovs.emplace_back(0, request_.fileSize, fd);
    }

    if (!registerRemoteIovs(FILE_SEG, std::move(file_iovs))) {
        return {};
    }

    const size_t buffer_size = request_.common.blockSize * request_.common.batchSize;
    const nixl_mem_t local_memory_type = localMemoryType();
    std::vector<std::vector<xferBenchIOV>> iov_lists;
    iov_lists.reserve(static_cast<size_t>(request_.common.threads));
    for (int thread = 0; thread < request_.common.threads; ++thread) {
        auto local_iov = allocateLocalIov(buffer_size, 0);
        if (!local_iov) {
            std::cerr << "Failed to allocate scenario transfer buffer for thread " << thread
                      << std::endl;
            return {};
        }
        if (request_.common.operation == NIXL_WRITE) {
            initializeLocalIov(*local_iov, XFERBENCH_INITIATOR_BUFFER_ELEMENT);
        }
        std::vector<xferBenchIOV> thread_iovs{*local_iov};
        if (!registerLocalIovs(thread_iovs)) {
            return {};
        }
        if (local_memory_type == DRAM_SEG && request_.common.checkConsistency &&
            request_.common.operation == NIXL_READ) {
            memset(reinterpret_cast<void *>(local_iov->addr),
                   XFERBENCH_INITIATOR_BUFFER_ELEMENT,
                   local_iov->len);
        }
        iov_lists.push_back(std::move(thread_iovs));
    }

    std::string region_error;
    auto regions = nixlbench::allocateOnceThreadRegions(request_, region_error);
    if (!regions) {
        std::cerr << "Failed to partition scenario files: " << region_error << std::endl;
        return {};
    }
    offsetSequences_.clear();
    offsetSequences_.reserve(regions->size());
    for (size_t thread = 0; thread < regions->size(); ++thread) {
        const uint64_t thread_seed =
            request_.seed + per_thread_seed_increment * static_cast<uint64_t>(thread + 1);
        offsetSequences_.emplace_back(
            (*regions)[thread], request_.common.batchSize, request_.offsetMode, thread_seed);
    }
    return iov_lists;
}

std::vector<std::vector<xferBenchIOV>>
xferBenchNixlAllocateOnceWorker::exchangeIOV(
    const std::vector<std::vector<xferBenchIOV>> &local_iovs,
    size_t block_size) {
    (void)block_size;
    std::string region_error;
    auto regions = nixlbench::allocateOnceThreadRegions(request_, region_error);
    if (!regions) {
        std::cerr << "Failed to partition scenario files: " << region_error << std::endl;
        return {};
    }

    std::vector<std::vector<xferBenchIOV>> result;
    result.reserve(local_iovs.size());
    for (size_t thread = 0; thread < local_iovs.size(); ++thread) {
        std::vector<xferBenchIOV> remote_iovs;
        remote_iovs.reserve(local_iovs[thread].size());
        const auto fd = remoteFileDescriptor((*regions)[thread].fileIndex);
        if (!fd) {
            std::cerr << "Scenario file descriptor is unavailable for thread " << thread
                      << std::endl;
            return {};
        }
        for (const auto &local_iov : local_iovs[thread]) {
            remote_iovs.emplace_back(0, local_iov.len, *fd);
        }
        result.push_back(std::move(remote_iovs));
    }
    return result;
}

std::variant<xferBenchStats, int>
xferBenchNixlAllocateOnceWorker::transfer(
    size_t block_size,
    const std::vector<std::vector<xferBenchIOV>> &local_iovs,
    const std::vector<std::vector<xferBenchIOV>> &remote_iovs) {
    lastRemoteIovs_ = remote_iovs;
    iteration_lifecycle_factory_t lifecycle_factory = [this, block_size](size_t thread,
                                                                         size_t slot) {
        (void)slot;
        return std::make_unique<offsetIterationLifecycle>(
            offsetSequences_[thread], block_size, thread, lastRemoteIovs_);
    };
    return transferWithLifecycle(block_size, local_iovs, remote_iovs, lifecycle_factory);
}

bool
xferBenchNixlAllocateOnceWorker::validateTransfer(
    bool is_initiator,
    std::vector<std::vector<xferBenchIOV>> &local_iovs,
    std::vector<std::vector<xferBenchIOV>> &remote_iovs) {
    (void)is_initiator;
    (void)remote_iovs;
    if (!xferBenchConfig::check_consistency) {
        return true;
    }
    auto &iovs = request_.common.operation == NIXL_READ ? local_iovs : lastRemoteIovs_;
    return xferBenchUtils::checkConsistency(iovs);
}
