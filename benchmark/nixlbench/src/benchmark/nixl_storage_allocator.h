/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_NIXL_STORAGE_ALLOCATOR_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_NIXL_STORAGE_ALLOCATOR_H

#include "benchmark/benchmark_run_components.h"

#include <nixl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace nixlbench {

struct storageFileConfig {
    std::string backend_name;
    std::string op_type = XFERBENCH_OP_WRITE;
    std::string filepath;
    std::string filenames;
    int num_files = 1;
    bool enable_direct = false;
    std::size_t page_size = 0;
};

struct objectStorageConfig {
    std::string name_prefix = "nixlbench_obj";
    int num_devices = 1;
};

struct blockStorageDevice {
    int device_id = 0;
    std::size_t offset = 0;
};

class localIovStrategy {
public:
    virtual
    ~localIovStrategy() = default;

    virtual std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(int num_threads, std::size_t buffer_size) = 0;

    virtual void
    cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) = 0;

    virtual nixl_mem_t
    segmentType() const = 0;
};

class dramLocalIovStrategy : public localIovStrategy {
public:
    explicit
    dramLocalIovStrategy(std::size_t page_size = 0);

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(int num_threads, std::size_t buffer_size) override;

    void
    cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) override;

    nixl_mem_t
    segmentType() const override;

private:
    std::size_t page_size_;
};

class vramLocalIovStrategy : public localIovStrategy {
public:
    explicit
    vramLocalIovStrategy(bool enable_vmm = false);

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(int num_threads, std::size_t buffer_size) override;

    void
    cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) override;

    nixl_mem_t
    segmentType() const override;
private:
    bool enable_vmm_;
};

std::unique_ptr<localIovStrategy>
makeLocalIovStrategy(const std::string &segment_type,
                     std::size_t page_size = 0,
                     bool enable_vmm = false);

class remoteIovStrategy {
public:
    virtual
    ~remoteIovStrategy() = default;

    virtual std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(int num_threads, std::size_t buffer_size) = 0;

    virtual std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    createTransferIovs(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                       std::size_t block_size) const = 0;

    virtual void
    cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) = 0;

    virtual nixl_mem_t
    segmentType() const = 0;
};

class fileRemoteIovStrategy : public remoteIovStrategy {
public:
    struct fileState {
        int fd = -1;
        std::uint64_t file_size = 0;
        std::uint64_t offset = 0;
    };

    explicit
    fileRemoteIovStrategy(storageFileConfig config);

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(int num_threads, std::size_t buffer_size) override;

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    createTransferIovs(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                       std::size_t block_size) const override;

    void
    cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) override;

    nixl_mem_t
    segmentType() const override;

private:
    std::variant<std::vector<fileState>, int>
    openFiles();

    storageFileConfig config_;
    std::vector<fileState> files_;
};

class objectRemoteIovStrategy : public remoteIovStrategy {
public:
    explicit
    objectRemoteIovStrategy(objectStorageConfig config = {});

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(int num_threads, std::size_t buffer_size) override;

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    createTransferIovs(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                       std::size_t block_size) const override;

    void
    cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) override;

    nixl_mem_t
    segmentType() const override;

private:
    objectStorageConfig config_;
};

class blockRemoteIovStrategy : public remoteIovStrategy {
public:
    explicit
    blockRemoteIovStrategy(std::vector<blockStorageDevice> devices);

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(int num_threads, std::size_t buffer_size) override;

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    createTransferIovs(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                       std::size_t block_size) const override;

    void
    cleanup(std::vector<std::vector<xferBenchIOV>> &iov_lists) override;

    nixl_mem_t
    segmentType() const override;

private:
    std::vector<blockStorageDevice> devices_;
};

class nixlStorageAllocator : public benchmarkMemoryAllocator {
public:
    nixlStorageAllocator(nixlAgent &agent,
                         nixlBackendH *backend,
                         int num_threads,
                         std::size_t total_buffer_size,
                         bool align_for_direct_io,
                         localIovStrategy &local_strategy,
                         remoteIovStrategy *remote_strategy = nullptr);

    allocationResult
    allocate() override;

    void
    deallocate(benchmarkAllocation &allocation) override;

private:
    bool
    registerIovLists(const std::vector<std::vector<xferBenchIOV>> &iov_lists,
                     nixl_mem_t segment_type);

    void
    deregisterIovLists(const std::vector<std::vector<xferBenchIOV>> &iov_lists,
                       nixl_mem_t segment_type);

    std::size_t
    perThreadBufferSize() const;

    nixlAgent &agent_;
    nixlBackendH *backend_;
    int num_threads_;
    std::size_t total_buffer_size_;
    bool align_for_direct_io_;
    std::size_t page_size_;
    localIovStrategy &local_strategy_;
    remoteIovStrategy *remote_strategy_;
};

} // namespace nixlbench

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_NIXL_STORAGE_ALLOCATOR_H
