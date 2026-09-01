/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_ALLOCATE_ONCE_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_ALLOCATE_ONCE_H

#include "benchmark/scenario.h"
#include "utils/utils.h"

#include <nixl_types.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace nixlbench {

/** @brief Offset selection policy owned by the allocate-once scenario. */
enum class offset_mode_t {
    SEQUENTIAL, ///< Walk and wrap each thread's disjoint block partition.
    RANDOM, ///< Sample unique block slots inside each thread partition.
};

/** @brief Fully resolved configuration owned by the allocate-once path. */
struct allocateOnceRequest {
    scenarioConfig common;
    std::vector<std::filesystem::path> files;
    size_t fileSize = 0;
    offset_mode_t offsetMode = offset_mode_t::RANDOM;
    uint64_t seed = 0;
    bool managedFiles = true;
    bool direct = false;
};

/** @brief Disjoint block range assigned to one worker thread within one file. */
struct threadFileRegion {
    size_t fileIndex = 0;
    uint64_t firstSlot = 0;
    uint64_t slotCount = 0;
};

/**
 * @brief Partition fixed files into disjoint per-thread block regions.
 * @param request Resolved allocate-once request
 * @param error Error text populated on failure
 * @return thread regions on success, otherwise std::nullopt
 */
std::optional<std::vector<threadFileRegion>>
allocateOnceThreadRegions(const allocateOnceRequest &request, std::string &error);

/** @brief Produces sequential or reproducibly randomized block offsets within one thread region. */
class offsetSequence {
public:
    /** @brief Construct one thread's reproducible offset sequence. */
    offsetSequence(threadFileRegion region,
                   size_t batch_size,
                   offset_mode_t offset_mode,
                   uint64_t seed);

    /** @brief Return the next batch of block slots. */
    std::vector<uint64_t>
    next();

private:
    threadFileRegion region_;
    size_t batchSize_;
    bool randomize_;
    uint64_t nextSequentialSlot_ = 0;
    std::mt19937_64 random_;
};

/** @brief Prepare and validate backing files before worker construction. */
bool
prepareAllocateOnceFiles(const allocateOnceRequest &request, std::ostream &err);

/** @brief Return whether plugin metadata supports the allocate-once scenario. */
bool
supportsAllocateOnce(const pluginMetadata &metadata);

/** @brief Resolve explicit or managed backing-file names. */
std::vector<std::filesystem::path>
allocateOnceFileNames(const fileOptions &file);

/** @brief Return bounded transfer-memory capacity for an allocate-once request. */
size_t
allocateOnceWorkingMemory(const allocateOnceRequest &request);

/** @brief Resolve zero to a nonzero random seed and preserve explicit nonzero seeds. */
uint64_t
resolveOffsetSeed(uint64_t configured_seed);

/** @brief Allocate-once CLI, lifecycle policy, and worker-strategy factory. */
class allocateOnceScenario final : public benchmarkScenario {
public:
    /** @brief Construct the allocate-once scenario definition. */
    allocateOnceScenario();
    ~allocateOnceScenario() override;

    /** @brief Prepare allocate-once backing files. */
    bool
    prepare(std::ostream &err) const override;

    /** @brief Create the allocate-once NIXL worker strategy. */
    std::unique_ptr<xferBenchWorker>
    createWorker(const std::vector<std::string> &devices) const override;

protected:
    void
    addScenarioOptions(CLI::App &command) override;

    int
    finalizeScenario(std::ostream &err) override;

    void
    printScenarioPlan(std::ostream &out) const override;

    void
    printDryRunPlan(std::ostream &out) const override;

    void
    configureLegacyWorker(legacyWorkerConfig &config) const override;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace nixlbench

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_ALLOCATE_ONCE_H
