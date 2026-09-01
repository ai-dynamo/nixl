/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_SCENARIO_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_SCENARIO_H

#include "utils/cli_common.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace CLI {
class App;
}

class xferBenchWorker;

namespace nixlbench {

/** @brief Validated common configuration consumed by scenario worker strategies. */
struct scenarioConfig {
    std::string pluginName;
    nixl_b_params_t pluginParameters;
    size_t blockSize = 0;
    size_t batchSize = 1;
    int threads = 1;
    int iterations = 1000;
    int warmupIterations = 10;
    nixl_xfer_op_t operation = NIXL_WRITE;
    nixl_mem_t initiatorMemory = DRAM_SEG;
    bool checkConsistency = false;
    bool dryRun = false;
};

/** @brief Typed input to the legacy xferBenchConfig compatibility adapter. */
struct legacyWorkerConfig {
    scenarioConfig common;
    size_t workingMemory = 0;
    nixl_mem_t targetMemory = DRAM_SEG;
    bool recreateTransferRequest = false;
    std::vector<std::string> fileNames;
    bool storageDirect = false;
};

/** @brief Scenario plugin compatibility predicate. */
using scenario_plugin_filter_t = std::function<bool(const pluginMetadata &)>;

/**
 * @brief Translate typed scenario configuration into the legacy gflags bridge.
 * @param config Typed worker configuration
 * @param program_name Program name used as argv[0]
 * @return complete legacy argument vector
 */
std::vector<std::string>
legacyWorkerArguments(const legacyWorkerConfig &config, const std::string &program_name);

/**
 * @brief Extension point for a complete benchmark path.
 *
 * The framework owns common options, plugin discovery and selection, file options, common
 * validation, common plan output, dry-run behavior, and legacy translation. A scenario supplies
 * only its distinct options, validation, plan details, resource preparation, and worker strategy.
 */
class benchmarkScenario {
public:
    /**
     * @brief Construct a scenario definition.
     * @param name CLI subcommand name
     * @param description CLI subcommand description
     * @param plugin_filter Plugin compatibility predicate
     * @param has_file_options Whether the scenario uses shared file resource options
     */
    benchmarkScenario(std::string name,
                      std::string description,
                      scenario_plugin_filter_t plugin_filter,
                      bool has_file_options);
    virtual ~benchmarkScenario();

    /** @brief Register this scenario with the shared command hierarchy. */
    int
    addCommand(CLI::App &scenario, const std::vector<pluginMetadata> &metadata, std::ostream &err);

    /** @brief Return whether this scenario was selected by CLI11. */
    bool
    selected() const;

    /** @brief Resolve shared state, then validate scenario-specific state. */
    int
    finalize(std::ostream &err);

    /** @brief Print the resolved common and scenario-specific execution plan. */
    void
    printPlan(std::ostream &out) const;

    /** @brief Return whether execution was disabled by --dry-run. */
    bool
    dryRun() const;

    /** @brief Prepare scenario-owned resources before worker construction. */
    virtual bool
    prepare(std::ostream &err) const = 0;

    /** @brief Build the complete typed legacy worker configuration. */
    legacyWorkerConfig
    legacyWorkerConfiguration() const;

    /** @brief Create the scenario-specific worker strategy. */
    virtual std::unique_ptr<xferBenchWorker>
    createWorker(const std::vector<std::string> &devices) const = 0;

protected:
    /** @brief Add only options that are distinct to this scenario. */
    virtual void
    addScenarioOptions(CLI::App &command) = 0;

    /** @brief Validate and resolve only state that is distinct to this scenario. */
    virtual int
    finalizeScenario(std::ostream &err) = 0;

    /** @brief Print only plan details that are distinct to this scenario. */
    virtual void
    printScenarioPlan(std::ostream &out) const = 0;

    /** @brief Print scenario-specific dry-run resource guarantees. */
    virtual void
    printDryRunPlan(std::ostream &out) const = 0;

    /** @brief Add scenario-specific values to the legacy worker configuration. */
    virtual void
    configureLegacyWorker(legacyWorkerConfig &config) const = 0;

    /** @brief Return the resolved common scenario configuration. */
    const scenarioConfig &
    commonConfig() const;

    /** @brief Return parsed shared file options. */
    const fileOptions &
    commonFileOptions() const;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

/** @brief Result of parsing and resolving the explicit scenario command hierarchy. */
struct scenarioCommandResult {
    int status = EXIT_SUCCESS;
    bool execute = false;
    std::unique_ptr<benchmarkScenario> scenario;
};

/** @brief Return whether argv selects the explicit scenario hierarchy. */
bool
isScenarioCommand(int argc, char *argv[]);

/** @brief Discover plugins and prepare a scenario command for execution. */
scenarioCommandResult
prepareScenarioCommand(int argc, char *argv[], std::ostream &out, std::ostream &err);

/** @brief Prepare a scenario command using supplied plugin metadata. */
scenarioCommandResult
prepareScenarioCommand(int argc,
                       char *argv[],
                       const std::vector<pluginMetadata> &metadata,
                       std::ostream &out,
                       std::ostream &err);

} // namespace nixlbench

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_SCENARIO_H
