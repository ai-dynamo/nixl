/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/g3_scenario_command.h"
#include "worker/nixl/nixl_worker.h"
#include "benchmark_runner.h"
#include <memory>
#include <vector>

namespace nixlbench {

g3ScenarioCommand::g3ScenarioCommand()
    : options_{
          cliOption::option("file_size",
                            "File size, can be shorthand (5M, 10G ...)",
                            &request_.file_size,
                            false),
          cliOption::option("parallel-threads",
                            "Parallel threads",
                            &request_.parallel_threads,
                            false),
          cliOption::option("block-size",
                            "Block size - amount of data to transfer in each transfer in bytes",
                            &request_.block_size_bytes,
                            false),
          cliOption::option("batch-size",
                            "Batch size - number of data transfers to perform in each batch",
                            &request_.batch_size,
                            false),
          cliOption::option(
              "action-mode",
              "Sets whether the benchmark will read, write, or interleave reading and writing",
              &request_.action_mode,
              false),
          cliOption::option(
              "randomized-read-location",
              "Whether to read / write in random locations or sequentially (default true)",
              &request_.randomized_read_location,
              false)} {}

std::string_view
g3ScenarioCommand::name() const {
    return "g3";
}

std::string_view
g3ScenarioCommand::description() const {
    return "Run G3 storage scenario.  The G3 storage scenario reads or writes to a large per "
           "thread file in batches.  The file is opened once and closed at the end of the "
           "benchmark.  It only supports plugins that can read and write to files.";
}

const std::vector<cliOption> &
g3ScenarioCommand::getOptions() const {
    return options_;
}

scenario_type_t
g3ScenarioCommand::scenarioType() const {
    return scenario_type_t::G3;
}

bool
g3ScenarioCommand::supportsPlugin(nixlBackendPluginCapabilities pluginCapabilities) const {
    return pluginCapabilities.canReadWriteFiles;
}

bool
g3ScenarioCommand::isRequestValid(const g3ScenarioRequest &request) const {
    // validate file size
    if (request.file_size.empty()) {
        return false;
    }

    if (request.batch_size > parse_file_size(request.file_size)) {
        return false;
    }

    if (request.parallel_threads <= 0) {
        return false;
    }

    return true;
}

int
g3ScenarioCommand::run(southboundPluginBenchmarkCommand &plugin) {
    // validate request
    if (!isRequestValid(request_)) {
        return 1;
    }

    // run benchmark

    // create worker
    std::vector<std::string> devices;
    devices.push_back(std::string(plugin.name()));
    auto worker_ptr = std::make_unique<xferBenchNixlWorker>(devices);

    // create files, return handles
    std::vector<std::vector<xferBenchIOV>> iov_lists =
        worker_ptr->allocateMemory(request_.parallel_threads);


    int ret = processBatchSizes(*worker_ptr,
                      iov_lists,
                      request_.block_size_bytes,
                      request_.parallel_threads,
                      request_.randomized_read_location);
    if (ret != 0) {
        return ret;
    }

    // clean up
    worker_ptr->deallocateMemory(iov_lists);

    return 0;
}

const g3ScenarioRequest &
g3ScenarioCommand::request() const {
    return request_;
}

} // namespace nixlbench
