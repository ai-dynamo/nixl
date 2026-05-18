/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/g3_scenario_command.h"

#include "benchmark/benchmark_executor.h"
#include "benchmark/nixl_storage_allocator.h"
#include "benchmark/transfer_descriptor_strategy.h"
#include "benchmark_config.h"
#include "runtime/null_rt.h"
#include "utils/scope_guard.h"
#include "utils/utils.h"
#include "worker/nixl/nixl_backend_params.h"

#include <nixl.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <csignal>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <omp.h>
#include <string>
#include <string_view>
#include <utility>
#include <unistd.h>
#include <vector>

namespace nixlbench {
namespace {

std::atomic<int> g3_terminate{0};

void
g3SignalHandler(int signal) {
    (void)signal;
    static const char msg[] = "Ctrl-C received, exiting...\n";
    constexpr int stdout_fd = 1;
    constexpr int max_count = 1;
    auto size = write(stdout_fd, msg, sizeof(msg) - 1);
    (void)size;

    if (++g3_terminate > max_count) {
        std::_Exit(EXIT_FAILURE);
    }
}

bool
g3Signaled() {
    return g3_terminate.load() != 0;
}

size_t
parseG3FileSize(const std::string &input) {
    if (input.empty()) {
        return 0;
    }

    size_t suffix_pos = input.find_first_not_of("0123456789");
    const char *number_end = suffix_pos == std::string_view::npos ? input.data() + input.size() :
                                                                    input.data() + suffix_pos;

    size_t value = 0;
    auto [ptr, ec] = std::from_chars(input.data(), number_end, value);
    if (ec != std::errc{} || ptr != number_end) {
        return 0;
    }

    if (suffix_pos == std::string_view::npos) {
        return value;
    }

    std::string_view suffix(input.data() + suffix_pos, input.size() - suffix_pos);
    auto to_upper = [](char c) {
        return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    };

    size_t multiplier = 1;
    switch (to_upper(suffix[0])) {
    case 'K':
        multiplier = 1000LL;
        break;
    case 'M':
        multiplier = 1000000LL;
        break;
    case 'G':
        multiplier = 1000000000LL;
        break;
    case 'T':
        multiplier = 1000000000000LL;
        break;
    default:
        return value;
    }

    if (suffix.size() >= 2 && to_upper(suffix[1]) == 'I') {
        switch (to_upper(suffix[0])) {
        case 'K':
            multiplier = 1LL << 10;
            break;
        case 'M':
            multiplier = 1LL << 20;
            break;
        case 'G':
            multiplier = 1LL << 30;
            break;
        case 'T':
            multiplier = 1LL << 40;
            break;
        default:
            break;
        }
    }

    return value * multiplier;
}

const metadataPluginOptionValue *
findPluginOption(const metadata_plugin_option_map_t &options, const std::string &name) {
    const auto iter = options.find(name);
    return iter == options.end() ? nullptr : &iter->second;
}

std::string
pluginStringOption(const metadata_plugin_option_map_t &options,
                   const std::string &name,
                   const std::string &default_value = "") {
    const auto *option = findPluginOption(options, name);
    return option == nullptr ? default_value : option->value;
}

bool
pluginBoolOption(const metadata_plugin_option_map_t &options,
                 const std::string &name,
                 bool default_value = false) {
    const auto *option = findPluginOption(options, name);
    return option == nullptr ? default_value : option->boolValue;
}

int
pluginIntOption(const metadata_plugin_option_map_t &options,
                const std::string &name,
                int default_value = 1) {
    const auto *option = findPluginOption(options, name);
    if (option == nullptr || option->value.empty()) {
        return default_value;
    }

    try {
        return std::stoi(option->value);
    }
    catch (const std::exception &) {
        return default_value;
    }
}

void
iovListToNixlXferDlist(const std::vector<xferBenchIOV> &iov_list, nixl_xfer_dlist_t &dlist) {
    nixlBasicDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        dlist.addDesc(desc);
    }
}

nixl_status_t
executeSingleTransfer(nixlAgent &agent,
                      nixlXferReqH *req,
                      xferBenchTimer &timer,
                      xferBenchStats &thread_stats) {
    nixl_status_t rc = agent.postXferReq(req);
    thread_stats.post_duration.add(timer.lap());
    while (!g3Signaled() && NIXL_IN_PROG == rc) {
        rc = agent.getXferStatus(req);
    }
    return g3Signaled() ? NIXL_ERR_UNKNOWN : rc;
}

int
executeTransferIterations(nixlAgent &agent,
                          const nixl_xfer_op_t op,
                          nixl_xfer_dlist_t &local_desc,
                          nixl_xfer_dlist_t &remote_desc,
                          const std::string &target,
                          nixl_opt_args_t &params,
                          int num_iter,
                          xferBenchTimer &timer,
                          xferBenchStats &thread_stats) {
    if (num_iter <= 0) {
        return 0;
    }

    nixlXferReqH *req = nullptr;
    nixl_status_t create_rc = agent.createXferReq(op, local_desc, remote_desc, target, req, &params);
    if (NIXL_SUCCESS != create_rc) {
        std::cerr << "createXferReq failed: " << nixlEnumStrings::statusStr(create_rc) << std::endl;
        return EXIT_FAILURE;
    }
    thread_stats.prepare_duration.add(timer.lap());

    for (int i = 0; i < num_iter; ++i) {
        if (g3Signaled()) {
            agent.releaseXferReq(req);
            return EXIT_FAILURE;
        }

        nixl_status_t rc = executeSingleTransfer(agent, req, timer, thread_stats);
        if (rc != NIXL_SUCCESS) {
            std::cout << "NIXL Xfer failed with status: " << nixlEnumStrings::statusStr(rc)
                      << std::endl;
            agent.releaseXferReq(req);
            return EXIT_FAILURE;
        }
        thread_stats.transfer_duration.add(timer.lap());
    }

    if (agent.releaseXferReq(req) != NIXL_SUCCESS) {
        std::cout << "NIXL releaseXferReq failed" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int
executeTransfer(nixlAgent &agent,
                nixl_mem_t local_segment_type,
                nixl_mem_t remote_segment_type,
                const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                const std::vector<std::vector<xferBenchIOV>> &remote_iovs,
                nixl_xfer_op_t op,
                int num_iter,
                int num_threads,
                xferBenchStats &stats) {
    std::atomic<int> ret{EXIT_SUCCESS};
    stats.clear();

    xferBenchTimer total_timer;
#pragma omp parallel num_threads(num_threads)
    {
        xferBenchStats thread_stats;
        thread_stats.reserve(static_cast<size_t>(std::max(num_iter, 0)));
        xferBenchTimer timer;
        const int tid = omp_get_thread_num();
        const auto &local_iov = local_iovs[tid];
        const auto &remote_iov = remote_iovs[tid];

        nixl_xfer_dlist_t local_desc(local_segment_type);
        nixl_xfer_dlist_t remote_desc(remote_segment_type);
        iovListToNixlXferDlist(local_iov, local_desc);
        iovListToNixlXferDlist(remote_iov, remote_desc);

        nixl_opt_args_t params;
        const int result = executeTransferIterations(agent,
                                                     op,
                                                     local_desc,
                                                     remote_desc,
                                                     "initiator",
                                                     params,
                                                     num_iter,
                                                     timer,
                                                     thread_stats);
        if (result != EXIT_SUCCESS) {
            ret.store(result);
        }

#pragma omp critical
        { stats.add(thread_stats); }
    }

    stats.total_duration.add(total_timer.lap());
    return ret.load();
}

class g3NixlTransferStrategy : public benchmarkTransferStrategy {
public:
    g3NixlTransferStrategy(nixlAgent &agent,
                           const benchmarkConfig &config,
                           remoteIovStrategy &remote_strategy)
        : agent_(agent),
          config_(config),
          remote_strategy_(remote_strategy) {}

    std::variant<xferBenchStats, int>
    execute(const std::vector<std::vector<xferBenchIOV>> &local_descriptors) override {
        if (g3Signaled()) {
            return EXIT_FAILURE;
        }

        auto remote_result = remote_strategy_.createTransferIovs(local_descriptors,
                                                                 config_.transfer.start_block_size);
        if (std::holds_alternative<int>(remote_result)) {
            return std::get<int>(remote_result);
        }
        auto remote_descriptors = std::get<std::vector<std::vector<xferBenchIOV>>>(
            std::move(remote_result));

        int num_iter = config_.common.num_iter / config_.transfer.num_threads;
        int warmup_iter = config_.common.warmup_iter / config_.transfer.num_threads;
        if (config_.transfer.start_block_size > LARGE_BLOCK_SIZE) {
            num_iter /= config_.common.large_blk_iter_ftr;
            warmup_iter /= config_.common.large_blk_iter_ftr;
        }

        xferBenchStats stats;
        const nixl_xfer_op_t xfer_op = config_.transfer.op_type == XFERBENCH_OP_READ ? NIXL_READ :
                                                                                       NIXL_WRITE;
        int ret = executeTransfer(agent_,
                                  DRAM_SEG,
                                  FILE_SEG,
                                  local_descriptors,
                                  remote_descriptors,
                                  xfer_op,
                                  warmup_iter,
                                  config_.transfer.num_threads,
                                  stats);
        if (ret != EXIT_SUCCESS) {
            return ret;
        }

        stats.clear();
        ret = executeTransfer(agent_,
                              DRAM_SEG,
                              FILE_SEG,
                              local_descriptors,
                              remote_descriptors,
                              xfer_op,
                              num_iter,
                              config_.transfer.num_threads,
                              stats);
        if (ret != EXIT_SUCCESS) {
            return ret;
        }

        if (g3Signaled()) {
            return EXIT_FAILURE;
        }

        auto local_validation_descriptors = local_descriptors;
        if (!xferBenchUtils::validateTransfer(config_,
                                              true,
                                              local_validation_descriptors,
                                              remote_descriptors)) {
            return EXIT_FAILURE;
        }

        return stats;
    }

private:
    nixlAgent &agent_;
    benchmarkConfig config_;
    remoteIovStrategy &remote_strategy_;
};

class g3StatsResultSink : public benchmarkResultSink {
public:
    explicit
    g3StatsResultSink(benchmarkConfig config)
        : config_(std::move(config)) {}

    void
    record(const xferBenchStats &stats) override {
        xferBenchUtils::printStats(config_,
                                   false,
                                   config_.transfer.start_block_size,
                                   config_.transfer.start_batch_size,
                                   stats);
    }

private:
    benchmarkConfig config_;
};

benchmarkConfig
makeG3BenchmarkConfig(const g3ScenarioRequest &request,
                      southboundPluginBenchmarkCommand &plugin) {
    const auto &metadata = plugin.metadataOptions();
    benchmarkConfig config;
    config.backend.name = std::string(plugin.name());
    config.backend.capabilities = plugin.capabilities();
    config.backend.memory_types = plugin.supportedMemoryTypes();
    config.backend.options = metadata;
    config.transfer.num_threads = request.parallel_threads;
    config.transfer.start_block_size = request.block_size_bytes;
    config.transfer.max_block_size = request.block_size_bytes;
    config.transfer.start_batch_size = request.batch_size;
    config.transfer.max_batch_size = request.batch_size;
    config.transfer.op_type = request.action_mode == "read" || request.action_mode == "READ" ?
        XFERBENCH_OP_READ :
        XFERBENCH_OP_WRITE;
    config.transfer.total_buffer_size = parseG3FileSize(request.file_size);
    config.storage.filepath = pluginStringOption(metadata, "filepath");
    config.storage.filenames = pluginStringOption(metadata, "filenames");
    config.storage.num_files = request.parallel_threads * pluginIntOption(metadata, "num_files", 1);
    config.storage.enable_direct = pluginBoolOption(metadata, "enable_direct");
    if (config.backend.capabilities.requiresDirectStorage) {
        config.storage.enable_direct = true;
    }
    return config;
}

} // namespace

g3ScenarioCommand::g3ScenarioCommand()
    : options_{
          cliOption::option("file-size",
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
g3ScenarioCommand::supportsPlugin(nixl_mem_list_t supportedMemoryTypes, nixlBackendPluginCapabilities pluginCapabilities) const {
    if (std::find(supportedMemoryTypes.begin(), supportedMemoryTypes.end(), FILE_SEG) == supportedMemoryTypes.end()) {
        return false;
    }
    return true;
}

bool
g3ScenarioCommand::isRequestValid(const g3ScenarioRequest &request) const {
    // validate file size
    const size_t file_size = parseG3FileSize(request.file_size);
    if (file_size == 0) {
        return false;
    }

    if (request.batch_size == 0 || request.batch_size > file_size) {
        return false;
    }

    if (request.block_size_bytes == 0) {
        return false;
    }

    if (request.parallel_threads <= 0) {
        return false;
    }

    return true;
}

int
g3ScenarioCommand::run(southboundPluginBenchmarkCommand &plugin) {
    // this should  never occur as the CLI should only present things that have the proper capabilities, but this is here just in case...
    if (!supportsPlugin(plugin.supportedMemoryTypes(), plugin.capabilities())) {
        std::cerr << "G3 requires a plugin that can read and write files" << std::endl;
        return 1;
    }

    if (!isRequestValid(request_)) {
        return 1;
    }

    g3_terminate.store(0);
    auto previous_signal_handler = std::signal(SIGINT, g3SignalHandler);
    auto signal_guard = make_scope_guard([previous_signal_handler] {
        std::signal(SIGINT, previous_signal_handler);
    });

    benchmarkConfig benchmark_config = makeG3BenchmarkConfig(request_, plugin);
    xferBenchNullRT runtime;
    xferBenchUtils::setRT(&runtime);
    std::cout << "Single instance storage backend - no synchronization needed" << std::endl;

    nixlAgentConfig agent_config;
    agent_config.syncMode = benchmark_config.transfer.num_threads > 1 ?
        nixl_thread_sync_t::NIXL_THREAD_SYNC_RW :
        nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT;
    nixlAgent agent("initiator", agent_config);

    nixl_mem_list_t mems;
    nixl_b_params_t backend_params;
    nixl_status_t status = agent.getPluginParams(benchmark_config.backend.name, mems, backend_params);
    if (status != NIXL_SUCCESS) {
        std::cerr << "getPluginParams failed: " << nixlEnumStrings::statusStr(status) << std::endl;
        return EXIT_FAILURE;
    }

    backend_params = applyPluginOptions(benchmark_config.backend.options, backend_params);
    nixlBackendH *backend = nullptr;
    status = agent.createBackend(benchmark_config.backend.name, backend_params, backend);
    if (status != NIXL_SUCCESS) {
        std::cerr << "createBackend failed: " << nixlEnumStrings::statusStr(status) << std::endl;
        return EXIT_FAILURE;
    }

    nullBenchmarkRuntimeSync sync;
    dramLocalIovStrategy local_iovs;
    fileRemoteIovStrategy remote_iovs(benchmark_config.storage, benchmark_config.backend.name, benchmark_config.transfer.op_type);
    nixlStorageAllocator allocator(agent,
                                   backend,
                                   benchmark_config.transfer.num_threads,
                                   benchmark_config.transfer.total_buffer_size,
                                   benchmark_config.storage.enable_direct,
                                   local_iovs,
                                   &remote_iovs);

    auto descriptors = makeTransferDescriptorStrategy(benchmark_config,
                                                      request_.randomized_read_location);
    g3NixlTransferStrategy transfer(agent, benchmark_config, remote_iovs);
    fixedIterationPolicy iterations(1, benchmarkAllocationLifecycle::AllocateOnce);
    g3StatsResultSink results(benchmark_config);

    xferBenchUtils::printStatsHeader(benchmark_config);
    benchmarkRunComponents components{sync, allocator, *descriptors, transfer, iterations, results};
    benchmarkExecutor executor;
    int ret = executor.run(components);
    if (g3Signaled()) {
        return EXIT_FAILURE;
    }

    return ret;
}

const g3ScenarioRequest &
g3ScenarioCommand::request() const {
    return request_;
}

} // namespace nixlbench
