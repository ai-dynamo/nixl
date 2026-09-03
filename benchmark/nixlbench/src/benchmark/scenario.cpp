/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/scenario.h"

#include "benchmark/allocate_once.h"
#include "utils/utils.h"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace nixlbench {
namespace {

    constexpr int fixed_scenario_large_block_iteration_factor = 1;
    constexpr int fixed_scenario_pipeline_depth = 1;

    struct scenarioOptions {
        size_t blockSize = 0;
        size_t batchSize = 1;
        int threads = 1;
        int iterations = 1000;
        int warmupIterations = 10;
        std::string operation = "write";
        std::string initiatorMemory = "auto";
        bool checkConsistency = false;
        bool dryRun = false;
    };

    struct scenarioPluginBinding {
        pluginMetadata metadata;
        CLI::App *command = nullptr;
        std::vector<std::pair<std::string, std::string>> overrides;
    };

    using scenario_plugin_bindings_t = std::vector<std::unique_ptr<scenarioPluginBinding>>;

    std::string
    upper(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
            return static_cast<char>(std::toupper(character));
        });
        return value;
    }

    std::string
    lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return value;
    }

    std::string
    legacyMemoryName(nixl_mem_t memory) {
        switch (memory) {
        case DRAM_SEG:
            return XFERBENCH_SEG_TYPE_DRAM;
        case VRAM_SEG:
            return XFERBENCH_SEG_TYPE_VRAM;
        case FILE_SEG:
            return XFERBENCH_SEG_TYPE_FILE;
        case BLK_SEG:
            return XFERBENCH_SEG_TYPE_BLK;
        case OBJ_SEG:
            return XFERBENCH_BACKEND_OBJ;
        }
        return {};
    }

    std::string
    displayMemoryName(nixl_mem_t memory) {
        return memory == VRAM_SEG ? "VRAM" : "DRAM";
    }

    std::string
    displayOperationName(nixl_xfer_op_t operation) {
        return operation == NIXL_READ ? "READ" : "WRITE";
    }

    std::vector<std::unique_ptr<benchmarkScenario>>
    scenarioRegistry() {
        std::vector<std::unique_ptr<benchmarkScenario>> scenarios;
        scenarios.push_back(std::make_unique<allocateOnceScenario>());
        return scenarios;
    }

} // namespace

static void
addCommonScenarioOptions(CLI::App &command, scenarioOptions &options);

static bool
resolveCommonScenarioOptions(const scenarioOptions &options,
                             const pluginMetadata &metadata,
                             const std::vector<std::pair<std::string, std::string>> &overrides,
                             scenarioConfig &config,
                             std::ostream &err);

static int
addScenarioPluginCommands(CLI::App &scenario,
                          const std::vector<pluginMetadata> &metadata,
                          const scenario_plugin_filter_t &filter,
                          scenario_plugin_bindings_t &bindings,
                          std::ostream &err);

static const scenarioPluginBinding *
selectedScenarioPlugin(const scenario_plugin_bindings_t &bindings);

struct benchmarkScenario::implementation {
    implementation(std::string scenario_name,
                   std::string scenario_description,
                   scenario_plugin_filter_t plugin_filter,
                   bool scenario_has_file_options)
        : name(std::move(scenario_name)),
          description(std::move(scenario_description)),
          filter(std::move(plugin_filter)),
          hasFileOptions(scenario_has_file_options) {}

    std::string name;
    std::string description;
    scenario_plugin_filter_t filter;
    bool hasFileOptions = false;
    scenarioOptions options;
    fileOptions file;
    CLI::App *command = nullptr;
    scenario_plugin_bindings_t plugins;
    scenarioConfig config;
};

benchmarkScenario::benchmarkScenario(std::string name,
                                     std::string description,
                                     scenario_plugin_filter_t plugin_filter,
                                     bool has_file_options)
    : implementation_(std::make_unique<implementation>(std::move(name),
                                                       std::move(description),
                                                       std::move(plugin_filter),
                                                       has_file_options)) {}

benchmarkScenario::~benchmarkScenario() = default;

int
benchmarkScenario::addCommand(CLI::App &scenario,
                              const std::vector<pluginMetadata> &metadata,
                              std::ostream &err) {
    implementation_->command =
        scenario.add_subcommand(implementation_->name, implementation_->description);
    implementation_->command->require_subcommand(1);

    addScenarioOptions(*implementation_->command);
    addCommonScenarioOptions(*implementation_->command, implementation_->options);
    if (implementation_->hasFileOptions) {
        addFileOptions(*implementation_->command, implementation_->file);
    }
    return addScenarioPluginCommands(*implementation_->command,
                                     metadata,
                                     implementation_->filter,
                                     implementation_->plugins,
                                     err);
}

bool
benchmarkScenario::selected() const {
    return implementation_->command != nullptr && implementation_->command->parsed();
}

int
benchmarkScenario::finalize(std::ostream &err) {
    const auto *selected_plugin = selectedScenarioPlugin(implementation_->plugins);
    if (selected_plugin == nullptr) {
        err << "Error: " << implementation_->name << " requires an installed compatible plugin\n";
        return inval_args_exit_code;
    }
    if (!resolveCommonScenarioOptions(implementation_->options,
                                      selected_plugin->metadata,
                                      selected_plugin->overrides,
                                      implementation_->config,
                                      err)) {
        return inval_args_exit_code;
    }
    const int status = finalizeScenario(err);
    if (status == EXIT_SUCCESS) {
        implementation_->command = nullptr;
        implementation_->plugins.clear();
    }
    return status;
}

void
benchmarkScenario::printPlan(std::ostream &out) const {
    const auto &config = implementation_->config;
    out << "Resolved NIXLBench scenario\n"
        << "  scenario: " << implementation_->name << "\n  backend: " << config.pluginName
        << "\n  initiator memory: " << displayMemoryName(config.initiatorMemory);
    printScenarioPlan(out);
    out << "\n  block size: " << formatSize(config.blockSize)
        << "\n  batch size: " << config.batchSize << " blocks"
        << "\n  worker threads: " << config.threads
        << "\n  operation: " << displayOperationName(config.operation)
        << "\n  warmup requests: " << config.warmupIterations << " per thread"
        << "\n  timed requests: " << config.iterations << " per thread, "
        << static_cast<uint64_t>(config.iterations) * static_cast<uint64_t>(config.threads)
        << " aggregate"
        << "\n  consistency check: " << (config.checkConsistency ? "enabled" : "disabled")
        << "\n  plugin parameters:\n";
    for (const auto &key : sortedParameterKeys(config.pluginParameters)) {
        out << "    " << key << ": " << config.pluginParameters.at(key) << '\n';
    }
    if (config.dryRun) {
        printDryRunPlan(out);
    }
}

bool
benchmarkScenario::dryRun() const {
    return implementation_->config.dryRun;
}

legacyWorkerConfig
benchmarkScenario::legacyWorkerConfiguration() const {
    legacyWorkerConfig config;
    config.common = implementation_->config;
    configureLegacyWorker(config);
    return config;
}

const scenarioConfig &
benchmarkScenario::commonConfig() const {
    return implementation_->config;
}

const fileOptions &
benchmarkScenario::commonFileOptions() const {
    return implementation_->file;
}

static void
addCommonScenarioOptions(CLI::App &command, scenarioOptions &options) {
    command.add_option("--block-size", options.blockSize, "Bytes in each transferred block")
        ->transform(binarySizeTransform())
        ->check(CLI::PositiveNumber)
        ->required()
        ->group("Common scenario options");
    command.add_option("--batch-size", options.batchSize, "Block descriptors in each request")
        ->group("Common scenario options");
    command.add_option("--threads", options.threads, "Parallel transfer threads")
        ->group("Common scenario options");
    command.add_option("--iterations", options.iterations, "Timed requests per thread")
        ->group("Common scenario options");
    command
        .add_option(
            "--warmup-iterations", options.warmupIterations, "Untimed warmup requests per thread")
        ->group("Common scenario options");
    command.add_option("--operation", options.operation, "Transfer direction: read or write")
        ->check(CLI::IsMember({"read", "write"}, CLI::ignore_case))
        ->group("Common scenario options");
    command
        .add_option("--initiator-memory",
                    options.initiatorMemory,
                    "Local buffer placement: auto, dram, or vram")
        ->check(CLI::IsMember({"auto", "dram", "vram"}, CLI::ignore_case))
        ->group("Common scenario options");
    command
        .add_flag("--check-consistency",
                  options.checkConsistency,
                  "Validate transferred bytes outside transfer timing")
        ->group("Common scenario options");
    command.add_flag("--dry-run", options.dryRun, "Print the resolved plan without executing")
        ->group("Common scenario options");
}

static int
addScenarioPluginCommands(CLI::App &scenario,
                          const std::vector<pluginMetadata> &metadata,
                          const scenario_plugin_filter_t &filter,
                          scenario_plugin_bindings_t &bindings,
                          std::ostream &err) {
    std::set<std::string> command_names;
    bindings.reserve(metadata.size());
    for (const auto &entry : metadata) {
        if (!filter(entry)) {
            continue;
        }
        const std::string command_name = lower(entry.name);
        if (!command_names.insert(command_name).second) {
            err << "Error: installed plugin names are ambiguous when used as CLI subcommands: "
                << command_name << '\n';
            return inval_args_exit_code;
        }

        auto binding = std::make_unique<scenarioPluginBinding>();
        binding->metadata = entry;
        binding->command =
            scenario.add_subcommand(command_name, "Run the installed " + entry.name + " backend");
        binding->command->fallthrough();
        binding->command->footer("Scenario options may be used before or after this plugin "
                                 "subcommand.");
        addPluginOptions(*binding->command, entry.parameters, binding->overrides);
        bindings.push_back(std::move(binding));
    }
    return EXIT_SUCCESS;
}

static const scenarioPluginBinding *
selectedScenarioPlugin(const scenario_plugin_bindings_t &bindings) {
    for (const auto &binding : bindings) {
        if (binding->command->parsed()) {
            return binding.get();
        }
    }
    return nullptr;
}

static bool
resolveCommonScenarioOptions(const scenarioOptions &options,
                             const pluginMetadata &metadata,
                             const std::vector<std::pair<std::string, std::string>> &overrides,
                             scenarioConfig &config,
                             std::ostream &err) {
    const auto fail = [&](const std::string &message) {
        err << "Error: " << message << '\n';
        return false;
    };

    if (options.batchSize == 0 || options.threads < 1 || options.iterations < 1 ||
        options.warmupIterations < 0) {
        return fail("batch size, threads, and iterations must be positive; warmup iterations "
                    "may be zero");
    }
    if (options.iterations > std::numeric_limits<int>::max() / options.threads ||
        options.warmupIterations > std::numeric_limits<int>::max() / options.threads) {
        return fail("aggregate iteration count is too large");
    }

    const std::string requested_memory = upper(options.initiatorMemory);
    const bool supports_dram = hasMemoryType(metadata, DRAM_SEG);
    const bool supports_vram = hasMemoryType(metadata, VRAM_SEG);
    if (requested_memory == "DRAM" && !supports_dram) {
        return fail(metadata.name + " does not advertise DRAM_SEG");
    }
    if (requested_memory == "VRAM" && !supports_vram) {
        return fail(metadata.name + " does not advertise VRAM_SEG");
    }

    config.pluginName = metadata.name;
    config.pluginParameters = metadata.parameters;
    for (const auto &[key, value] : overrides) {
        config.pluginParameters[key] = value;
    }
    config.blockSize = options.blockSize;
    config.batchSize = options.batchSize;
    config.threads = options.threads;
    config.iterations = options.iterations;
    config.warmupIterations = options.warmupIterations;
    config.operation = upper(options.operation) == "READ" ? NIXL_READ : NIXL_WRITE;
    if (requested_memory == "VRAM" || (requested_memory == "AUTO" && supports_vram)) {
        config.initiatorMemory = VRAM_SEG;
    } else if (supports_dram) {
        config.initiatorMemory = DRAM_SEG;
    } else {
        return fail(metadata.name + " has no supported initiator memory type");
    }
    config.checkConsistency = options.checkConsistency;
    config.dryRun = options.dryRun;
    return true;
}

std::vector<std::string>
legacyWorkerArguments(const legacyWorkerConfig &config, const std::string &program_name) {
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    std::ostringstream file_names;
    for (size_t index = 0; index < config.fileNames.size(); ++index) {
        if (index != 0) {
            file_names << ',';
        }
        file_names << config.fileNames[index];
    }

    std::vector<std::string> arguments = {
        program_name,
        std::string("--worker_type=") + XFERBENCH_WORKER_NIXL,
        "--backend=" + config.common.pluginName,
        "--initiator_seg_type=" + legacyMemoryName(config.common.initiatorMemory),
        "--target_seg_type=" + legacyMemoryName(config.targetMemory),
        std::string("--op_type=") +
            (config.common.operation == NIXL_READ ? XFERBENCH_OP_READ : XFERBENCH_OP_WRITE),
        "--check_consistency=" + std::string(boolean(config.common.checkConsistency)),
        "--total_buffer_size=" + std::to_string(config.workingMemory),
        "--start_block_size=" + std::to_string(config.common.blockSize),
        "--max_block_size=" + std::to_string(config.common.blockSize),
        "--start_batch_size=" + std::to_string(config.common.batchSize),
        "--max_batch_size=" + std::to_string(config.common.batchSize),
        "--num_iter=" + std::to_string(config.common.iterations * config.common.threads),
        "--warmup_iter=" + std::to_string(config.common.warmupIterations * config.common.threads),
        "--num_threads=" + std::to_string(config.common.threads),
        "--large_blk_iter_ftr=" + std::to_string(fixed_scenario_large_block_iteration_factor),
        "--pipeline_depth=" + std::to_string(fixed_scenario_pipeline_depth),
        "--recreate_xfer=" + std::string(boolean(config.recreateTransferRequest)),
        "--filenames=" + file_names.str(),
        "--num_files=" + std::to_string(config.fileNames.size()),
        "--storage_enable_direct=" + std::string(boolean(config.storageDirect)),
    };
    return arguments;
}

bool
isScenarioCommand(int argc, char *argv[]) {
    return argc > 1 && std::string_view(argv[1]) == "scenario";
}

scenarioCommandResult
prepareScenarioCommand(int argc,
                       char *argv[],
                       const std::vector<pluginMetadata> &metadata,
                       std::ostream &out,
                       std::ostream &err) {
    CLI::App app("NIXL data-transfer benchmark");
    app.require_subcommand(1);
    auto *scenario_command = app.add_subcommand("scenario", "Run a modeled transfer workload");
    scenario_command->require_subcommand(1);

    auto scenarios = scenarioRegistry();
    for (auto &scenario : scenarios) {
        const int status = scenario->addCommand(*scenario_command, metadata, err);
        if (status != EXIT_SUCCESS) {
            return {status, false, nullptr};
        }
    }

    try {
        app.parse(argc, argv);
    }
    catch (const CLI::CallForHelp &exception) {
        return {app.exit(exception, out, err), false, nullptr};
    }
    catch (const CLI::ParseError &exception) {
        return {app.exit(exception, out, err), false, nullptr};
    }

    std::unique_ptr<benchmarkScenario> selected;
    for (auto &scenario : scenarios) {
        if (!scenario->selected()) {
            continue;
        }
        if (selected) {
            err << "Error: exactly one scenario must be selected\n";
            return {inval_args_exit_code, false, nullptr};
        }
        selected = std::move(scenario);
    }
    if (!selected) {
        err << "Error: a scenario must be selected\n";
        return {inval_args_exit_code, false, nullptr};
    }

    const int status = selected->finalize(err);
    if (status != EXIT_SUCCESS) {
        return {status, false, nullptr};
    }
    selected->printPlan(out);
    const bool execute = !selected->dryRun();
    return {EXIT_SUCCESS, execute, std::move(selected)};
}

scenarioCommandResult
prepareScenarioCommand(int argc, char *argv[], std::ostream &out, std::ostream &err) {
    std::string discovery_error;
    auto metadata = discoverPluginMetadata(discovery_error);
    if (!metadata) {
        err << "Error: " << discovery_error << '\n';
        return {EXIT_FAILURE, false, nullptr};
    }
    return prepareScenarioCommand(argc, argv, *metadata, out, err);
}

} // namespace nixlbench
