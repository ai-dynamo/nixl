/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/raw_cli.h"

#include "utils/utils.h"

#include <CLI/CLI.hpp>
#include <nixl.h>

#include <algorithm>
#include <cctype>
#include <string_view>
#include <utility>

namespace nixlbench {
namespace {

    std::string
    upper(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });
        return value;
    }

    void
    addRawOptions(CLI::App &raw, RawOptions &options) {
        raw.add_option("--operation", options.operation, "Transfer direction: read or write")
            ->check(CLI::IsMember({"read", "write"}, CLI::ignore_case))
            ->group("Raw benchmark options");
        raw.add_option("--total-buffer-size",
                       options.total_buffer_size,
                       "Total buffer size using binary units (for example 4MB)")
            ->transform(binarySizeTransform())
            ->check(CLI::PositiveNumber)
            ->default_str(formatSize(options.total_buffer_size))
            ->group("Raw benchmark options");
        raw.add_option(
               "--start-block-size", options.start_block_size, "First block size in the sweep")
            ->transform(binarySizeTransform())
            ->check(CLI::PositiveNumber)
            ->default_str(formatSize(options.start_block_size))
            ->group("Raw benchmark options");
        raw.add_option("--max-block-size", options.max_block_size, "Last block size in the sweep")
            ->transform(binarySizeTransform())
            ->check(CLI::PositiveNumber)
            ->default_str(formatSize(options.max_block_size))
            ->group("Raw benchmark options");
        raw.add_option(
               "--start-batch-size", options.start_batch_size, "First batch size in the sweep")
            ->check(CLI::PositiveNumber)
            ->group("Raw benchmark options");
        raw.add_option("--max-batch-size", options.max_batch_size, "Last batch size in the sweep")
            ->check(CLI::PositiveNumber)
            ->group("Raw benchmark options");
        raw.add_option("--iterations", options.iterations, "Timed iterations")
            ->check(CLI::PositiveNumber)
            ->group("Raw benchmark options");
        raw.add_option("--warmup-iterations", options.warmup_iterations, "Warmup iterations")
            ->check(CLI::NonNegativeNumber)
            ->group("Raw benchmark options");
        raw.add_option("--threads", options.threads, "Benchmark worker threads")
            ->check(CLI::PositiveNumber)
            ->group("Raw benchmark options");
        raw.add_option("--pipeline-depth", options.pipeline_depth, "Transfer requests in flight")
            ->check(CLI::PositiveNumber)
            ->group("Raw benchmark options");
        raw.add_flag("--check-consistency", options.check_consistency, "Validate transferred bytes")
            ->group("Raw benchmark options");
        raw.add_flag("--dry-run", options.dry_run, "Print the resolved plan without executing")
            ->group("Raw benchmark options");
    }

    bool
    validateRawOptions(const RawOptions &raw, std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        if (raw.max_block_size < raw.start_block_size) {
            return fail("max block size must be at least start block size");
        }
        if (raw.max_batch_size < raw.start_batch_size) {
            return fail("max batch size must be at least start batch size");
        }
        return true;
    }

    bool
    validateFileOptions(const fileOptions &file, const RawOptions &raw, std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        std::string file_error;
        if (!nixlbench::validateFileOptions(file, file_error)) {
            return fail(file_error);
        }
        if (file.numFiles > raw.threads || raw.threads % file.numFiles != 0) {
            return fail("--num-files must divide --threads and cannot exceed it");
        }
        return true;
    }

} // namespace

void
printRawPlan(const RawCommandRequest &request,
             const pluginMetadata &metadata,
             int normalized_iterations,
             int normalized_warmup_iterations,
             std::ostream &out) {
    out << "Resolved NIXLBench plan\n"
        << "  command: raw posix\n"
        << "  backend: " << metadata.name << "\n"
        << "  memory types: ";
    auto memory_types = metadata.memoryTypes;
    std::sort(memory_types.begin(), memory_types.end());
    for (size_t i = 0; i < memory_types.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << nixlEnumStrings::memTypeStr(memory_types[i]);
    }
    out << "\n  benchmark options:\n"
        << "    operation: " << request.raw.operation
        << "\n    total buffer: " << formatSize(request.raw.total_buffer_size)
        << "\n    block sizes: " << formatSize(request.raw.start_block_size) << " .. "
        << formatSize(request.raw.max_block_size)
        << "\n    batch sizes: " << request.raw.start_batch_size << " .. "
        << request.raw.max_batch_size;
    if (request.raw.iterations == normalized_iterations &&
        request.raw.warmup_iterations == normalized_warmup_iterations) {
        out << "\n    iterations: " << request.raw.iterations << " (warmup "
            << request.raw.warmup_iterations << ")";
    } else {
        out << "\n    requested iterations: " << request.raw.iterations << " (warmup "
            << request.raw.warmup_iterations << ")"
            << "\n    normalized iterations: " << normalized_iterations << " (warmup "
            << normalized_warmup_iterations << ")"
            << "\n    normalization: aligned for thread and large-block iteration distribution";
    }
    out << "\n    threads: " << request.raw.threads
        << "\n    pipeline depth: " << request.raw.pipeline_depth
        << "\n    consistency check: " << (request.raw.check_consistency ? "enabled" : "disabled");
    if (request.has_file_options) {
        out << "\n  file-resource options:\n"
            << "    path: "
            << (request.file.path.empty() ? "<current working directory>" : request.file.path)
            << "\n    filenames: "
            << (request.file.filenames.empty() ? "<automatic>" : request.file.filenames)
            << "\n    files: " << request.file.numFiles
            << "\n    direct I/O: " << (request.file.direct ? "enabled" : "disabled");
    }
    out << "\n  plugin parameters:\n";
    for (const auto &key : sortedParameterKeys(request.plugin_parameters)) {
        out << "    " << key << ": " << request.plugin_parameters.at(key) << '\n';
    }
    if (request.raw.dry_run) {
        out << "Dry run: no worker was created and no allocation or transfer was attempted.\n";
    }
}

bool
isRawCommand(int argc, char *argv[]) {
    return argc > 1 && std::string_view(argv[1]) == "raw";
}

int
parseRawPosixCommand(int argc,
                     char *argv[],
                     const pluginMetadata &metadata,
                     RawCommandRequest &request,
                     bool &help_requested,
                     std::ostream &out,
                     std::ostream &err) {
    help_requested = false;
    request.plugin_parameters = metadata.parameters;
    request.has_file_options = hasMemoryType(metadata, FILE_SEG);

    std::vector<std::pair<std::string, std::string>> plugin_parameter_overrides;

    CLI::App app("NIXL data-transfer benchmark");
    app.require_subcommand(1);
    auto *raw = app.add_subcommand("raw", "Configure a low-level benchmark explicitly");
    raw->require_subcommand(1);
    auto *posix = raw->add_subcommand("posix", "Run the installed POSIX storage backend");
    posix->fallthrough();
    posix->footer("Raw benchmark and FILE_SEG options are documented by 'nixlbench raw --help' "
                  "and may be used before or after the posix subcommand.");

    addRawOptions(*raw, request.raw);

    if (request.has_file_options) {
        addFileOptions(*raw, request.file);
    }
    addPluginOptions(*posix, metadata.parameters, plugin_parameter_overrides);

    try {
        app.parse(argc, argv);
    }
    catch (const CLI::CallForHelp &exception) {
        help_requested = true;
        return app.exit(exception, out, err);
    }
    catch (const CLI::ParseError &exception) {
        return app.exit(exception, out, err);
    }

    if (!hasMemoryType(metadata, DRAM_SEG)) {
        err << "Error: " << metadata.name << " plugin must advertise DRAM_SEG for local memory\n";
        return inval_args_exit_code;
    }
    if (!hasMemoryType(metadata, FILE_SEG)) {
        err << "Error: " << metadata.name << " plugin must advertise FILE_SEG for backing files\n";
        return inval_args_exit_code;
    }

    for (const auto &[key, value] : plugin_parameter_overrides) {
        request.plugin_parameters[key] = value;
    }

    request.raw.operation = upper(request.raw.operation);
    if (!validateRawOptions(request.raw, err)) {
        return inval_args_exit_code;
    }
    if (request.has_file_options && !validateFileOptions(request.file, request.raw, err)) {
        return inval_args_exit_code;
    }
    return EXIT_SUCCESS;
}

std::vector<std::string>
benchmarkFileArguments(const RawCommandRequest &request, const std::string &program_name) {
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    std::vector<std::string> arguments = {
        program_name,
        // Fixed values select the existing NIXL/POSIX runner.
        std::string("--worker_type=") + XFERBENCH_WORKER_NIXL,
        std::string("--backend=") + XFERBENCH_BACKEND_POSIX,
        std::string("--initiator_seg_type=") + XFERBENCH_SEG_TYPE_DRAM,
        // Backend-neutral raw benchmark configuration.
        "--op_type=" + request.raw.operation,
        "--check_consistency=" + std::string(boolean(request.raw.check_consistency)),
        "--total_buffer_size=" + std::to_string(request.raw.total_buffer_size),
        "--start_block_size=" + std::to_string(request.raw.start_block_size),
        "--max_block_size=" + std::to_string(request.raw.max_block_size),
        "--start_batch_size=" + std::to_string(request.raw.start_batch_size),
        "--max_batch_size=" + std::to_string(request.raw.max_batch_size),
        "--num_iter=" + std::to_string(request.raw.iterations),
        "--warmup_iter=" + std::to_string(request.raw.warmup_iterations),
        "--num_threads=" + std::to_string(request.raw.threads),
        "--pipeline_depth=" + std::to_string(request.raw.pipeline_depth)};
    if (request.has_file_options) {
        arguments.push_back("--filepath=" + request.file.path);
        arguments.push_back("--filenames=" + request.file.filenames);
        arguments.push_back("--num_files=" + std::to_string(request.file.numFiles));
        arguments.push_back("--storage_enable_direct=" + std::string(boolean(request.file.direct)));
    }
    return arguments;
}

RawCommandResult
prepareRawCommand(int argc, char *argv[], std::ostream &out, std::ostream &err) {
    std::string discovery_error;
    const auto metadata = discoverPluginMetadata(XFERBENCH_BACKEND_POSIX, discovery_error);
    if (!metadata) {
        err << "Error: " << discovery_error << '\n';
        return {EXIT_FAILURE, false};
    }

    RawCommandRequest request;
    bool help_requested = false;
    const int parse_status =
        parseRawPosixCommand(argc, argv, *metadata, request, help_requested, out, err);
    if (parse_status != EXIT_SUCCESS || help_requested) {
        return {parse_status, false};
    }

    auto arguments = benchmarkFileArguments(request, argv[0]);
    std::vector<char *> argument_pointers;
    argument_pointers.reserve(arguments.size());
    for (auto &argument : arguments) {
        argument_pointers.push_back(argument.data());
    }
    int legacy_argc = static_cast<int>(argument_pointers.size());
    char **legacy_argv = argument_pointers.data();
    if (xferBenchConfig::parseConfig(legacy_argc, legacy_argv) != EXIT_SUCCESS) {
        return {EXIT_FAILURE, false};
    }

    printRawPlan(request, *metadata, xferBenchConfig::num_iter, xferBenchConfig::warmup_iter, out);
    if (!request.raw.dry_run) {
        xferBenchConfig::plugin_parameters = std::move(request.plugin_parameters);
    }
    return {EXIT_SUCCESS, !request.raw.dry_run};
}

} // namespace nixlbench
