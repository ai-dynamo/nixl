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
#include <charconv>
#include <iomanip>
#include <limits>
#include <sstream>
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

    bool
    hasMemoryType(const PluginMetadata &metadata, nixl_mem_t memory_type) {
        return std::find(metadata.memory_types.begin(), metadata.memory_types.end(), memory_type) !=
            metadata.memory_types.end();
    }

    size_t
    countCommaSeparated(const std::string &value) {
        if (value.empty()) {
            return 0;
        }
        return static_cast<size_t>(std::count(value.begin(), value.end(), ',')) + 1;
    }

    std::vector<std::string>
    sortedParameterKeys(const nixl_b_params_t &parameters) {
        std::vector<std::string> keys;
        keys.reserve(parameters.size());
        for (const auto &[key, value] : parameters) {
            (void)value;
            keys.push_back(key);
        }
        std::sort(keys.begin(), keys.end());
        return keys;
    }

    bool
    validateRawOptions(const RawOptions &raw, std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        if (raw.operation != "READ" && raw.operation != "WRITE") {
            return fail("--operation must be read or write");
        }
        if (raw.threads < 1 || raw.iterations < 1 || raw.warmup_iterations < 0 ||
            raw.pipeline_depth < 1) {
            return fail(
                "threads, iterations, and pipeline depth must be positive; warmup may be zero");
        }
        if (raw.start_block_size == 0 || raw.max_block_size < raw.start_block_size) {
            return fail("block sizes must be positive and max must be at least start");
        }
        if (raw.start_batch_size == 0 || raw.max_batch_size < raw.start_batch_size) {
            return fail("batch sizes must be positive and max must be at least start");
        }
        return true;
    }

    bool
    validateFileOptions(const FileOptions &file, const RawOptions &raw, std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        if (!file.path.empty() && !file.filenames.empty()) {
            return fail("--path and --filenames are mutually exclusive");
        }
        if (file.num_files < 1) {
            return fail("--num-files must be at least 1");
        }
        if (!file.filenames.empty() &&
            (file.filenames.front() == ',' || file.filenames.back() == ',' ||
             file.filenames.find(",,") != std::string::npos)) {
            return fail("--filenames must not contain empty entries");
        }
        if (!file.filenames.empty() &&
            countCommaSeparated(file.filenames) != static_cast<size_t>(file.num_files)) {
            return fail("--filenames must contain exactly --num-files entries");
        }
        if (file.num_files > raw.threads || raw.threads % file.num_files != 0) {
            return fail("--num-files must divide --threads and cannot exceed it");
        }
        return true;
    }

    std::string
    formatSize(size_t bytes) {
        static constexpr const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
        double value = static_cast<double>(bytes);
        size_t unit = 0;
        while (value >= 1024.0 && unit + 1 < std::size(units)) {
            value /= 1024.0;
            ++unit;
        }
        std::ostringstream output;
        output << std::fixed << std::setprecision(value == static_cast<size_t>(value) ? 0 : 2)
               << value << ' ' << units[unit] << " (" << bytes << " bytes)";
        return output.str();
    }

} // namespace

void
printRawPosixPlan(const RawPosixRequest &request,
                  const PluginMetadata &metadata,
                  std::ostream &out) {
    out << "Resolved NIXLBench plan\n"
        << "  command: raw posix\n"
        << "  backend: " << metadata.name << "\n"
        << "  memory types: ";
    auto memory_types = metadata.memory_types;
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
        << request.raw.max_batch_size << "\n    iterations: " << request.raw.iterations
        << " (warmup " << request.raw.warmup_iterations << ")"
        << "\n    threads: " << request.raw.threads
        << "\n    pipeline depth: " << request.raw.pipeline_depth
        << "\n    consistency check: " << (request.raw.check_consistency ? "enabled" : "disabled");
    if (request.has_file_options) {
        out << "\n  file-resource options:\n"
            << "    path: "
            << (request.file.path.empty() ? "<current working directory>" : request.file.path)
            << "\n    filenames: "
            << (request.file.filenames.empty() ? "<automatic>" : request.file.filenames)
            << "\n    files: " << request.file.num_files
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

std::optional<size_t>
parseHumanSize(const std::string &input, std::string &error) {
    std::string value;
    value.reserve(input.size());
    for (unsigned char ch : input) {
        if (!std::isspace(ch)) {
            value.push_back(static_cast<char>(std::toupper(ch)));
        }
    }
    if (value.empty()) {
        error = "size cannot be empty";
        return std::nullopt;
    }

    size_t digit_count = 0;
    while (digit_count < value.size() &&
           std::isdigit(static_cast<unsigned char>(value[digit_count]))) {
        ++digit_count;
    }
    if (digit_count == 0) {
        error = "size must begin with a positive integer";
        return std::nullopt;
    }

    uint64_t number = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + digit_count, number);
    if (parsed.ec != std::errc() || number == 0) {
        error = "size must be a positive integer";
        return std::nullopt;
    }

    const std::string suffix = value.substr(digit_count);
    uint64_t multiplier = 0;
    if (suffix.empty() || suffix == "B") {
        multiplier = 1;
    } else if (suffix == "K" || suffix == "KB" || suffix == "KIB") {
        multiplier = 1024ULL;
    } else if (suffix == "M" || suffix == "MB" || suffix == "MIB") {
        multiplier = 1024ULL * 1024;
    } else if (suffix == "G" || suffix == "GB" || suffix == "GIB") {
        multiplier = 1024ULL * 1024 * 1024;
    } else if (suffix == "T" || suffix == "TB" || suffix == "TIB") {
        multiplier = 1024ULL * 1024 * 1024 * 1024;
    } else {
        error = "unsupported size suffix '" + suffix +
            "' (use B, KiB, MiB, GiB, or TiB; KB/MB/GB/TB are binary aliases)";
        return std::nullopt;
    }
    if (number > std::numeric_limits<size_t>::max() / multiplier) {
        error = "size is too large for this platform";
        return std::nullopt;
    }
    return static_cast<size_t>(number * multiplier);
}

std::optional<PluginMetadata>
discoverPluginMetadata(const std::string &name, std::string &error) {
    nixlAgent agent("nixlbench-cli", nixlAgentConfig{});
    std::vector<nixl_backend_t> plugins;
    const auto list_status = agent.getAvailPlugins(plugins);
    if (list_status != NIXL_SUCCESS) {
        error = "failed to discover NIXL plugins: " + nixlEnumStrings::statusStr(list_status);
        return std::nullopt;
    }
    if (std::find(plugins.begin(), plugins.end(), name) == plugins.end()) {
        error = name + " plugin is not installed or not visible in the NIXL plugin path";
        return std::nullopt;
    }

    PluginMetadata metadata;
    metadata.name = name;
    const auto status = agent.getPluginParams(name, metadata.memory_types, metadata.parameters);
    if (status != NIXL_SUCCESS) {
        error =
            "failed to query " + name + " plugin metadata: " + nixlEnumStrings::statusStr(status);
        return std::nullopt;
    }
    return metadata;
}

int
parseRawPosixCommand(int argc,
                     char *argv[],
                     const PluginMetadata &metadata,
                     RawPosixRequest &request,
                     bool &help_requested,
                     std::ostream &out,
                     std::ostream &err) {
    help_requested = false;
    request.plugin_parameters = metadata.parameters;
    request.has_file_options = hasMemoryType(metadata, FILE_SEG);

    std::string total_buffer_size = std::to_string(request.raw.total_buffer_size);
    std::string start_block_size = std::to_string(request.raw.start_block_size);
    std::string max_block_size = std::to_string(request.raw.max_block_size);
    std::vector<std::pair<std::string, std::string>> plugin_parameter_overrides;

    CLI::App app("NIXL data-transfer benchmark");
    app.require_subcommand(1);
    auto *raw = app.add_subcommand("raw", "Configure a low-level benchmark explicitly");
    raw->require_subcommand(1);
    auto *posix = raw->add_subcommand("posix", "Run the installed POSIX storage backend");
    posix->fallthrough();
    posix->footer("Raw benchmark options are documented by 'nixlbench raw --help' and may be used "
                  "before or after the posix subcommand.");

    raw->add_option("--operation", request.raw.operation, "Transfer direction: read or write")
        ->check(CLI::IsMember({"read", "write"}, CLI::ignore_case))
        ->group("Raw benchmark options");
    raw->add_option("--total-buffer-size",
                    total_buffer_size,
                    "Total buffer size using binary units (for example 4MiB or 4MB)")
        ->default_str(formatSize(request.raw.total_buffer_size))
        ->group("Raw benchmark options");
    raw->add_option("--start-block-size", start_block_size, "First block size in the sweep")
        ->default_str(formatSize(request.raw.start_block_size))
        ->group("Raw benchmark options");
    raw->add_option("--max-block-size", max_block_size, "Last block size in the sweep")
        ->default_str(formatSize(request.raw.max_block_size))
        ->group("Raw benchmark options");
    raw->add_option(
           "--start-batch-size", request.raw.start_batch_size, "First batch size in the sweep")
        ->group("Raw benchmark options");
    raw->add_option("--max-batch-size", request.raw.max_batch_size, "Last batch size in the sweep")
        ->group("Raw benchmark options");
    raw->add_option("--iterations", request.raw.iterations, "Timed iterations")
        ->group("Raw benchmark options");
    raw->add_option("--warmup-iterations", request.raw.warmup_iterations, "Warmup iterations")
        ->group("Raw benchmark options");
    raw->add_option("--threads", request.raw.threads, "Benchmark worker threads")
        ->group("Raw benchmark options");
    raw->add_option("--pipeline-depth", request.raw.pipeline_depth, "Transfer requests in flight")
        ->group("Raw benchmark options");
    raw->add_flag(
           "--check-consistency", request.raw.check_consistency, "Validate transferred bytes")
        ->group("Raw benchmark options");
    raw->add_flag("--dry-run", request.raw.dry_run, "Print the resolved plan without executing")
        ->group("Raw benchmark options");

    if (request.has_file_options) {
        posix->add_option("--path", request.file.path, "Directory for automatically named files")
            ->group("FILE_SEG resource options");
        posix
            ->add_option(
                "--filenames", request.file.filenames, "Comma-separated explicit file names")
            ->group("FILE_SEG resource options");
        posix->add_option("--num-files", request.file.num_files, "Number of backing files")
            ->group("FILE_SEG resource options");
        posix->add_flag("--direct", request.file.direct, "Use direct file opening")
            ->group("FILE_SEG resource options");
    }
    if (!metadata.parameters.empty()) {
        posix
            ->add_option("--plugin-param",
                         plugin_parameter_overrides,
                         "Override an advertised plugin parameter")
            ->check(CLI::IsMember(sortedParameterKeys(metadata.parameters)).application_index(0))
            ->type_name("KEY VALUE")
            ->group("Plugin initialization parameters");
    }

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

    for (const auto &[key, value] : plugin_parameter_overrides) {
        request.plugin_parameters[key] = value;
    }

    request.raw.operation = upper(request.raw.operation);
    std::vector<std::pair<const std::string *, size_t *>> sizes = {
        {&total_buffer_size, &request.raw.total_buffer_size},
        {&start_block_size, &request.raw.start_block_size},
        {&max_block_size, &request.raw.max_block_size},
    };
    for (const auto &[text, destination] : sizes) {
        std::string size_error;
        const auto parsed = parseHumanSize(*text, size_error);
        if (!parsed) {
            err << "Error: invalid size '" << *text << "': " << size_error << '\n';
            return 2;
        }
        *destination = *parsed;
    }

    if (!validateRawOptions(request.raw, err)) {
        return 2;
    }
    if (request.has_file_options && !validateFileOptions(request.file, request.raw, err)) {
        return 2;
    }
    return 0;
}

std::vector<std::string>
benchmarkFileArguments(const RawPosixRequest &request, const std::string &program_name) {
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    std::vector<std::string> arguments = {
        program_name,
        // Fixed values select the existing NIXL/POSIX runner.
        "--worker_type=nixl",
        "--backend=POSIX",
        "--initiator_seg_type=DRAM",
        "--target_seg_type=DRAM",
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
        arguments.push_back("--num_files=" + std::to_string(request.file.num_files));
        arguments.push_back("--storage_enable_direct=" + std::string(boolean(request.file.direct)));
    }
    return arguments;
}

RawCommandResult
prepareRawCommand(int argc, char *argv[], std::ostream &out, std::ostream &err) {
    std::string discovery_error;
    const auto metadata = discoverPluginMetadata("POSIX", discovery_error);
    if (!metadata) {
        err << "Error: " << discovery_error << '\n';
        return {1, false};
    }

    RawPosixRequest request;
    bool help_requested = false;
    const int parse_status =
        parseRawPosixCommand(argc, argv, *metadata, request, help_requested, out, err);
    if (parse_status != 0 || help_requested) {
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
    if (xferBenchConfig::parseConfig(legacy_argc, legacy_argv) != 0) {
        return {1, false};
    }

    printRawPosixPlan(request, *metadata, out);
    return {0, !request.raw.dry_run, std::move(request.plugin_parameters)};
}

} // namespace nixlbench
