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

namespace nixlbench {
namespace {

    // Keep these limits synchronized with the POSIX I/O queue normalization.
    constexpr int POSIX_MIN_IO_POOL_SIZE = 64;
    constexpr int POSIX_MAX_IO_POOL_SIZE = 65536;
    constexpr int POSIX_MIN_KERNEL_QUEUE_SIZE = 16;
    constexpr int POSIX_MAX_KERNEL_QUEUE_SIZE = 1024;

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

    std::vector<std::string>
    availableApis(const PluginMetadata &metadata) {
        std::vector<std::string> result;
        if (metadata.parameters.contains("use_aio")) {
            result.emplace_back("aio");
        }
        if (metadata.parameters.contains("use_uring")) {
            result.emplace_back("uring");
        }
        if (metadata.parameters.contains("use_posix_aio")) {
            result.emplace_back("posix-aio");
        }
        return result;
    }

    std::string
    defaultApi(const PluginMetadata &metadata, const std::vector<std::string> &apis) {
        const auto enabled = [&](const char *name) {
            const auto it = metadata.parameters.find(name);
            return it != metadata.parameters.end() && upper(it->second) == "TRUE";
        };
        if (enabled("use_aio")) {
            return "aio";
        }
        if (enabled("use_uring")) {
            return "uring";
        }
        if (enabled("use_posix_aio")) {
            return "posix-aio";
        }
        return apis.empty() ? std::string() : apis.front();
    }

    std::optional<int>
    metadataIntegerDefault(const PluginMetadata &metadata,
                           const char *name,
                           int fallback,
                           std::ostream &err) {
        const auto it = metadata.parameters.find(name);
        if (it == metadata.parameters.end()) {
            return fallback;
        }
        int value = 0;
        const auto result =
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), value);
        if (result.ec != std::errc() || result.ptr != it->second.data() + it->second.size() ||
            value <= 0) {
            err << "Error: POSIX plugin metadata parameter '" << name
                << "' must be a positive integer\n";
            return std::nullopt;
        }
        return value;
    }

    size_t
    countCommaSeparated(const std::string &value) {
        if (value.empty()) {
            return 0;
        }
        return static_cast<size_t>(std::count(value.begin(), value.end(), ',')) + 1;
    }

    bool
    validateRequest(const RawPosixRequest &request,
                    const std::vector<std::string> &apis,
                    std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        if (request.raw.operation != "READ" && request.raw.operation != "WRITE") {
            return fail("--operation must be read or write");
        }
        if (std::find(apis.begin(), apis.end(), request.posix.api) == apis.end()) {
            return fail("--api is not available in the installed POSIX plugin");
        }
        if (!request.posix.path.empty() && !request.posix.filenames.empty()) {
            return fail("--path and --filenames are mutually exclusive");
        }
        if (request.posix.num_files < 1) {
            return fail("--num-files must be at least 1");
        }
        if (!request.posix.filenames.empty() &&
            countCommaSeparated(request.posix.filenames) !=
                static_cast<size_t>(request.posix.num_files)) {
            return fail("--filenames must contain exactly --num-files entries");
        }
        if (request.raw.threads < 1 || request.raw.iterations < 1 ||
            request.raw.warmup_iterations < 0 || request.raw.pipeline_depth < 1) {
            return fail(
                "threads, iterations, and pipeline depth must be positive; warmup may be zero");
        }
        if (request.posix.num_files > request.raw.threads ||
            request.raw.threads % request.posix.num_files != 0) {
            return fail("--num-files must divide --threads and cannot exceed it");
        }
        if (request.raw.start_block_size == 0 ||
            request.raw.max_block_size < request.raw.start_block_size) {
            return fail("block sizes must be positive and max must be at least start");
        }
        if (request.raw.start_batch_size == 0 ||
            request.raw.max_batch_size < request.raw.start_batch_size) {
            return fail("batch sizes must be positive and max must be at least start");
        }
        if (request.posix.io_pool_size < POSIX_MIN_IO_POOL_SIZE ||
            request.posix.io_pool_size > POSIX_MAX_IO_POOL_SIZE) {
            return fail("--io-pool-size must be between 64 and 65536");
        }
        if (request.posix.kernel_queue_size < POSIX_MIN_KERNEL_QUEUE_SIZE ||
            request.posix.kernel_queue_size > POSIX_MAX_KERNEL_QUEUE_SIZE) {
            return fail("--kernel-queue-size must be between 16 and 1024");
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

    void
    printPlan(const RawPosixRequest &request, const PluginMetadata &metadata, std::ostream &out) {
        out << "Resolved NIXLBench plan\n"
            << "  command: raw posix\n"
            << "  backend: " << metadata.name << "\n"
            << "  memory types: ";
        for (size_t i = 0; i < metadata.memory_types.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << nixlEnumStrings::memTypeStr(metadata.memory_types[i]);
        }
        out << "\n  operation: " << xferBenchConfig::op_type << "\n  path: "
            << (xferBenchConfig::filepath.empty() ? "<current working directory>" :
                                                    xferBenchConfig::filepath)
            << "\n  filenames: "
            << (xferBenchConfig::filenames.empty() ? "<automatic>" : xferBenchConfig::filenames)
            << "\n  files: " << xferBenchConfig::num_files
            << "\n  total buffer: " << formatSize(xferBenchConfig::total_buffer_size)
            << "\n  block sizes: " << formatSize(xferBenchConfig::start_block_size) << " .. "
            << formatSize(xferBenchConfig::max_block_size)
            << "\n  batch sizes: " << xferBenchConfig::start_batch_size << " .. "
            << xferBenchConfig::max_batch_size << "\n  iterations: " << xferBenchConfig::num_iter
            << " (warmup " << xferBenchConfig::warmup_iter << ")"
            << "\n  threads: " << xferBenchConfig::num_threads
            << "\n  pipeline depth: " << xferBenchConfig::pipeline_depth
            << "\n  consistency check: "
            << (xferBenchConfig::check_consistency ? "enabled" : "disabled") << "\n  direct I/O: "
            << (xferBenchConfig::storage_enable_direct ? "enabled" : "disabled")
            << "\n  POSIX API: " << xferBenchConfig::posix_api_type
            << "\n  POSIX I/O pool size: " << xferBenchConfig::posix_ios_pool_size
            << "\n  POSIX kernel queue size: " << xferBenchConfig::posix_kernel_queue_size << '\n';
        if (request.raw.dry_run) {
            out << "Dry run: no worker was created and no allocation or transfer was attempted.\n";
        }
    }

} // namespace

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
    if (metadata.name != "POSIX" || !hasMemoryType(metadata, DRAM_SEG) ||
        !hasMemoryType(metadata, FILE_SEG)) {
        err << "Error: POSIX metadata must advertise DRAM and FILE memory types\n";
        return 2;
    }
    const auto apis = availableApis(metadata);
    if (apis.empty()) {
        err << "Error: the installed POSIX plugin advertises no usable I/O API\n";
        return 2;
    }
    request.posix.api = defaultApi(metadata, apis);
    const auto io_pool_size =
        metadataIntegerDefault(metadata, "ios_pool_size", request.posix.io_pool_size, err);
    const auto kernel_queue_size =
        metadataIntegerDefault(metadata, "kernel_queue_size", request.posix.kernel_queue_size, err);
    if (!io_pool_size || !kernel_queue_size) {
        return 2;
    }
    request.posix.io_pool_size = *io_pool_size;
    request.posix.kernel_queue_size = *kernel_queue_size;

    std::string total_buffer_size = std::to_string(request.raw.total_buffer_size);
    std::string start_block_size = std::to_string(request.raw.start_block_size);
    std::string max_block_size = std::to_string(request.raw.max_block_size);

    CLI::App app("NIXL data-transfer benchmark");
    app.require_subcommand(1);
    auto *raw = app.add_subcommand("raw", "Configure a low-level benchmark explicitly");
    raw->require_subcommand(1);
    auto *posix = raw->add_subcommand("posix", "Run the installed POSIX storage backend");
    posix->fallthrough();
    posix->footer("Raw benchmark options are documented by 'nixlbench raw --help' and may be used "
                  "before or after the posix subcommand.");

    raw->add_option("--operation", request.raw.operation, "Transfer direction: read or write")
        ->check(CLI::IsMember({"read", "write"}, CLI::ignore_case));
    raw->add_option("--total-buffer-size",
                    total_buffer_size,
                    "Total buffer size using binary units (for example 4MiB or 4MB)");
    raw->add_option("--start-block-size", start_block_size, "First block size in the sweep");
    raw->add_option("--max-block-size", max_block_size, "Last block size in the sweep");
    raw->add_option(
        "--start-batch-size", request.raw.start_batch_size, "First batch size in the sweep");
    raw->add_option("--max-batch-size", request.raw.max_batch_size, "Last batch size in the sweep");
    raw->add_option("--iterations", request.raw.iterations, "Timed iterations");
    raw->add_option("--warmup-iterations", request.raw.warmup_iterations, "Warmup iterations");
    raw->add_option("--threads", request.raw.threads, "Benchmark worker threads");
    raw->add_option("--pipeline-depth", request.raw.pipeline_depth, "Transfer requests in flight");
    raw->add_flag(
        "--check-consistency", request.raw.check_consistency, "Validate transferred bytes");
    raw->add_flag("--dry-run", request.raw.dry_run, "Print the resolved plan without executing");

    posix->add_option("--path", request.posix.path, "Directory for automatically named files");
    posix->add_option(
        "--filenames", request.posix.filenames, "Comma-separated explicit file names");
    posix->add_option("--num-files", request.posix.num_files, "Number of backing files");
    posix->add_flag("--direct", request.posix.direct, "Use direct I/O");
    auto *api_option =
        posix->add_option("--api", request.posix.api, "POSIX I/O API exposed by plugin metadata");
    api_option->check(CLI::IsMember(apis, CLI::ignore_case));
    api_option->default_str(request.posix.api);
    if (metadata.parameters.contains("ios_pool_size")) {
        posix
            ->add_option(
                "--io-pool-size", request.posix.io_pool_size, "POSIX reusable I/O entry count")
            ->check(CLI::Range(POSIX_MIN_IO_POOL_SIZE, POSIX_MAX_IO_POOL_SIZE))
            ->default_str(std::to_string(request.posix.io_pool_size));
    }
    if (metadata.parameters.contains("kernel_queue_size")) {
        posix
            ->add_option(
                "--kernel-queue-size", request.posix.kernel_queue_size, "POSIX kernel queue depth")
            ->check(CLI::Range(POSIX_MIN_KERNEL_QUEUE_SIZE, POSIX_MAX_KERNEL_QUEUE_SIZE))
            ->default_str(std::to_string(request.posix.kernel_queue_size));
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

    request.raw.operation = upper(request.raw.operation);
    request.posix.api = upper(request.posix.api);
    if (request.posix.api == "POSIX-AIO") {
        request.posix.api = "POSIXAIO";
    }
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

    std::vector<std::string> normalized_apis;
    normalized_apis.reserve(apis.size());
    for (auto api : apis) {
        api = upper(api);
        normalized_apis.push_back(api == "POSIX-AIO" ? "POSIXAIO" : api);
    }
    return validateRequest(request, normalized_apis, err) ? 0 : 2;
}

std::vector<std::string>
legacyArguments(const RawPosixRequest &request, const std::string &program_name) {
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    return {program_name,
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
            "--pipeline_depth=" + std::to_string(request.raw.pipeline_depth),
            // POSIX backend configuration.
            "--filepath=" + request.posix.path,
            "--filenames=" + request.posix.filenames,
            "--num_files=" + std::to_string(request.posix.num_files),
            "--storage_enable_direct=" + std::string(boolean(request.posix.direct)),
            "--posix_api_type=" + request.posix.api,
            "--posix_ios_pool_size=" + std::to_string(request.posix.io_pool_size),
            "--posix_kernel_queue_size=" + std::to_string(request.posix.kernel_queue_size)};
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

    auto arguments = legacyArguments(request, argv[0]);
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

    printPlan(request, *metadata, out);
    return {0, !request.raw.dry_run};
}

} // namespace nixlbench
