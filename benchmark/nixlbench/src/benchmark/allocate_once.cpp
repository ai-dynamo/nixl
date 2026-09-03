/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/allocate_once.h"

#include "utils/utils.h"
#include "benchmark/allocate_once_worker.h"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <ostream>
#include <set>
#include <string_view>
#include <sys/stat.h>
#include <unordered_set>
#include <unistd.h>
#include <utility>

namespace nixlbench {
namespace {

    constexpr size_t initialization_chunk = 1024 * 1024;
    constexpr mode_t managed_file_permissions = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

    class fileHandle {
    public:
        explicit fileHandle(int fd) : fd_(fd) {}

        ~fileHandle() {
            if (fd_ >= 0) {
                ::close(fd_);
            }
        }

        fileHandle(const fileHandle &) = delete;
        fileHandle &
        operator=(const fileHandle &) = delete;

    private:
        int fd_;
    };

    bool
    initializeManagedFile(int fd, uint64_t size, size_t alignment, bool direct, std::ostream &err) {
        if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
            err << "Failed to resize managed file: " << strerror(errno) << '\n';
            return false;
        }

        const size_t chunk_size = std::min<size_t>(initialization_chunk, size);
        void *storage = nullptr;
        const int allocation_status = posix_memalign(&storage, alignment, chunk_size);
        if (allocation_status != 0 || storage == nullptr) {
            err << "Failed to allocate managed-file initialization buffer: "
                << strerror(allocation_status) << '\n';
            return false;
        }
        std::unique_ptr<void, decltype(&free)> chunk(storage, &free);
        memset(chunk.get(), XFERBENCH_TARGET_BUFFER_ELEMENT, chunk_size);

        uint64_t offset = 0;
        while (offset < size) {
            const size_t count = std::min<uint64_t>(chunk_size, size - offset);
            size_t written_total = 0;
            while (written_total < count) {
                const ssize_t written = pwrite(fd,
                                               static_cast<char *>(chunk.get()) + written_total,
                                               count - written_total,
                                               static_cast<off_t>(offset + written_total));
                if (written <= 0 ||
                    (direct && static_cast<size_t>(written) != count - written_total)) {
                    err << "Failed to initialize managed file at offset " << offset + written_total
                        << ": " << (written < 0 ? strerror(errno) : "short write") << '\n';
                    return false;
                }
                written_total += static_cast<size_t>(written);
            }
            offset += count;
        }
        return true;
    }

} // namespace

std::optional<std::vector<threadFileRegion>>
allocateOnceThreadRegions(const allocateOnceRequest &request, std::string &error) {
    if (request.files.empty() || request.common.threads < 1 || request.common.blockSize == 0) {
        error = "files, threads, and block size must be configured";
        return std::nullopt;
    }
    const uint64_t slots_per_file = request.fileSize / request.common.blockSize;
    std::vector<size_t> threads_per_file(request.files.size(), 0);
    for (int thread = 0; thread < request.common.threads; ++thread) {
        ++threads_per_file[static_cast<size_t>(thread) % request.files.size()];
    }

    std::vector<size_t> file_thread_index(request.files.size(), 0);
    std::vector<threadFileRegion> regions;
    regions.reserve(static_cast<size_t>(request.common.threads));
    for (int thread = 0; thread < request.common.threads; ++thread) {
        const size_t file_index = static_cast<size_t>(thread) % request.files.size();
        const size_t ordinal = file_thread_index[file_index]++;
        const size_t count = threads_per_file[file_index];
        const uint64_t base_slot_count = slots_per_file / count;
        const uint64_t extra_slots = slots_per_file % count;
        const uint64_t first = base_slot_count * ordinal + std::min<uint64_t>(ordinal, extra_slots);
        const uint64_t slot_count = base_slot_count + (ordinal < extra_slots ? 1 : 0);
        if (slot_count < request.common.batchSize) {
            error = "each thread partition must contain at least --batch-size blocks";
            return std::nullopt;
        }
        regions.push_back({file_index, first, slot_count});
    }
    return regions;
}

offsetSequence::offsetSequence(threadFileRegion region,
                               size_t batch_size,
                               offset_mode_t offset_mode,
                               uint64_t seed)
    : region_(region),
      batchSize_(batch_size),
      randomize_(offset_mode == offset_mode_t::RANDOM),
      nextSequentialSlot_(region.firstSlot),
      random_(seed) {}

std::vector<uint64_t>
offsetSequence::next() {
    std::vector<uint64_t> result;
    result.reserve(batchSize_);
    if (!randomize_) {
        for (size_t index = 0; index < batchSize_; ++index) {
            result.push_back(nextSequentialSlot_);
            const uint64_t relative = nextSequentialSlot_ - region_.firstSlot;
            nextSequentialSlot_ = region_.firstSlot + (relative + 1) % region_.slotCount;
        }
        return result;
    }

    std::unordered_set<uint64_t> selected;
    const uint64_t start = region_.slotCount - batchSize_;
    for (uint64_t candidate = start; candidate < region_.slotCount; ++candidate) {
        std::uniform_int_distribution<uint64_t> distribution(0, candidate);
        const uint64_t random_slot = distribution(random_);
        const uint64_t chosen =
            selected.find(random_slot) == selected.end() ? random_slot : candidate;
        selected.insert(chosen);
        result.push_back(region_.firstSlot + chosen);
    }
    return result;
}

bool
prepareAllocateOnceFiles(const allocateOnceRequest &request, std::ostream &err) {
    const long page_size_value = sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0) {
        err << "Could not determine system page size\n";
        return false;
    }
    const size_t alignment = static_cast<size_t>(page_size_value);
    std::set<std::pair<dev_t, ino_t>> identities;

    for (const auto &path : request.files) {
        int flags =
            (request.managedFiles || request.common.operation == NIXL_WRITE ? O_RDWR : O_RDONLY) |
            O_LARGEFILE;
        if (request.managedFiles) {
            flags |= O_CREAT | O_NOFOLLOW;
        }
        if (request.direct) {
            flags |= O_DIRECT;
        }

        const int fd = open(path.c_str(), flags, managed_file_permissions);
        if (fd < 0) {
            err << "Failed to open " << path << ": " << strerror(errno) << '\n';
            return false;
        }
        fileHandle file(fd);

        struct stat info{};
        if (fstat(fd, &info) != 0) {
            err << "Failed to inspect " << path << ": " << strerror(errno) << '\n';
            return false;
        }
        if (request.managedFiles && !S_ISREG(info.st_mode)) {
            err << "Managed path must be a regular file: " << path << '\n';
            return false;
        }
        if (!identities.emplace(info.st_dev, info.st_ino).second) {
            err << "Backing files must refer to distinct files: " << path << '\n';
            return false;
        }
        if (request.managedFiles) {
            if ((static_cast<uint64_t>(info.st_size) != request.fileSize ||
                 request.common.checkConsistency) &&
                !initializeManagedFile(fd, request.fileSize, alignment, request.direct, err)) {
                return false;
            }
        } else if (static_cast<uint64_t>(info.st_size) < request.fileSize) {
            err << "Explicit file " << path << " is smaller than --file-size\n";
            return false;
        }
    }
    return true;
}

namespace {

    std::string
    lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return value;
    }

    bool
    multiplyFits(size_t left, size_t right) {
        return left == 0 || right <= std::numeric_limits<size_t>::max() / left;
    }

    bool
    validateRequest(allocateOnceRequest &request,
                    const fileOptions &file,
                    bool seed_provided,
                    std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        if (request.fileSize == 0) {
            return fail("file size must be positive");
        }
        if (request.fileSize > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            return fail("file size exceeds the platform file-offset limit");
        }
        std::string file_error;
        if (!validateFileOptions(file, file_error)) {
            return fail(file_error);
        }
        if (file.numFiles > request.common.threads) {
            return fail("--num-files cannot exceed --threads");
        }
        if (request.common.checkConsistency && !file.filenames.empty()) {
            return fail("--check-consistency requires NIXLBench-managed files");
        }
        if (request.offsetMode == offset_mode_t::SEQUENTIAL && seed_provided) {
            return fail("--seed requires --offset-mode=random");
        }
        if (request.fileSize % request.common.blockSize != 0) {
            return fail("--file-size must be an exact multiple of --block-size");
        }

        const size_t slots_per_file = request.fileSize / request.common.blockSize;
        const size_t thread_count = static_cast<size_t>(request.common.threads);
        const size_t file_count = static_cast<size_t>(file.numFiles);
        const size_t max_threads_per_file = (thread_count - 1) / file_count + 1;
        if (!multiplyFits(max_threads_per_file, request.common.batchSize) ||
            slots_per_file < max_threads_per_file * request.common.batchSize) {
            return fail(
                "each thread needs at least --batch-size block slots in its file partition");
        }
        if (!multiplyFits(request.fileSize, static_cast<size_t>(file.numFiles))) {
            return fail("total dataset size is too large for this platform");
        }
        if (!multiplyFits(request.common.blockSize, request.common.batchSize) ||
            !multiplyFits(request.common.blockSize * request.common.batchSize,
                          static_cast<size_t>(request.common.threads))) {
            return fail("working-memory size is too large for this platform");
        }

        if (file.direct) {
            const long page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0) {
                return fail("could not determine the system page size for direct I/O");
            }
            const size_t alignment = static_cast<size_t>(page_size);
            if (request.fileSize % alignment != 0 || request.common.blockSize % alignment != 0) {
                return fail(
                    "--direct requires file and block sizes aligned to the system page size");
            }
        }

        auto files = allocateOnceFileNames(file, file_error);
        if (!files) {
            return fail(file_error);
        }
        request.files = std::move(*files);
        request.managedFiles = file.filenames.empty();
        request.direct = file.direct;
        return true;
    }

} // namespace

struct allocateOnceScenario::implementation {
    size_t fileSize = 0;
    std::string offsetMode = "random";
    uint64_t seed = 0;
    CLI::Option *seedOption = nullptr;
    allocateOnceRequest request;
};

allocateOnceScenario::allocateOnceScenario()
    : benchmarkScenario("allocate-once",
                        "Reuse fixed registered files while transferring at changing offsets",
                        supportsAllocateOnce,
                        true),
      implementation_(std::make_unique<implementation>()) {}

allocateOnceScenario::~allocateOnceScenario() = default;

bool
supportsAllocateOnce(const pluginMetadata &metadata) {
    return hasMemoryType(metadata, FILE_SEG) &&
        (hasMemoryType(metadata, DRAM_SEG) || hasMemoryType(metadata, VRAM_SEG));
}

std::optional<std::vector<std::filesystem::path>>
allocateOnceFileNames(const fileOptions &file, std::string &error) {
    std::vector<std::filesystem::path> names;
    if (!file.filenames.empty()) {
        for (const auto &name : splitFileNames(file.filenames)) {
            names.emplace_back(name);
        }
        return names;
    }

    std::filesystem::path directory(file.path);
    if (file.path.empty()) {
        std::error_code path_error;
        directory = std::filesystem::current_path(path_error);
        if (path_error) {
            error = "could not determine the current working directory: " + path_error.message();
            return std::nullopt;
        }
    }
    names.reserve(static_cast<size_t>(file.numFiles));
    for (int index = 0; index < file.numFiles; ++index) {
        names.push_back(directory / ("nixlbench_allocate_once_" + std::to_string(index) + ".dat"));
    }
    return names;
}

size_t
allocateOnceWorkingMemory(const allocateOnceRequest &request) {
    return request.common.blockSize * request.common.batchSize *
        static_cast<size_t>(request.common.threads);
}

uint64_t
resolveOffsetSeed(uint64_t configured_seed) {
    if (configured_seed != 0) {
        return configured_seed;
    }

    std::random_device random_device;
    const uint64_t seed =
        (static_cast<uint64_t>(random_device()) << 32) | static_cast<uint64_t>(random_device());
    return seed == 0 ? 1 : seed;
}

void
allocateOnceScenario::addScenarioOptions(CLI::App &command) {
    command
        .add_option("--file-size", implementation_->fileSize, "Logical size of each backing file")
        ->transform(binarySizeTransform())
        ->check(CLI::PositiveNumber)
        ->required()
        ->group("Allocate-once options");
    command
        .add_option(
            "--offset-mode", implementation_->offsetMode, "Block selection: sequential or random")
        ->check(CLI::IsMember({"sequential", "random"}, CLI::ignore_case))
        ->group("Allocate-once options");
    implementation_->seedOption = command
                                      .add_option("--seed",
                                                  implementation_->seed,
                                                  "Random-offset seed; zero selects a random seed")
                                      ->group("Allocate-once options");
}

int
allocateOnceScenario::finalizeScenario(std::ostream &err) {
    auto &request = implementation_->request;
    request.common = commonConfig();

    request.fileSize = implementation_->fileSize;
    request.offsetMode = lower(implementation_->offsetMode) == "random" ? offset_mode_t::RANDOM :
                                                                          offset_mode_t::SEQUENTIAL;
    request.seed =
        request.offsetMode == offset_mode_t::RANDOM ? resolveOffsetSeed(implementation_->seed) : 0;

    const bool seed_provided = implementation_->seedOption->count() != 0;
    implementation_->seedOption = nullptr;
    if (!validateRequest(request, commonFileOptions(), seed_provided, err)) {
        return inval_args_exit_code;
    }
    return EXIT_SUCCESS;
}

void
allocateOnceScenario::printScenarioPlan(std::ostream &out) const {
    const auto &request = implementation_->request;
    const size_t total_dataset = request.fileSize * request.files.size();
    out << "\n  dataset policy: "
        << (request.managedFiles ?
                "managed, initialize if needed, reinitialize for consistency checks, keep after "
                "run" :
                "explicit files, reuse only")
        << "\n  files: " << request.files.size() << "\n  resolved file paths:";
    for (const auto &file : request.files) {
        out << "\n    " << file;
    }
    out << "\n  size per file: " << formatSize(request.fileSize)
        << "\n  total dataset: " << formatSize(total_dataset)
        << "\n  working memory: " << formatSize(allocateOnceWorkingMemory(request))
        << "\n  offset mode: "
        << (request.offsetMode == offset_mode_t::RANDOM ? "random" : "sequential");
    if (request.offsetMode == offset_mode_t::RANDOM) {
        out << ", seed " << request.seed;
    }
    out << "\n  direct I/O: " << (request.direct ? "enabled" : "disabled")
        << "\n  lifecycle: open, allocate, and register once; release each transfer request"
        << "\n  execution: shared NIXLBench worker facilities"
        << "\n  timing: transfer-phase request preparation, post, latency, and throughput; "
           "file setup and final cleanup excluded";
}

void
allocateOnceScenario::printDryRunPlan(std::ostream &out) const {
    out << "Dry run: no backing file was opened and no transfer buffer, registration, or "
           "transfer request was created.\n";
}

bool
allocateOnceScenario::prepare(std::ostream &err) const {
    return prepareAllocateOnceFiles(implementation_->request, err);
}

void
allocateOnceScenario::configureLegacyWorker(legacyWorkerConfig &config) const {
    const auto &request = implementation_->request;
    config.workingMemory = allocateOnceWorkingMemory(request);
    config.targetMemory = FILE_SEG;
    config.recreateTransferRequest = true;
    config.fileNames.reserve(request.files.size());
    for (const auto &file : request.files) {
        config.fileNames.push_back(file.string());
    }
    config.storageDirect = request.direct;
}

std::unique_ptr<xferBenchWorker>
allocateOnceScenario::createWorker(const std::vector<std::string> &devices) const {
    return std::make_unique<xferBenchNixlAllocateOnceWorker>(devices, implementation_->request);
}

} // namespace nixlbench
