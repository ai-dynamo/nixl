/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "configuration.h"
#include "hostname.h"
#include "nixl_log.h"
#include "scoped_fd.h"
#include "absl/base/no_destructor.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/container/flat_hash_map.h"
#include <charconv>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace {

// Structure to hold logging settings
struct LogLevelSettings {
    absl::LogSeverityAtLeast min_severity;
    int vlog_level;
};

// Default log level if nothing else is specified
constexpr std::string_view kDefaultLogLevel = "WARN";

// Names the file that log records are mirrored into. Unset disables the sink.
constexpr const char *log_file_env_var = "NIXL_LOG_FILE";

// Bounds that file. Unset or empty lets it grow without limit.
constexpr const char *log_file_size_env_var = "NIXL_LOG_FILE_SIZE";

// Makes a log file setup failure fatal. Unset or false keeps the default:
// report the failure and carry on without the file.
constexpr const char *log_file_error_is_fatal_env_var = "NIXL_LOG_FILE_ERROR_IS_FATAL";

// A fatal stack trace is the largest record this sink writes. Abseil keeps up
// to 64 frames; 16 KiB holds a typical symbolized trace.
constexpr std::uintmax_t min_log_file_size = 16 * 1024;

// Appended to the log file's name to hold the records rotated out of it.
constexpr const char *rotated_suffix = ".1";

/**
 * @brief Nanoseconds since the epoch, sampled once, for %t.
 */
[[nodiscard]] uint64_t
processRunMarker() {
    static const uint64_t marker = [] {
        struct timespec ts{};
        ::clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
    }();
    return marker;
}

/** @brief Expands %h, %p, %t and %%. nullopt if an escape is unknown. */
[[nodiscard]] std::optional<std::string>
expandLogPath(const std::string &pattern) {
    std::string expanded;

    for (size_t at = 0; at < pattern.size(); ++at) {
        if (pattern[at] != '%') {
            expanded += pattern[at];
            continue;
        }
        if (at + 1 == pattern.size()) {
            return std::nullopt;
        }

        switch (pattern[at + 1]) {
        case 'h':
            expanded += nixl::getHostname().value_or("unknown-host");
            ++at;
            break;
        case 'p':
            expanded += std::to_string(::getpid());
            ++at;
            break;
        case 't':
            expanded += std::to_string(processRunMarker());
            ++at;
            break;
        case '%':
            expanded += '%';
            ++at;
            break;
        default:
            return std::nullopt;
        }
    }
    return expanded;
}

/**
 * @brief Parses NIXL_LOG_FILE_SIZE, for example "64M". K, M, G are powers of 1024.
 * @return Bytes, 0 for no limit, or nullopt if @p text is not a size.
 */
[[nodiscard]] std::optional<std::uintmax_t>
parseLogFileSize(std::string_view text) {
    if (text.empty()) {
        return 0;
    }

    const char *begin = text.data();
    const char *end = begin + text.size();
    std::uintmax_t value = 0;
    const auto [suffix_begin, error] = std::from_chars(begin, end, value);
    if (error != std::errc{}) {
        return std::nullopt;
    }

    std::uintmax_t scale = 1;
    const std::string_view suffix(suffix_begin, static_cast<size_t>(end - suffix_begin));
    if (suffix == "K" || suffix == "k") {
        scale = 1024;
    } else if (suffix == "M" || suffix == "m") {
        scale = 1024 * 1024;
    } else if (suffix == "G" || suffix == "g") {
        scale = 1024 * 1024 * 1024;
    } else if (!suffix.empty()) {
        return std::nullopt;
    }

    if (value > std::numeric_limits<std::uintmax_t>::max() / scale) {
        return std::nullopt;
    }

    return value * scale;
}

/** @brief Whether NIXL_LOG_FILE_ERROR_IS_FATAL holds a recognised true value. */
[[nodiscard]] bool
logFileErrorIsFatal() {
    try {
        return nixl::config::getValueDefaulted(log_file_error_is_fatal_env_var, false);
    }
    catch (const std::exception &) {
        return false;
    }
}

/** @brief Reports a setup failure; fatal when logFileErrorIsFatal(), else an error. */
void
reportSetupFailure(const std::string &reason) {
    if (logFileErrorIsFatal()) {
        NIXL_FATAL << reason;
    }
    NIXL_ERROR << reason << ", continuing without a log file";
}

/** @brief Appends log records to a file, formatted exactly as on stderr. */
class fileLogSink final : public absl::LogSink {
public:
    /**
     * @brief Opens @p path for append; check isOpen() rather than catching.
     * @param path  Where to write, already expanded and made absolute.
     * @param limit Bytes before rotating, 0 for none. Counted from the current
     *              size, since appending inherits whatever the file holds.
     */
    fileLogSink(const std::string &path, std::uintmax_t limit)
        : path_(path),
          fd_(::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666)),
          limit_(limit) {
        if (!fd_.valid()) {
            return;
        }

        struct stat st;
        if (::fstat(fd_.get(), &st) == 0) {
            written_ = st.st_size;
        }
    }

    /** @brief False if the sink cannot write, and must not be registered. */
    [[nodiscard]] bool
    isOpen() const noexcept {
        return fd_.valid();
    }

    /**
     * @brief Writes one record directly to the file descriptor.
     * @param entry Borrowed; valid only for this call.
     */
    void
    Send(const absl::LogEntry &entry) override {
        const auto payload = entry.stacktrace().empty() ?
            entry.text_message_with_prefix_and_newline() :
            entry.stacktrace();
        writePayload(payload);
    }

    /** @brief Writes one payload. Takes mutex_. */
    void
    writePayload(std::string_view payload) {
        const std::lock_guard lock(mutex_);
        if (failed_) {
            return;
        }

        if (limit_ != 0) {
            // Report the first oversized record, then keep accepting others.
            if (payload.size() > limit_) {
                reportOversizedRecord();
                return;
            }

            // Rotate before adding a record that would exceed the limit.
            if (written_ > limit_ - payload.size()) {
                rotate();
                if (failed_) {
                    return;
                }
            }
        }

        size_t offset = 0;
        while (offset < payload.size()) {
            const ssize_t result =
                ::write(fd_.get(), payload.data() + offset, payload.size() - offset);
            if (result > 0) {
                offset += static_cast<size_t>(result);
            } else if (result < 0 && errno == EINTR) {
                continue;
            } else {
                reportFailure(result < 0 ? errno : EIO);
                return;
            }
        }
        written_ += payload.size();
    }

private:
    /**
     * @brief Moves the full file aside and starts a new one, keeping the newest
     *        records. Existing oversized files are preserved until a later
     *        rotation replaces them. A failed rotation stops the sink.
     *        Called with mutex_ held.
     */
    void
    rotate() {
        fd_.reset();

        std::error_code ec;
        std::filesystem::rename(path_, path_ + rotated_suffix, ec);
        if (ec) {
            reportRotateFailure(ec);
            return;
        }

        fd_ =
            nixl::scopedFd(::open(path_.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666));
        if (!fd_.valid()) {
            reportFailure(errno);
            return;
        }
        written_ = 0;
    }

    /**
     * @brief Reports the first write failure and stops using the file, going
     *        to stderr directly rather than through the machinery that failed.
     * @param reason errno from the failed operation, or 0. Called with mutex_ held.
     */
    void
    reportFailure(int reason) {
        failed_ = true;

        const std::string detail = (reason != 0) ? (": " + nixl_strerror(reason)) : "";
        std::fprintf(stderr,
                     "NIXL: could not write to %s '%s'%s; dropping further records\n",
                     log_file_env_var,
                     path_.c_str(),
                     detail.c_str());
    }

    /** @brief Reports the first oversized record. Called with mutex_ held. */
    void
    reportOversizedRecord() {
        if (reported_oversize_) {
            return;
        }
        reported_oversize_ = true;

        std::fprintf(stderr,
                     "NIXL: a record exceeded %s (%ju bytes) for '%s'; "
                     "omitting records larger than the limit\n",
                     log_file_size_env_var,
                     limit_,
                     path_.c_str());
    }

    /**
     * @brief Reports a failed rotation and stops using the file, leaving it as
     *        it is: those records are all there will be. Reports once, since
     *        failed_ stops Send() rotating again.
     * @param reason Why the rename failed. Called with mutex_ held.
     */
    void
    reportRotateFailure(const std::error_code &reason) {
        failed_ = true;

        std::fprintf(stderr,
                     "NIXL: could not rotate %s '%s' at its %s (%s); "
                     "dropping further records\n",
                     log_file_env_var,
                     path_.c_str(),
                     log_file_size_env_var,
                     reason.message().c_str());
    }

    std::mutex mutex_;
    std::string path_;
    nixl::scopedFd fd_;
    std::uintmax_t limit_ = 0;
    std::uintmax_t written_ = 0;
    bool failed_ = false;
    bool reported_oversize_ = false;
};

struct logFileState {
    std::mutex mutex;
    fileLogSink *sink = nullptr;
};

/**
 * @brief Process-lifetime state that remains usable from the destructor hook.
 */
[[nodiscard]] logFileState &
getLogFileState() {
    static absl::NoDestructor<logFileState> state;
    return *state;
}

/** @brief Applies NIXL_LOG_LEVEL and NIXL_LOG_FILE before any NIXL code logs. */
void
InitializeNixlLogging() __attribute__((constructor));

void
InitializeNixlLogging() {
    // Map from log level string to settings
    const absl::flat_hash_map<std::string_view, LogLevelSettings> kLogLevelMap = {
        {"TRACE", {absl::LogSeverityAtLeast::kInfo, 2}},
        {"DEBUG", {absl::LogSeverityAtLeast::kInfo, 1}},
        {"INFO", {absl::LogSeverityAtLeast::kInfo, 0}},
        {"WARN", {absl::LogSeverityAtLeast::kWarning, 0}},
        {"ERROR", {absl::LogSeverityAtLeast::kError, 0}},
        {"FATAL", {absl::LogSeverityAtLeast::kFatal, 0}},
    };

    // This is the fallback log level, an option of last resort if nothing else is specified.
    std::string_view level_to_use = kDefaultLogLevel;
    bool invalid_env_var = false;

    // Check environment variable, it has priority over compile-time default.
    // Not use facilities from nixl::config to prevent cyclic initialization dependency.
    const char *env_log_level = std::getenv("NIXL_LOG_LEVEL");
    std::string env_level_str_upper;
    if (env_log_level != nullptr) {
        env_level_str_upper = absl::AsciiStrToUpper(env_log_level);
        if (kLogLevelMap.contains(env_level_str_upper)) {
            level_to_use = env_level_str_upper;
        } else {
            // Fall back to kDefaultLogLevel if env var is invalid
            invalid_env_var = true;
        }
    }

    // Apply the settings
    auto it = kLogLevelMap.find(level_to_use);
    const LogLevelSettings &settings =
        (it != kLogLevelMap.end()) ? it->second : kLogLevelMap.at(kDefaultLogLevel);
    absl::SetMinLogLevel(settings.min_severity);
    absl::SetVLogLevel("*", settings.vlog_level);
    absl::SetStderrThreshold(settings.min_severity);
    absl::InitializeLog();

    nixl::initLogFile();

#ifdef NIXL_VERSION
    NIXL_INFO << "NIXL version: " << NIXL_VERSION
#ifdef NIXL_GIT_HASH
              << " (git: " << NIXL_GIT_HASH << ")"
#endif
        ;
#endif

    if (invalid_env_var) {
        NIXL_WARN << "Invalid NIXL_LOG_LEVEL environment variable, using default log level: "
                  << kDefaultLogLevel;
    }
}

} // anonymous namespace

namespace nixl {

/** @brief Test-only path to the file sink, so a record can exceed Abseil's cap. */
void
submitLogPayloadForTest(std::string_view payload) {
    auto &state = getLogFileState();
    if (state.sink != nullptr) {
        state.sink->writePayload(payload);
    }
}

/** @brief Registers the NIXL_LOG_FILE sink; see nixl_log.h for the contract. */
bool
initLogFile() {
    auto &state = getLogFileState();
    const std::lock_guard lock(state.mutex);

    if (state.sink != nullptr) {
        return true;
    }

    const char *configured = std::getenv(log_file_env_var);
    if (configured == nullptr || *configured == '\0') {
        return false;
    }
    const auto path = expandLogPath(configured);
    if (!path.has_value()) {
        reportSetupFailure(absl::StrCat("Invalid ",
                                        log_file_env_var,
                                        " '",
                                        configured,
                                        "': expected only %h, %p, %t or %% escapes"));
        return false;
    }

    const char *size_setting = std::getenv(log_file_size_env_var);
    const std::string_view configured_size = size_setting != nullptr ? size_setting : "";
    const auto limit = parseLogFileSize(configured_size);
    if (!limit.has_value()) {
        reportSetupFailure(
            absl::StrCat("Invalid ",
                         log_file_size_env_var,
                         " '",
                         configured_size,
                         "': expected a byte count, optionally suffixed with K, M or G"));
        return false;
    }
    if (!configured_size.empty() && *limit < min_log_file_size) {
        reportSetupFailure(absl::StrCat("Invalid ",
                                        log_file_size_env_var,
                                        " '",
                                        configured_size,
                                        "': value is below the minimum of ",
                                        min_log_file_size,
                                        " bytes"));
        return false;
    }

    std::error_code path_error;
    const std::filesystem::path resolved_path = std::filesystem::absolute(*path, path_error);
    if (path_error) {
        reportSetupFailure(absl::StrCat(
            "Could not open ", log_file_env_var, " '", *path, "': ", path_error.message()));
        return false;
    }

    auto sink = new fileLogSink(resolved_path.string(), *limit);
    if (!sink->isOpen()) {
        const int open_errno = errno;
        delete sink;
        reportSetupFailure(absl::StrCat("Could not open ",
                                        log_file_env_var,
                                        " '",
                                        resolved_path.string(),
                                        "'",
                                        open_errno != 0 ? ": " + nixl_strerror(open_errno) : ""));
        return false;
    }

    absl::AddLogSink(sink);
    state.sink = sink;
    return true;
}

/** @brief Removes the NIXL_LOG_FILE sink: unregister, then destroy. */
void
shutdownLogFile() {
    auto &state = getLogFileState();
    const std::lock_guard lock(state.mutex);

    if (state.sink == nullptr) {
        return;
    }

    // Unregistered first, so no record can arrive while the file is closing.
    // RemoveLogSink waits for calls already inside Send() to return.
    absl::RemoveLogSink(state.sink);
    delete state.sink;
    state.sink = nullptr;
}

} // namespace nixl

namespace {

/** @brief Unload hook. On glibc this runs after static destructors. */
void
shutdownNixlLogging() __attribute__((destructor));

/** @brief Definition of the destructor-attribute hook declared above. */
void
shutdownNixlLogging() {
    nixl::shutdownLogFile();
}

} // anonymous namespace
