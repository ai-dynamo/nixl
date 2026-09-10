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

#include "hostname.h"
#include "nixl_log.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/ascii.h"
#include "absl/container/flat_hash_map.h"
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
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

// Bounds that file. Unset, empty or 0 lets it grow without limit.
constexpr const char *log_file_size_env_var = "NIXL_LOG_FILE_SIZE";

// Appended to the log file's name to hold the records rotated out of it.
constexpr const char *rotated_suffix = ".1";

/**
 * @brief Nanoseconds since the epoch, sampled once, for %t.
 */
uint64_t
processRunMarker() {
    static const uint64_t marker = [] {
        struct timespec ts {};
        ::clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
    }();
    return marker;
}

/**
 * @brief Expands the escapes in a log file path:
 *
 *   %h  host name
 *   %p  process id
 *   %t  this process's run marker, in nanoseconds
 *   %%  a literal percent
 *
 * Lets one NIXL_LOG_FILE serve every worker of a run and still give each its
 * own file.
 */
std::string
expandLogPath(const std::string &pattern) {
    std::string expanded;
    expanded.reserve(pattern.size() + 32);

    for (size_t at = 0; at < pattern.size(); ++at) {
        if (pattern[at] != '%' || at + 1 == pattern.size()) {
            expanded += pattern[at];
            continue;
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
            expanded += pattern[at];
            break;
        }
    }
    return expanded;
}

/**
 * @brief Parses NIXL_LOG_FILE_SIZE, for example "64M". K, M, G are powers of 1024.
 * @return Bytes, 0 for no limit, or nullopt if @p text is not a size.
 */
std::optional<std::uintmax_t>
parseLogFileSize(const std::string &text) {
    if (text.empty()) {
        return 0;
    }

    // stoull reads "-1" as a wrap-around to the largest possible limit, and
    // also accepts leading whitespace and a plus. Insist on a digit first.
    if (std::isdigit(static_cast<unsigned char>(text.front())) == 0) {
        return std::nullopt;
    }

    size_t digits = 0;
    std::uintmax_t value = 0;
    try {
        value = std::stoull(text, &digits);
    }
    catch (const std::exception &) {
        return std::nullopt;
    }

    std::uintmax_t scale = 1;
    const std::string suffix = text.substr(digits);
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

/**
 * @brief Appends log records to a file, formatted exactly as on stderr.
 *
 * Abseil may call Send() from any thread, so writes are serialized. Every
 * record is flushed as it arrives: a log that explains a crash cannot hold its
 * tail in a buffer.
 */
class fileLogSink final : public absl::LogSink {
public:
    /**
     * @brief Opens @p path for append; check isOpen() rather than catching.
     * @param path  Where to write, already expanded.
     * @param limit Bytes before rotating, 0 for none. Counted from the current
     *              size, since appending inherits whatever the file holds.
     */
    fileLogSink(const std::string &path, std::uintmax_t limit)
        : path_(path),
          file_(path, std::ios::app),
          limit_(limit) {
        if (!file_.is_open()) {
            return;
        }

        std::error_code ec;
        const auto existing = std::filesystem::file_size(path, ec);
        written_ = ec ? 0 : existing;
    }

    /** @brief False if the sink cannot write, and must not be registered. */
    bool
    isOpen() const {
        return file_.is_open();
    }

    /**
     * @brief Writes one record and flushes it.
     * @param entry Borrowed; valid only for this call.
     */
    void
    Send(const absl::LogEntry &entry) override {
        const auto line = entry.text_message_with_prefix_and_newline();

        const std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) {
            return;
        }

        // Before the write, so the limit is one the file stays under.
        if (limit_ != 0 && written_ != 0 && written_ + line.size() > limit_) {
            rotate();
            if (failed_) {
                return;
            }
        }

        // Cleared: iostreams need not set errno, so a stale one could be read.
        errno = 0;
        file_.write(line.data(), static_cast<std::streamsize>(line.size()));
        file_.flush();
        if (!file_) {
            reportFailure(errno);
            return;
        }
        written_ += line.size();
    }

    /** @brief Honours absl::FlushLogSinks(); Send() already flushes each record. */
    void
    Flush() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) {
            return;
        }

        errno = 0;
        file_.flush();
        if (!file_) {
            reportFailure(errno);
        }
    }

private:
    /**
     * @brief Moves the full file aside and starts a new one, keeping the newest
     *        records. One generation is kept, so at most twice the limit is on
     *        disk. A rotation that cannot be done stops the sink rather than
     *        ignoring the limit. Called with mutex_ held.
     */
    void
    rotate() {
        file_.close();

        std::error_code ec;
        std::filesystem::rename(path_, path_ + rotated_suffix, ec);
        if (ec) {
            reportRotateFailure(ec);
            return;
        }

        errno = 0;
        file_.open(path_, std::ios::app);
        if (!file_) {
            reportFailure(errno);
            return;
        }
        written_ = 0;
    }

    /**
     * @brief Reports the first write failure and stops using the file.
     *
     * Once a stream has failed, every later write on it is a silent no-op, so
     * without this the file would simply stop part way through with nothing to
     * say why. Records after the failure are dropped rather than retried: the
     * process being described must not be held up by its own log file.
     *
     * Goes straight to stderr rather than through NIXL_WARN so that reporting
     * the failure does not depend on the machinery that just failed.
     *
     * @param reason errno from the failed operation, or 0 if it was not set.
     *               Called with mutex_ held.
     */
    void
    reportFailure(int reason) {
        failed_ = true;

        const std::string detail = reason != 0 ? ": " + nixl_strerror(reason) : "";
        std::fprintf(stderr,
                     "NIXL: could not write to %s '%s'%s; dropping further records\n",
                     log_file_env_var,
                     path_.c_str(),
                     detail.c_str());
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
    std::ofstream file_;
    std::uintmax_t limit_ = 0;
    std::uintmax_t written_ = 0;
    bool failed_ = false;
};

std::mutex log_file_mutex;

// Owned manually: a static smart pointer would unregister the sink during
// static destruction, while later shutdown code can still be logging.
fileLogSink *log_file_sink = nullptr; // guarded by log_file_mutex

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

/**
 * @brief Registers the NIXL_LOG_FILE sink; see nixl_log.h for the contract.
 * @return true if a sink is registered on return, including when one already was.
 */
bool
initLogFile() {
    const std::lock_guard<std::mutex> lock(log_file_mutex);

    if (log_file_sink != nullptr) {
        return true;
    }

    const char *configured = std::getenv(log_file_env_var);
    if (configured == nullptr || *configured == '\0') {
        return false;
    }
    const std::string path = expandLogPath(configured);

    const char *configured_size = std::getenv(log_file_size_env_var);
    const auto limit = parseLogFileSize(configured_size != nullptr ? configured_size : "");
    if (!limit.has_value()) {
        NIXL_ERROR << "Invalid " << log_file_size_env_var << " '" << configured_size
                   << "': expected a byte count, optionally suffixed with K, M or G";
        return false;
    }

    // Cleared: ofstream need not set errno, so a stale one could be read.
    errno = 0;
    auto sink = new fileLogSink(path, *limit);
    if (!sink->isOpen()) {
        const int open_errno = errno;
        delete sink;
        // Losing the log file must not stop the process it describes.
        NIXL_ERROR << "Could not open " << log_file_env_var << " '" << path
                   << "', continuing without a log file"
                   << (open_errno != 0 ? ": " + nixl_strerror(open_errno) : "");
        return false;
    }

    absl::AddLogSink(sink);
    log_file_sink = sink;
    return true;
}

/** @brief Removes the NIXL_LOG_FILE sink: unregister, flush, then destroy. */
void
shutdownLogFile() {
    const std::lock_guard<std::mutex> lock(log_file_mutex);

    if (log_file_sink == nullptr) {
        return;
    }

    // Unregistered first, so no record can arrive while the file is closing.
    // RemoveLogSink waits for calls already inside Send() to return.
    absl::RemoveLogSink(log_file_sink);
    log_file_sink->Flush();
    delete log_file_sink;
    log_file_sink = nullptr;
}

} // namespace nixl

namespace {

/**
 * @brief Tears the log file down at library unload.
 *
 * Placed in .fini_array rather than in a static destructor because a
 * self-destroying sink would unregister itself while later shutdown code could
 * still be logging. On glibc this also runs after the exit-handler queue that
 * __cxa_atexit registers static destructors on, so records emitted late in
 * shutdown still reach the file. That ordering is loader behaviour rather than
 * a language guarantee, so it is covered by a test rather than assumed:
 * nixlLogFileTest.RecordsFromStaticDestructorsReachTheFile.
 *
 * Correctness does not depend on the ordering even so. Send() flushes every
 * record as it is written, so the worst a different order can cost is the few
 * records emitted after this runs; it can never lose an earlier record, and it
 * cannot leave a registered sink dangling, because the sink is removed from
 * Abseil before it is destroyed.
 */
void
shutdownNixlLogging() __attribute__((destructor));

/** @brief Definition of the destructor-attribute hook declared above. */
void
shutdownNixlLogging() {
    nixl::shutdownLogFile();
}

} // anonymous namespace
