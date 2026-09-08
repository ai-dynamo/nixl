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

#include "nixl_log.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/ascii.h"
#include "absl/container/flat_hash_map.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
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
 * @brief This machine's host name, for %h.
 * @return The host name, or "unknown-host" if it cannot be read, since a
 *         usable-but-vague filename beats failing to log at all.
 */
std::string
hostName() {
    char buffer[256] = {};
    if (::gethostname(buffer, sizeof(buffer) - 1) != 0) {
        return "unknown-host";
    }
    return buffer;
}

/**
 * @brief Expands %h, %p and %% in a log file path.
 *
 * Lets every process be given the same NIXL_LOG_FILE while still writing to its
 * own file, which is the only way the setting is usable across the many workers
 * of a disaggregated inference run.
 *
 * @param pattern The configured path, which need not contain any escape.
 * @return @p pattern with %h replaced by the host name, %p by the process id
 *         and %% by a literal %. An unrecognized escape is left exactly as it
 *         was, so a path that legitimately contains a percent still works.
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
            expanded += hostName();
            ++at;
            break;
        case 'p':
            expanded += std::to_string(::getpid());
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
 * @brief Parses NIXL_LOG_FILE_SIZE, which accepts a K, M or G suffix.
 *
 * @param text The configured value, for example "64M". Suffixes are powers of
 *             1024 and case does not matter.
 * @return The limit in bytes, 0 meaning no limit, or nullopt if @p text is not
 *         a size, which the caller reports rather than guessing at.
 */
std::optional<std::uintmax_t>
parseLogFileSize(const std::string &text) {
    if (text.empty()) {
        return 0;
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
 * Keeping the formatting identical lets the two outputs be compared line for
 * line.
 *
 * Abseil may call Send() from any thread, so writes are serialized. Each record
 * is flushed as it arrives: the point of the file is to explain what a process
 * did before it crashed or hung, and holding the tail of the log in a buffer is
 * precisely the failure that would defeat that.
 */
class fileLogSink final : public absl::LogSink {
public:
    /**
     * @brief Opens @p path for append, creating it if needed.
     *
     * Never throws: a failure to open leaves the sink closed, which the caller
     * detects with isOpen() rather than by catching an exception.
     *
     * @param path  Where to write, already expanded.
     * @param limit Bytes to hold before rotating, or 0 for no limit. Counted
     *              from the file's current size, since opening for append onto
     *              an earlier run's file inherits whatever it already holds.
     */
    fileLogSink(const std::string &path, std::uintmax_t limit)
        : path_(path),
          file_(path, std::ios::app),
          limit_(limit) {
        std::error_code ec;
        const auto existing = std::filesystem::file_size(path, ec);
        written_ = ec ? 0 : existing;
    }

    /**
     * @brief Reports whether the file was opened successfully.
     * @return false if the sink cannot write, in which case it must not be
     *         registered with Abseil.
     */
    bool
    isOpen() const {
        return file_.is_open();
    }

    /**
     * @brief Writes one record to the file, then flushes it.
     *
     * Abseil calls this from the logging thread and requires it to be
     * thread-safe, so the write is serialized on mutex_.
     *
     * @param entry The record to write. Its formatted text is borrowed and is
     *              valid only for the duration of this call.
     */
    void
    Send(const absl::LogEntry &entry) override {
        // Carries the severity, timestamp and source location.
        const auto line = entry.text_message_with_prefix_and_newline();

        const std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) {
            return;
        }

        // Rotated before the write rather than after, so the limit is one the
        // file stays under rather than one it briefly exceeds. A record larger
        // than the whole limit is written anyway, to an empty file, since
        // rotating for it would loop without ever making room.
        if (limit_ != 0 && written_ != 0 && written_ + line.size() > limit_) {
            rotate();
            if (failed_) {
                return;
            }
        }

        // Cleared so the reason reported below belongs to this write; iostreams
        // are not required to set errno.
        errno = 0;
        file_.write(line.data(), static_cast<std::streamsize>(line.size()));
        file_.flush();
        if (!file_) {
            reportFailure(errno);
            return;
        }
        written_ += line.size();
    }

    /**
     * @brief Flushes the file on demand.
     *
     * Send() already flushes every record, so this exists to honour the
     * absl::LogSink contract and absl::FlushLogSinks(), which may be called
     * from any thread.
     */
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
     * @brief Moves the full file aside and starts a new one.
     *
     * Keeps the most recent records, which are the ones worth having when the
     * question is what a process did before it hung. The cost is that up to
     * twice the limit is on disk: the new file and the one rotated out of it.
     * Exactly one generation is kept, so the total stays bounded.
     *
     * A failed rename leaves rotation switched off rather than retried every
     * record. The file then grows past the limit, which is the better of the
     * two bad outcomes: a log that is too large can still be read.
     *
     * Called with mutex_ held.
     */
    void
    rotate() {
        file_.close();

        const std::string rotated = path_ + rotated_suffix;
        std::error_code ec;
        std::filesystem::rename(path_, rotated, ec);
        if (ec) {
            limit_ = 0;
            std::fprintf(stderr,
                         "NIXL: could not rotate log file '%s' to '%s' (%s); "
                         "continuing without a size limit\n",
                         path_.c_str(),
                         rotated.c_str(),
                         ec.message().c_str());
        }

        // Append either way: if the rename failed the file is still there and
        // truncating it would throw away the very records rotation exists to
        // preserve.
        errno = 0;
        file_.open(path_, std::ios::app);
        if (!file_) {
            reportFailure(errno);
            return;
        }
        written_ = ec ? written_ : 0;
    }

    /**
     * @brief Reports the first write failure and stops using the file.
     *
     * Once a stream has failed, every later write on it is a silent no-op, so
     * without this the file would simply stop part way through with nothing to
     * say why. Records after the failure are dropped rather than retried: the
     * process being described must not be held up by its own log file.
     *
     * Goes straight to stderr rather than through NIXL_WARN, and this is not a
     * style choice. It runs inside a log sink while holding mutex_, so emitting
     * a record here would re-enter Send() on this thread and deadlock on that
     * very mutex.
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

    std::mutex mutex_;
    std::string path_;
    std::ofstream file_;
    std::uintmax_t limit_ = 0;
    std::uintmax_t written_ = 0;
    bool failed_ = false;
};

std::mutex log_file_mutex;

// Owned manually rather than through a smart pointer with static storage.
// Static destructors run before .fini_array, so a self-destroying sink would
// unregister itself while later shutdown code could still be logging; deleting
// it from the destructor-attribute function below keeps it alive to the end.
fileLogSink *log_file_sink = nullptr; // guarded by log_file_mutex

/**
 * @brief Applies NIXL_LOG_LEVEL and NIXL_LOG_FILE, before main() runs.
 *
 * Invoked through the constructor attribute so logging is configured before any
 * NIXL code can emit a record.
 */
void
InitializeNixlLogging() __attribute__((constructor));

/** @brief Definition of the constructor-attribute hook declared above. */
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

    // Registered before the records below so the version banner, the most
    // useful line for identifying which build a process is running, is
    // captured in the file as well.
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
        // Left unbounded rather than guessed at. Picking a limit here could
        // discard records the operator meant to keep.
        NIXL_WARN << "Ignoring " << log_file_size_env_var << " '" << configured_size
                  << "': expected a byte count, optionally suffixed with K, M or G";
    }

    // Cleared so the reason below cannot report a leftover value from some
    // unrelated earlier call; ofstream is not required to set errno.
    errno = 0;
    auto sink = new fileLogSink(path, limit.value_or(0));
    if (!sink->isOpen()) {
        const int open_errno = errno;
        delete sink;
        // Reported on stderr and then dropped. Losing the log file must not
        // stop the process it was meant to describe.
        NIXL_WARN << "Could not open " << log_file_env_var << " '" << path
                  << "', continuing without a log file"
                  << (open_errno != 0 ? ": " + nixl_strerror(open_errno) : "");
        return false;
    }

    // Registered last: until this call the sink is invisible to Abseil, so a
    // concurrent log record can never reach a half-built sink.
    absl::AddLogSink(sink);
    log_file_sink = sink;
    return true;
}

/**
 * @brief Removes the NIXL_LOG_FILE sink; see nixl_log.h for the contract.
 *
 * The ordering below is the part worth reading: unregister, then flush, then
 * destroy.
 */
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
