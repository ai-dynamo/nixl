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
#include <cstdlib>
#include <fstream>
#include <ios>
#include <mutex>
#include <string>
#include <string_view>

namespace {

// Structure to hold logging settings
struct LogLevelSettings {
    absl::LogSeverityAtLeast min_severity;
    int vlog_level;
};

// Default log level if nothing else is specified
constexpr std::string_view kDefaultLogLevel = "WARN";

// Names the file that log records are mirrored into. Unset disables the sink.
constexpr const char *kLogFileEnvVar = "NIXL_LOG_FILE";

/*
 * Appends log records to a file, formatted exactly as they appear on stderr so
 * the two outputs can be compared line for line.
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
     */
    explicit fileLogSink(const std::string &path) : file_(path, std::ios::app) {}

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
        /* Carries the severity, timestamp and source location. */
        const auto line = entry.text_message_with_prefix_and_newline();

        const std::lock_guard<std::mutex> lock(mutex_);
        file_.write(line.data(), static_cast<std::streamsize>(line.size()));
        file_.flush();
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
        file_.flush();
    }

private:
    std::mutex mutex_;
    std::ofstream file_;
};

std::mutex log_file_mutex;

/*
 * Owned manually rather than through a smart pointer with static storage.
 * Static destructors run before .fini_array, so a self-destroying sink would
 * unregister itself while later shutdown code could still be logging; deleting
 * it from the destructor-attribute function below keeps it alive to the end.
 */
fileLogSink *log_file_sink = nullptr; // guarded by log_file_mutex

/**
 * @brief Applies NIXL_LOG_LEVEL and NIXL_LOG_FILE, before main() runs.
 *
 * Invoked through the constructor attribute so logging is configured before any
 * NIXL code can emit a record.
 */
void InitializeNixlLogging() __attribute__((constructor));

/** @brief Definition of the constructor-attribute hook declared above. */
void InitializeNixlLogging()
{
    // Map from log level string to settings
    const absl::flat_hash_map<std::string_view, LogLevelSettings> kLogLevelMap = {
        {"TRACE", {absl::LogSeverityAtLeast::kInfo, 2}},
        {"DEBUG", {absl::LogSeverityAtLeast::kInfo, 1}},
        {"INFO",  {absl::LogSeverityAtLeast::kInfo, 0}},
        {"WARN",  {absl::LogSeverityAtLeast::kWarning, 0}},
        {"ERROR", {absl::LogSeverityAtLeast::kError, 0}},
        {"FATAL", {absl::LogSeverityAtLeast::kFatal, 0}},
    };

    // This is the fallback log level, an option of last resort if nothing else is specified.
    std::string_view level_to_use = kDefaultLogLevel;
    bool invalid_env_var = false;

    // Check environment variable, it has priority over compile-time default.
    // Not use facilities from nixl::config to prevent cyclic initialization dependency.
    const char* env_log_level = std::getenv("NIXL_LOG_LEVEL");
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
    const LogLevelSettings& settings = (it != kLogLevelMap.end()) ? it->second : kLogLevelMap.at(kDefaultLogLevel);
    absl::SetMinLogLevel(settings.min_severity);
    absl::SetVLogLevel("*", settings.vlog_level);
    absl::SetStderrThreshold(settings.min_severity);
    absl::InitializeLog();

    /* Registered before the records below so the version banner, the most
     * useful line for identifying which build a process is running, is
     * captured in the file as well. */
    nixl::initLogFile();

#ifdef NIXL_VERSION
    NIXL_INFO << "NIXL version: " << NIXL_VERSION
#ifdef NIXL_GIT_HASH
              << " (git: " << NIXL_GIT_HASH << ")"
#endif
        ;
#endif

    if (invalid_env_var) {
        NIXL_WARN << "Invalid NIXL_LOG_LEVEL environment variable, using default log level: " << kDefaultLogLevel;
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

    const char *path = std::getenv(kLogFileEnvVar);
    if (path == nullptr || *path == '\0') {
        return false;
    }

    /* Cleared so the reason below cannot report a leftover value from some
     * unrelated earlier call; ofstream is not required to set errno. */
    errno = 0;
    auto sink = new fileLogSink(path);
    if (!sink->isOpen()) {
        const int open_errno = errno;
        delete sink;
        /* Reported on stderr and then dropped. Losing the log file must not
         * stop the process it was meant to describe. */
        NIXL_WARN << "Could not open " << kLogFileEnvVar << " '" << path
                  << "', continuing without a log file"
                  << (open_errno != 0 ? ": " + nixl_strerror(open_errno) : "");
        return false;
    }

    /* Registered last: until this call the sink is invisible to Abseil, so a
     * concurrent log record can never reach a half-built sink. */
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

    /* Unregistered first, so no record can arrive while the file is closing.
     * RemoveLogSink waits for calls already inside Send() to return. */
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
 * still be logging. On glibc that also puts it after the destructors
 * __cxa_atexit queues, so records emitted late in shutdown still reach the
 * file, but that part is a property of the loader and not something the
 * standard promises.
 *
 * Correctness does not depend on the ordering either way. Send() flushes every
 * record as it is written, so the worst a different order can cost is the few
 * records emitted after this runs; it can never lose an earlier record, and it
 * cannot leave a registered sink dangling, because the sink is removed from
 * Abseil before it is destroyed.
 */
void
ShutdownNixlLogging() __attribute__((destructor));

/** @brief Definition of the destructor-attribute hook declared above. */
void
ShutdownNixlLogging() {
    nixl::shutdownLogFile();
}

} // anonymous namespace
