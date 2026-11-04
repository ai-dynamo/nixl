/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_UTILS_COMMON_NIXL_LOG_FILE_SINK_H
#define NIXL_SRC_UTILS_COMMON_NIXL_LOG_FILE_SINK_H

#include "scoped_fd.h"

#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>

namespace nixl {

/** @brief Appends log records to a file, formatted exactly as on stderr. */
class __attribute__((visibility("hidden"))) fileLogSink final : public absl::LogSink {
public:
    static constexpr const char *log_file_env_var = "NIXL_LOG_FILE";
    static constexpr const char *log_file_size_env_var = "NIXL_LOG_FILE_SIZE";
    static constexpr const char *rotated_suffix = ".1";

    /**
     * @brief Opens @p path for append; check isOpen() rather than catching.
     * @param path  Where to write, already expanded and made absolute.
     * @param limit Bytes before rotating, 0 for none. Counted from the current
     *              size, since appending inherits whatever the file holds.
     */
    fileLogSink(const std::string &path, std::uintmax_t limit);

    /** @brief False if the sink cannot write, and must not be registered. */
    [[nodiscard]] bool
    isOpen() const noexcept;

    /**
     * @brief Writes one record directly to the file descriptor.
     * @param entry Borrowed; valid only for this call.
     */
    void
    Send(const absl::LogEntry &entry) override;

    /** @brief Writes one payload. Takes mutex_. */
    void
    writePayload(std::string_view payload);

private:
    /**
     * @brief Moves the full file aside and starts a new one, keeping the newest
     *        records. Existing oversized files are preserved until a later
     *        rotation replaces them. A failed rotation stops the sink.
     *        Called with mutex_ held.
     */
    void
    rotate();

    /**
     * @brief Reports the first write failure and stops using the file, going
     *        to stderr directly rather than through the machinery that failed.
     * @param reason errno from the failed operation, or 0. Called with mutex_ held.
     */
    void
    reportFailure(int reason);

    /** @brief Reports the first oversized record. Called with mutex_ held. */
    void
    reportOversizedRecord();

    /**
     * @brief Reports a failed rotation and stops using the file, leaving it as
     *        it is: those records are all there will be. Reports once, since
     *        failed_ stops Send() rotating again.
     * @param reason Why the rename failed. Called with mutex_ held.
     */
    void
    reportRotateFailure(const std::error_code &reason);

    std::mutex mutex_;
    std::string path_;
    nixl::scopedFd fd_;
    std::uintmax_t limit_ = 0;
    std::uintmax_t written_ = 0;
    bool failed_ = false;
    bool reported_oversize_ = false;
};

} // namespace nixl

#endif /* NIXL_SRC_UTILS_COMMON_NIXL_LOG_FILE_SINK_H */
