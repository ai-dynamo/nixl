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

#include "nixl_log_file_sink.h"

#include "nixl_log.h"

#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>

namespace nixl {

fileLogSink::fileLogSink(const std::string &path, std::uintmax_t limit)
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

bool
fileLogSink::isOpen() const noexcept {
    return fd_.valid();
}

void
fileLogSink::Send(const absl::LogEntry &entry) {
    const auto payload = entry.stacktrace().empty() ? entry.text_message_with_prefix_and_newline() :
                                                      entry.stacktrace();
    writePayload(payload);
}

void
fileLogSink::writePayload(std::string_view payload) {
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
        const ssize_t result = ::write(fd_.get(), payload.data() + offset, payload.size() - offset);
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

void
fileLogSink::rotate() {
    fd_.reset();

    std::error_code ec;
    std::filesystem::rename(path_, path_ + rotated_suffix, ec);
    if (ec) {
        reportRotateFailure(ec);
        return;
    }

    fd_ = nixl::scopedFd(::open(path_.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666));
    if (!fd_.valid()) {
        reportFailure(errno);
        return;
    }
    written_ = 0;
}

void
fileLogSink::reportFailure(int reason) {
    failed_ = true;

    const std::string detail = (reason != 0) ? (": " + nixl_strerror(reason)) : "";
    std::fprintf(stderr,
                 "NIXL: could not write to %s '%s'%s; dropping further records\n",
                 log_file_env_var,
                 path_.c_str(),
                 detail.c_str());
}

void
fileLogSink::reportOversizedRecord() {
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

void
fileLogSink::reportRotateFailure(const std::error_code &reason) {
    failed_ = true;

    std::fprintf(stderr,
                 "NIXL: could not rotate %s '%s' at its %s (%s); "
                 "dropping further records\n",
                 log_file_env_var,
                 path_.c_str(),
                 log_file_size_env_var,
                 reason.message().c_str());
}

} // namespace nixl
