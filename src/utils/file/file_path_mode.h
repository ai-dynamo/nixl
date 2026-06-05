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
#ifndef __FILE_PATH_MODE_H
#define __FILE_PATH_MODE_H

#include <sys/types.h>

#include <optional>
#include <string>

#include "backend/backend_engine.h"

// Path-mode parser for FILE_SEG. Grammar (intentionally verbose: API contract).
//
//   metaInfo := <modes>:<path>     # path-mode
//             | <anything else>    # fd-in-devId mode
//   modes    := <access>[,<flag>]*
//   access   := "ro"               # O_RDONLY
//             | "rw"               # O_RDWR
//   flag     := "direct"           # | O_DIRECT
//             | "sync"             # | O_SYNC
//             | "noatime"          # | O_NOATIME
//             | "create"           # | O_CREAT (mode 0644)
//
// Unknown tokens: parsePathMeta returns std::nullopt (fail-loud).

namespace nixl {

// `mode` is only consumed when O_CREAT is in `flags`.
struct PathSpec {
    std::string path;
    int flags;
    mode_t mode;
};

std::optional<PathSpec>
parsePathMeta(const std::string &s);

// RAII wrapper bundling an fd + ownership flag + (optional) path string.
// Used by FILE_SEG backends so per-MD bookkeeping (owned/path) lives in
// one place. One ctor covers both registration modes:
//
//   FileFd(fallback_fd, metaInfo)
//     metaInfo parses as path-mode -> open(spec) and own the fd;
//                                     throws std::system_error on open() failure.
//     otherwise                    -> fd = fallback_fd, not owned.
//
// Move-only. fd() returns the held fd; dtor closes iff owned.
class FileFd {
public:
    FileFd() = default;
    FileFd(int fallback_fd, const std::string &metaInfo);

    FileFd(const FileFd &) = delete;
    FileFd &
    operator=(const FileFd &) = delete;
    FileFd(FileFd &&other) noexcept;
    FileFd &
    operator=(FileFd &&other) noexcept;
    ~FileFd();

    int
    fd() const noexcept {
        return fd_;
    }

    const std::string &
    path() const noexcept {
        return path_;
    }

private:
    int fd_ = -1;
    bool owned_ = false;
    std::string path_;
};

} // namespace nixl

class nixlFilePathMD : public nixlBackendMD {
public:
    nixl::FileFd file_fd;

    nixlFilePathMD() : nixlBackendMD(true /*isPrivate*/) {}

    explicit nixlFilePathMD(nixl::FileFd &&fd) : nixlBackendMD(true), file_fd(std::move(fd)) {}
};

#endif // __FILE_PATH_MODE_H
