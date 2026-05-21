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

// RAII wrapper holding an fd. One ctor handles both modes:
//   * metaInfo parses as path-mode -> open(spec); fd is owned and closed
//     on dtor. Throws std::system_error if open() fails.
//   * otherwise                    -> fd = fallback_fd, not owned.
// fd() returns the held fd in both cases (-1 if default-constructed).
// Move-only; dtor closes iff owned. Dtor is virtual so backend-specific
// "FileHandle" classes can inherit cleanly.
class fdHandle {
public:
    fdHandle() = default;
    explicit fdHandle(const std::string &metaInfo, int fallback_fd = -1);

    fdHandle(const fdHandle &) = delete;
    fdHandle &operator=(const fdHandle &) = delete;
    fdHandle(fdHandle &&other) noexcept;
    fdHandle &operator=(fdHandle &&other) noexcept;

    virtual ~fdHandle();

    int fd() const noexcept { return fd_; }

private:
    int fd_ = -1;
    bool owned_ = false;
};

} // namespace nixl

#endif // __FILE_PATH_MODE_H
