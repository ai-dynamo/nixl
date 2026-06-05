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
#include "file_path_mode.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <system_error>
#include <utility>
#include <vector>

#include <absl/strings/str_split.h>

#include "common/nixl_log.h"

namespace nixl {

std::optional<PathSpec>
parsePathMeta(const std::string &s) {
    auto colon = s.find(':');
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    std::string modes = s.substr(0, colon);
    std::string path = s.substr(colon + 1);

    std::vector<std::string> tokens = absl::StrSplit(modes, ',');

    int flags;
    if (tokens[0] == "ro") {
        flags = O_RDONLY;
    } else if (tokens[0] == "rw") {
        flags = O_RDWR;
    } else {
        return std::nullopt;
    }

    // From here on: Starts with ro or rw and has colon -> caller intends path-mode -> warn on error
    if (path.empty()) {
        NIXL_DEBUG << "path-mode registration ignored: empty path in \"" << s << "\"";
        return std::nullopt;
    }

    mode_t mode = 0;
    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "direct") {
            flags |= O_DIRECT;
        } else if (tokens[i] == "sync") {
            flags |= O_SYNC;
        } else if (tokens[i] == "noatime") {
            flags |= O_NOATIME;
        } else if (tokens[i] == "create") {
            flags |= O_CREAT;
            mode = 0644;
        } else {
            NIXL_DEBUG << "path-mode registration ignored: unknown flag token \"" << tokens[i]
                       << "\" in \"" << s << "\"";
            return std::nullopt;
        }
    }

    return PathSpec{path, flags, mode};
}

FileFd::FileFd(int fallback_fd, const std::string &metaInfo) {
    auto spec = parsePathMeta(metaInfo);
    if (!spec) {
        fd_ = fallback_fd;
        return;
    }
    int fd = ::open(spec->path.c_str(), spec->flags, spec->mode);
    if (fd < 0) {
        throw std::system_error(
            errno, std::generic_category(), "nixl::FileFd(\"" + spec->path + "\")");
    }
    fd_ = fd;
    owned_ = true;
    path_ = std::move(spec->path);
}

FileFd::FileFd(FileFd &&other) noexcept
    : fd_(other.fd_),
      owned_(other.owned_),
      path_(std::move(other.path_)) {
    other.fd_ = -1;
    other.owned_ = false;
}

FileFd &
FileFd::operator=(FileFd &&other) noexcept {
    if (this != &other) {
        if (owned_ && fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = other.fd_;
        owned_ = other.owned_;
        path_ = std::move(other.path_);
        other.fd_ = -1;
        other.owned_ = false;
    }
    return *this;
}

FileFd::~FileFd() {
    if (owned_ && fd_ >= 0) {
        ::close(fd_);
    }
}

} // namespace nixl
