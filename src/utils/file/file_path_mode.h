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
#ifndef NIXL_SRC_UTILS_FILE_FILE_PATH_MODE_HPP
#define NIXL_SRC_UTILS_FILE_FILE_PATH_MODE_HPP

#include <sys/types.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>

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
//
// Path-mode contract: each path-mode file must use a unique devId - the section keys
// on devId, so two files sharing one collide; backends reject a reused path-mode devId on register.

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

// Path-mode contract: each path-mode file registration must use a unique devId
class PathModeDevIdRegistry {
public:
    enum class Ok : bool { No = false, Yes = true };
    enum class Held : bool { No = false, Yes = true };

    class Reservation {
    public:
        Reservation(Reservation &&other) noexcept
            : reg_(other.reg_),
              devId_(other.devId_),
              ok_(other.ok_),
              held_(other.held_) {
            other.reg_ = nullptr;
            other.held_ = Held::No;
        }

        Reservation(const Reservation &) = delete;
        Reservation &
        operator=(const Reservation &) = delete;

        ~Reservation() {
            if (reg_ && held_ == Held::Yes) {
                reg_->release(devId_);
            }
        }

        // Ok::No: this devId is already registered in path-mode -> caller must reject.
        bool
        ok() const noexcept {
            return static_cast<bool>(ok_);
        }

        // Keep the hold past this scope; released later via release() on deregister.
        void
        commit() noexcept {
            held_ = Held::No;
        }

    private:
        friend class PathModeDevIdRegistry;

        Reservation(PathModeDevIdRegistry *reg, uint64_t devId, Ok ok, Held held)
            : reg_(reg),
              devId_(devId),
              ok_(ok),
              held_(held) {}

        PathModeDevIdRegistry *reg_;
        uint64_t devId_;
        Ok ok_;
        Held held_;
    };

    // fd-mode (metaInfo not path-mode) is always ok and untracked.
    [[nodiscard]] Reservation
    reserve(uint64_t devId, const std::string &metaInfo) {
        if (!parsePathMeta(metaInfo)) {
            return Reservation(this, devId, Ok::Yes, Held::No);
        }
        const bool inserted = devids_.insert(devId).second;
        if (inserted) {
            return Reservation(this, devId, Ok::Yes, Held::Yes);
        }
        return Reservation(this, devId, Ok::No, Held::No);
    }

    void
    release(uint64_t devId) {
        devids_.erase(devId);
    }

private:
    std::unordered_set<uint64_t> devids_;
};

} // namespace nixl

class nixlFilePathMD : public nixlBackendMD {
public:
    nixl::FileFd file_fd;
    uint64_t devId = 0;

    nixlFilePathMD() : nixlBackendMD(true /*isPrivate*/) {}

    nixlFilePathMD(uint64_t devid, const std::string &metaInfo)
        : nixlBackendMD(true),
          file_fd(devid, metaInfo),
          devId(devid) {}
};

#endif // NIXL_SRC_UTILS_FILE_FILE_PATH_MODE_HPP
