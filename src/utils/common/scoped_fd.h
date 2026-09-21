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
#ifndef NIXL_SRC_UTILS_COMMON_SCOPED_FD_H
#define NIXL_SRC_UTILS_COMMON_SCOPED_FD_H

#include <unistd.h>

#include <utility>

namespace nixl {

/**
 * @class scopedFd
 * @brief Move-only owner of a file descriptor, closed on destruction.
 *
 * Held as a member, it also closes the descriptor when a constructor throws
 * after acquiring it. An empty owner (no descriptor) is a valid state, so a
 * factory can return one to mean failure.
 */
class scopedFd {
public:
    scopedFd() = default;

    explicit scopedFd(int fd) noexcept : fd_(fd) {}

    ~scopedFd() {
        reset();
    }

    scopedFd(const scopedFd &) = delete;
    scopedFd &
    operator=(const scopedFd &) = delete;

    scopedFd(scopedFd &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

    scopedFd &
    operator=(scopedFd &&other) noexcept {
        if (this != &other) {
            reset();
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }

    [[nodiscard]] int
    get() const noexcept {
        return fd_;
    }

    [[nodiscard]] bool
    valid() const noexcept {
        return fd_ >= 0;
    }

    void
    reset() noexcept {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    int fd_ = -1;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_SCOPED_FD_H
