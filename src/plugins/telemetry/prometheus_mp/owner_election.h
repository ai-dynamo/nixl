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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_OWNER_ELECTION_H
#define NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_OWNER_ELECTION_H

#include "common/nixl_log.h"
#include "scoped_fd.h"

#include <fcntl.h>
#include <sys/file.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <string>

namespace nixl::telemetry::mp {

// Deliberately outside the collector's <prefix>*<suffix> store pattern so the
// directory scan never sees it.
inline constexpr char ownerLockFileName[] = "nixl-owner.lock";

/**
 * @class ownerElection
 * @brief Picks the single process of a telemetry directory allowed to serve the
 *        scrape endpoint.
 *
 * The election is an flock rather than the port bind itself: two processes
 * binding concurrently cannot tell which of them got there first, whereas the
 * lock admits exactly one, so only the winner ever binds. The winner then
 * publishes the endpoint it bound, which is how a loser can tell that it was
 * configured for a different one. The kernel releases the lock when the holder
 * dies, so it needs no cleanup.
 */
class ownerElection {
public:
    ownerElection() = default;

    /**
     * @brief Runs the election for @p dir, without blocking.
     * @param dir The shared telemetry directory the ranks contend for.
     */
    explicit ownerElection(const std::filesystem::path &dir)
        : fd_(::open((dir / ownerLockFileName).c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600)) {
        // A directory that cannot hold the lock file at all leaves every process
        // thinking it won, which is the unelected behaviour: they all try to bind
        // and the port decides.
        won_ = !fd_.valid() || ::flock(fd_.get(), LOCK_EX | LOCK_NB) == 0;
    }

    [[nodiscard]] bool
    won() const noexcept {
        return won_;
    }

    /**
     * @brief Records where the winner listens, for the losers to read back.
     * @param endpoint The bound "address:port".
     *
     * Best-effort: it only decides whether a loser warns.
     */
    void
    publishEndpoint(const std::string &endpoint) const {
        if (!fd_.valid()) {
            return;
        }
        if (::ftruncate(fd_.get(), 0) != 0 ||
            ::pwrite(fd_.get(), endpoint.data(), endpoint.size(), 0) !=
                static_cast<ssize_t>(endpoint.size())) {
            NIXL_DEBUG << "prometheus_mp: cannot record the owner endpoint in " << ownerLockFileName
                       << ": " << strerror(errno);
        }
    }

    /**
     * @brief The endpoint the winner published.
     * @return Empty if nothing has been published yet -- the election is decided
     *         before the bind, so a loser can observe the gap.
     */
    [[nodiscard]] std::string
    publishedEndpoint() const {
        char buf[64];
        const ssize_t len = fd_.valid() ? ::pread(fd_.get(), buf, sizeof(buf), 0) : -1;
        return len > 0 ? std::string(buf, static_cast<std::size_t>(len)) : std::string();
    }

    void
    release() noexcept {
        fd_.reset();
    }

private:
    scopedFd fd_;
    bool won_ = false;
};

} // namespace nixl::telemetry::mp

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_OWNER_ELECTION_H
