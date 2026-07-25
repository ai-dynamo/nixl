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
        : fd_(::open((dir / ownerLockFileName).c_str(),
                     O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW,
                     0600)) {
        // Anything that leaves the lock unusable -- no lock file, a filesystem
        // without flock, ENOLCK -- must not read as a loss: every rank would
        // concede and none would serve. Only EWOULDBLOCK means a sibling holds
        // it. The rest degrade to the unelected behaviour, where every process
        // tries to bind and the port decides.
        if (!fd_.valid()) {
            won_ = true;
            warnUnusable(strerror(errno));
            return;
        }
        if (::flock(fd_.get(), LOCK_EX | LOCK_NB) == 0) {
            won_ = true;
            return;
        }
        won_ = errno != EWOULDBLOCK;
        if (won_) {
            warnUnusable(strerror(errno));
        }
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
    // Every rank then believes it was elected, so those that go on to lose the
    // bind report the port as held from outside the run while a sibling is in
    // fact serving. This is the context that makes those reports readable.
    static void
    warnUnusable(const char *reason) {
        NIXL_WARN << "prometheus_mp: cannot use " << ownerLockFileName << " (" << reason
                  << "); falling back to letting the port bind decide which process serves, so "
                  << "a later report of the port being held from outside the run may be a sibling";
    }

    scopedFd fd_;
    bool won_ = false;
};

} // namespace nixl::telemetry::mp

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_OWNER_ELECTION_H
