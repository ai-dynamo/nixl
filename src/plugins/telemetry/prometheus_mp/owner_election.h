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
#include "common/scoped_fd.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace nixl::telemetry::mp {

// Deliberately outside the collector's <prefix>*<suffix> store pattern so the
// directory scan never sees it.
inline constexpr char ownerLockPrefix[] = "nixl-owner.";
inline constexpr char ownerLockSuffix[] = ".lock";

/**
 * @brief The lock file one address of a telemetry directory is elected on.
 * @param address The "address:port" a rank is configured to serve.
 *
 * The address is part of the name, so ranks contend only with the ranks they
 * would collide with, and no lock file ever needs contents: what its holder
 * serves is the name itself.
 */
[[nodiscard]] inline std::string
ownerLockFileName(const std::string &address) {
    std::string name = std::string(ownerLockPrefix) + address + ownerLockSuffix;
    std::replace(name.begin(), name.end(), '/', '_');
    return name;
}

/**
 * @brief The address a lock file name belongs to, empty if it is not one.
 */
[[nodiscard]] inline std::string
addressOfLockFile(const std::string &name) {
    constexpr std::size_t prefix_len = sizeof(ownerLockPrefix) - 1;
    constexpr std::size_t suffix_len = sizeof(ownerLockSuffix) - 1;
    if (name.size() <= prefix_len + suffix_len || !name.starts_with(ownerLockPrefix) ||
        !name.ends_with(ownerLockSuffix)) {
        return {};
    }
    return name.substr(prefix_len, name.size() - prefix_len - suffix_len);
}

/**
 * @class ownerElection
 * @brief Picks the single process of a telemetry directory allowed to serve one
 *        address.
 *
 * The election is an flock rather than the port bind itself: two processes
 * binding concurrently cannot tell which of them got there first, whereas the
 * lock admits exactly one, so only the winner ever binds. It is per address --
 * the lock file is named after it -- so ranks configured for different ports do
 * not contend at all: each such address gets its own owner, which is what the
 * operator asked for by configuring them differently. heldAddressesExcept() is
 * how that is noticed and reported, separately from the election itself. The
 * kernel releases the lock when the holder dies, so it needs no cleanup -- and
 * a writer that re-runs the election then wins it, which is how the address is
 * served again after the owner's death.
 *
 * An election is run by constructing one and given up by destroying it: there
 * is no empty state and no way to move one, because an election re-run by the
 * process already holding it always loses -- flock contends between two open
 * file descriptions of the same process -- so overwriting a held election in
 * place would close the descriptor that holds the lock. Hold it in an optional
 * and reset before re-electing.
 */
class ownerElection {
public:
    /**
     * @brief Runs the election for @p address in @p dir, without blocking.
     * @param dir The shared telemetry directory the ranks contend in.
     * @param address The "address:port" this process would serve.
     * @param warn_if_unusable Whether an unusable lock is worth a warning. False
     *        for the periodic re-elections a writer runs to detect the owner's
     *        death: the condition is unchanged since startup said it once, and
     *        repeating it every retry would bury the log.
     */
    ownerElection(const std::filesystem::path &dir,
                  const std::string &address,
                  bool warn_if_unusable = true)
        : lockName_(ownerLockFileName(address)),
          fd_(::open((dir / lockName_).c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600)) {
        // Anything that leaves the lock unusable -- no lock file, a filesystem
        // without flock, ENOLCK -- must not read as a loss: every rank would
        // concede and none would serve. Only EWOULDBLOCK means a sibling holds
        // it. The rest degrade to the unelected behaviour, where every process
        // tries to bind and the port decides.
        if (!fd_.valid()) {
            won_ = true;
            warnUnusable(strerror(errno), warn_if_unusable);
            return;
        }
        // A co-tenant of a shared directory can hold a lock on a file it planted,
        // which would read as a sibling win and leave every rank writer-only.
        struct stat st{};
        if (::fstat(fd_.get(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != ::geteuid()) {
            fd_.reset();
            won_ = true;
            warnUnusable("not a regular file owned by this user", warn_if_unusable);
            return;
        }
        if (::flock(fd_.get(), LOCK_EX | LOCK_NB) == 0) {
            won_ = true;
            return;
        }
        won_ = errno != EWOULDBLOCK;
        if (won_) {
            warnUnusable(strerror(errno), warn_if_unusable);
        }
    }

    ownerElection(ownerElection &&) = delete;
    ownerElection &
    operator=(ownerElection &&) = delete;

    [[nodiscard]] bool
    won() const noexcept {
        return won_;
    }

private:
    // Every rank then believes it was elected, so those that go on to lose the
    // bind report the port as held from outside the run while a sibling is in
    // fact serving. This is the context that makes those reports readable.
    void
    warnUnusable(const char *reason, bool enabled) const {
        if (!enabled) {
            return;
        }
        NIXL_WARN << "prometheus_mp: cannot use " << lockName_ << " (" << reason
                  << "); falling back to letting the port bind decide which process serves, so "
                  << "a later report of the port being held from outside the run may be a sibling";
    }

    std::string lockName_;
    scopedFd fd_;
    bool won_ = false;
};

/**
 * @brief The addresses of @p dir that some live process is currently elected on,
 *        other than @p address.
 *
 * Ranks that disagree about the port each win their own election, so this is
 * what tells an owner that the directory is served more than once. Lock files
 * outlive their run -- unlinking one would let a rank between open() and flock()
 * lock a file nobody else can find, and two owners would serve one address --
 * so a leftover is told from a live one by trying to take its lock rather than
 * by its existence.
 */
[[nodiscard]] inline std::vector<std::string>
heldAddressesExcept(const std::filesystem::path &dir, const std::string &address) {
    const std::string own = ownerLockFileName(address);
    std::vector<std::string> held;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name == own) {
            continue;
        }
        const std::string other = addressOfLockFile(name);
        if (other.empty()) {
            continue;
        }
        const scopedFd fd(::open(entry.path().c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        if (!fd.valid()) {
            continue;
        }
        if (::flock(fd.get(), LOCK_EX | LOCK_NB) == 0) {
            continue;
        }
        if (errno == EWOULDBLOCK) {
            held.push_back(other);
        }
    }
    return held;
}

} // namespace nixl::telemetry::mp

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_OWNER_ELECTION_H
