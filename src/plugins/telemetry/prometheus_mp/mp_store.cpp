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
#include "mp_store.h"

#include "common/mapped_region.h"
#include "common/nixl_log.h"
#include "common/scoped_fd.h"
#include "mp_store_layout.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace nixl::telemetry::mp {

namespace {

    // A store owned by another uid is never reaped, so the same file would warn on
    // every scrape for the life of the process. Bounded so a directory full of
    // planted files cannot grow this without limit.
    [[nodiscard]] bool
    firstSightOf(const std::filesystem::path &path) {
        constexpr std::size_t max_remembered = 64;
        static std::mutex mutex;
        static std::set<std::filesystem::path> seen;
        const std::lock_guard<std::mutex> lock(mutex);
        return seen.size() < max_remembered && seen.insert(path).second;
    }

    // A store being created: nameless where the filesystem allows it, otherwise
    // under a staging name no reader looks at. Either way it is invisible as a
    // store until publishStore() links it in.
    struct stagedStore {
        scopedFd fd;
        // Empty when the descriptor has no name at all.
        std::filesystem::path stagingPath;
    };

    [[nodiscard]] stagedStore
    stageStore(const std::filesystem::path &path) {
        scopedFd nameless(
            ::open(path.parent_path().c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, S_IRUSR | S_IWUSR));
        if (nameless.valid()) {
            return {std::move(nameless), {}};
        }
        // No O_TMPFILE (NFS, or a kernel older than 3.11). The staging name is
        // unique to this process and instance, so O_EXCL can only collide with a
        // leftover from a process that had this pid and run marker -- and if one
        // somehow exists, refusing to adopt a file of unknown provenance is the
        // right answer.
        std::filesystem::path staging = path;
        staging += MP_STORE_STAGING_SUFFIX;
        return {scopedFd(::open(staging.c_str(),
                                O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW,
                                S_IRUSR | S_IWUSR)),
                std::move(staging)};
    }

    // Gives the initialized, locked store its final name, at which point readers
    // can find it. AT_SYMLINK_FOLLOW resolves the /proc/self/fd symlink to the
    // nameless inode, which is the unprivileged way to link one in (the
    // AT_EMPTY_PATH form needs CAP_DAC_READ_SEARCH).
    void
    publishStore(int fd, const std::filesystem::path &staging, const std::filesystem::path &path) {
        if (!staging.empty()) {
            if (::rename(staging.c_str(), path.c_str()) != 0) {
                throw std::runtime_error("prometheus_mp: cannot publish telemetry store '" +
                                         path.string() + "': " + std::strerror(errno));
            }
            return;
        }
        const std::string proc_path = "/proc/self/fd/" + std::to_string(fd);
        if (::linkat(AT_FDCWD, proc_path.c_str(), AT_FDCWD, path.c_str(), AT_SYMLINK_FOLLOW) != 0) {
            throw std::runtime_error("prometheus_mp: cannot publish telemetry store '" +
                                     path.string() + "': " + std::strerror(errno));
        }
    }

} // namespace

std::string
makeStoreFileName(int64_t pid, uint64_t run_marker, uint64_t instance) {
    return std::string(MP_STORE_FILE_PREFIX) + std::to_string(pid) + "." +
        std::to_string(run_marker) + "." + std::to_string(instance) +
        std::string(MP_STORE_FILE_SUFFIX);
}

uint64_t
processRunMarker() noexcept {
    static const uint64_t marker = [] {
        struct timespec ts{};
        ::clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
    }();
    return marker;
}

storeWriter::storeWriter(std::filesystem::path path,
                         const std::string &agent_name,
                         const std::string &hostname,
                         const std::string &local_rank,
                         uint64_t instance,
                         const std::vector<double> &histogram_buckets)
    : path_(std::move(path)) {
    if (histogram_buckets.size() > MP_STORE_MAX_BUCKETS) {
        throw std::runtime_error("prometheus_mp: NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US has " +
                                 std::to_string(histogram_buckets.size()) +
                                 " bounds, more than the " + std::to_string(MP_STORE_MAX_BUCKETS) +
                                 " a multi-process store can hold");
    }

    auto staged = stageStore(path_);
    if (!staged.fd.valid()) {
        throw std::runtime_error("prometheus_mp: cannot create telemetry store '" + path_.string() +
                                 "': " + std::strerror(errno));
    }

    // Not fatal: the store is still written and scraped, only its liveness stops
    // being provable. Which way that degrades depends on whether readers can lock
    // it either, so the warning states both outcomes rather than one.
    if (::flock(staged.fd.get(), LOCK_EX | LOCK_NB) != 0) {
        NIXL_WARN << "prometheus_mp: cannot lock telemetry store '" << path_.string() << "' ("
                  << std::strerror(errno)
                  << "); no reader can tell whether this process is alive. One that can lock the "
                  << "store reads it as abandoned, so this rank's series are dropped and its file "
                  << "reaped if it stops exporting for longer than the stale TTL, while it is "
                  << "still running; where no process can lock at all, nothing is reaped instead "
                  << "and departed ranks stay published indefinitely";
    }

    if (::ftruncate(staged.fd.get(), static_cast<off_t>(sizeof(storeLayout))) != 0) {
        throw std::runtime_error("prometheus_mp: cannot size telemetry store '" + path_.string() +
                                 "': " + std::strerror(errno));
    }

    mappedRegion mapping(staged.fd.get(), sizeof(storeLayout), PROT_READ | PROT_WRITE, MAP_SHARED);
    if (!mapping.valid()) {
        throw std::runtime_error("prometheus_mp: cannot map telemetry store '" + path_.string() +
                                 "': " + std::strerror(errno));
    }

    auto *layout = static_cast<storeLayout *>(mapping.get());
    std::memset(layout, 0, mapping.size());
    layout->schemaVersion = MP_STORE_SCHEMA_VERSION;
    layout->slotCount = static_cast<uint32_t>(MP_STORE_SLOT_COUNT);
    layout->pid = static_cast<int64_t>(::getpid());
    layout->instance = instance;
    layout->bucketCount = histogram_buckets.size();
    std::copy(histogram_buckets.begin(), histogram_buckets.end(), layout->bucketBounds);
    copyField(layout->agentName, MP_MAX_AGENT_NAME, agent_name, "agent name");
    copyField(layout->hostname, MP_MAX_HOSTNAME, hostname, "hostname");
    copyField(layout->localRank, MP_MAX_LOCAL_RANK, local_rank, "local_rank");
    __atomic_store_n(&layout->lastUpdateNs, monotonicNs(), __ATOMIC_RELAXED);
    __atomic_store_n(&layout->magic, MP_STORE_MAGIC, __ATOMIC_RELEASE);

    publishStore(staged.fd.get(), staged.stagingPath, path_);
    fd_ = std::move(staged.fd);
    mapping_ = std::move(mapping);
}

bool
storeWriterAlive(const std::filesystem::path &path) {
    const scopedFd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!fd.valid()) {
        // Nothing was probed, so nothing is concluded: whatever stopped the open
        // is for the next operation on the path to report.
        return true;
    }
    // Taking the lock proves nobody holds it; closing the descriptor gives it
    // straight back, so nothing is kept from a writer that never existed.
    return ::flock(fd.get(), LOCK_EX | LOCK_NB) != 0;
}

uint64_t
storeWriter::refreshHeartbeat() noexcept {
    auto *layout = static_cast<storeLayout *>(mapping_.get());
    const uint64_t now = monotonicNs();
    __atomic_store_n(&layout->lastUpdateNs, now, __ATOMIC_RELAXED);
    return now;
}

void
storeWriter::addCounter(nixl_telemetry_event_type_t type, uint64_t delta) noexcept {
    const auto idx = static_cast<std::size_t>(type);
    if (idx >= MP_STORE_SLOT_COUNT) {
        return;
    }
    auto *layout = static_cast<storeLayout *>(mapping_.get());
    __atomic_fetch_add(&layout->counters[idx], delta, __ATOMIC_RELAXED);
}

void
storeWriter::setGauge(nixl_telemetry_event_type_t type, uint64_t value) noexcept {
    const auto idx = static_cast<std::size_t>(type);
    if (idx >= MP_STORE_SLOT_COUNT) {
        return;
    }
    auto *layout = static_cast<storeLayout *>(mapping_.get());
    __atomic_store_n(&layout->gauges[idx], value, __ATOMIC_RELAXED);
}

void
storeWriter::observeHistogram(nixl_telemetry_event_type_t type, uint64_t value) noexcept {
    const auto idx = static_cast<std::size_t>(type);
    // Same predicate the reader uses to build MP_STORE_HISTOGRAM_SLOTS, so a slot
    // can never be written without being read back.
    if (idx >= MP_STORE_SLOT_COUNT ||
        nixlEnumStrings::telemetryMetricDescriptor(type).histogramName == nullptr) {
        return;
    }
    auto *layout = static_cast<storeLayout *>(mapping_.get());
    const double *const first = layout->bucketBounds;
    const double *const last = first + layout->bucketCount;
    // lower_bound, not upper_bound: Prometheus buckets are `value <= le`, so the
    // observation belongs in the first bucket whose bound is not below it. Values
    // above every bound land in the trailing overflow slot.
    const double *const bound = std::lower_bound(first, last, static_cast<double>(value));
    __atomic_fetch_add(
        &layout->histBuckets[idx][static_cast<std::size_t>(bound - first)], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&layout->histSums[idx], value, __ATOMIC_RELAXED);
}

storeReadResult
readStoreSnapshot(const std::filesystem::path &path) {
    const scopedFd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (fd.get() < 0) {
        // Missing or transiently unreadable (e.g. EMFILE): not necessarily an
        // orphan, so leave contentInvalid false and the collector never reaps a
        // live peer we simply failed to open.
        return {std::nullopt, false};
    }

    struct stat st{};
    if (::fstat(fd.get(), &st) != 0 || static_cast<std::size_t>(st.st_size) < sizeof(storeLayout)) {
        // Too small: a truncated/mid-creation store -- unusable content.
        return {std::nullopt, true};
    }

    if (st.st_uid != ::geteuid()) {
        // Someone else's file in a shared directory: its contents are attacker-
        // controlled, and it is not ours to reap either.
        if (firstSightOf(path)) {
            NIXL_WARN << "prometheus_mp: ignoring telemetry store '" << path.string()
                      << "' owned by uid " << st.st_uid
                      << "; it cannot be reaped, so this is reported once";
        }
        return {std::nullopt, false};
    }

    const mappedRegion mapping(fd.get(), sizeof(storeLayout), PROT_READ, MAP_SHARED);
    if (!mapping.valid()) {
        // Transient (e.g. ENOMEM): the file may be a healthy peer's, so do not
        // mark it reapable.
        NIXL_WARN << "prometheus_mp: cannot map telemetry store '" << path.string()
                  << "': " << std::strerror(errno);
        return {std::nullopt, false};
    }

    const auto *layout = static_cast<const storeLayout *>(mapping.get());

    const uint64_t magic = __atomic_load_n(&layout->magic, __ATOMIC_ACQUIRE);
    if (magic == 0) {
        // A store is named only once it is initialized, so a zeroed header is not
        // a writer mid-creation: it is a file somebody else left here. Quiet (no
        // WARN) because the collector reaps it once nobody holds it.
        return {std::nullopt, true};
    }
    if (magic != MP_STORE_MAGIC) {
        NIXL_WARN << "prometheus_mp: ignoring telemetry store '" << path.string()
                  << "' with bad magic";
        return {std::nullopt, true};
    }
    if (layout->schemaVersion != MP_STORE_SCHEMA_VERSION ||
        layout->slotCount != MP_STORE_SLOT_COUNT || layout->bucketCount > MP_STORE_MAX_BUCKETS) {
        NIXL_WARN << "prometheus_mp: ignoring telemetry store '" << path.string()
                  << "' with incompatible schema (version " << layout->schemaVersion << ", slots "
                  << layout->slotCount << ", buckets " << layout->bucketCount << ")";
        return {std::nullopt, true};
    }

    storeSnapshot snap;
    snap.pid = layout->pid;
    snap.instance = layout->instance;
    snap.lastUpdateNs = __atomic_load_n(&layout->lastUpdateNs, __ATOMIC_ACQUIRE);
    snap.agentName = readField(layout->agentName, MP_MAX_AGENT_NAME);
    snap.hostname = readField(layout->hostname, MP_MAX_HOSTNAME);
    snap.localRank = readField(layout->localRank, MP_MAX_LOCAL_RANK);
    snap.bucketCount = static_cast<uint32_t>(layout->bucketCount);
    std::copy_n(layout->bucketBounds, snap.bucketCount, snap.bucketBounds.begin());
    for (std::size_t i = 0; i < MP_STORE_SLOT_COUNT; ++i) {
        snap.counters[i] = __atomic_load_n(&layout->counters[i], __ATOMIC_RELAXED);
        snap.gauges[i] = __atomic_load_n(&layout->gauges[i], __ATOMIC_RELAXED);
    }
    for (const auto i : MP_STORE_HISTOGRAM_SLOTS) {
        snap.histSums[i] = __atomic_load_n(&layout->histSums[i], __ATOMIC_RELAXED);
        for (std::size_t b = 0; b <= snap.bucketCount; ++b) {
            snap.histBuckets[i][b] = __atomic_load_n(&layout->histBuckets[i][b], __ATOMIC_RELAXED);
        }
    }

    return {std::move(snap), false};
}

} // namespace nixl::telemetry::mp
