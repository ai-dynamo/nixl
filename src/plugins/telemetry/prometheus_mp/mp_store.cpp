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

#include "common/nixl_log.h"
#include "common/nixl_time.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace nixl::telemetry::mp {

namespace {

    // "NIXLMPS1" as a little-endian tag; changing the layout must change either this
    // or MP_STORE_SCHEMA_VERSION so stale-format files are rejected.
    constexpr uint64_t MP_STORE_MAGIC = 0x3153504d4c58494eULL;

    constexpr std::size_t MP_MAX_AGENT_NAME = 256;
    constexpr std::size_t MP_MAX_HOSTNAME = 128;
    constexpr std::size_t MP_MAX_LOCAL_RANK = 64;

    // Fixed on-disk layout. Plain trivially-copyable POD operated on with __atomic
    // builtins (not std::atomic) so it is safe to memset/reinterpret over an mmap'd
    // region shared between processes. Field order keeps every uint64 8-byte aligned.
    struct storeLayout {
        uint64_t magic;
        uint32_t schemaVersion;
        uint32_t slotCount;
        int64_t pid;
        uint64_t startTime;
        uint64_t lastUpdateNs;
        uint64_t instance;
        // 64-bit purely so the double array that follows stays 8-byte aligned
        // without implicit padding.
        uint64_t bucketCount;
        char agentName[MP_MAX_AGENT_NAME];
        char hostname[MP_MAX_HOSTNAME];
        char localRank[MP_MAX_LOCAL_RANK];
        double bucketBounds[MP_STORE_MAX_BUCKETS];
        uint64_t counters[MP_STORE_SLOT_COUNT];
        uint64_t gauges[MP_STORE_SLOT_COUNT];
        uint64_t histBuckets[MP_STORE_SLOT_COUNT][MP_STORE_MAX_BUCKETS + 1];
        uint64_t histSums[MP_STORE_SLOT_COUNT];
    };

    void
    copyField(char *dst, std::size_t cap, const std::string &src, const char *what) {
        if (src.size() >= cap) {
            NIXL_WARN << "prometheus_mp: " << what << " '" << src << "' exceeds " << (cap - 1)
                      << " chars; truncating in telemetry store";
        }
        const std::size_t n = std::min(src.size(), cap - 1);
        std::memcpy(dst, src.data(), n);
        dst[n] = '\0';
    }

    [[nodiscard]] std::string
    readField(const char *src, std::size_t cap) {
        const std::size_t n = ::strnlen(src, cap);
        return std::string(src, n);
    }

    class scopedFd {
    public:
        explicit scopedFd(int fd) noexcept : fd_(fd) {}

        ~scopedFd() {
            if (fd_ >= 0) {
                ::close(fd_);
            }
        }

        scopedFd(const scopedFd &) = delete;
        scopedFd &
        operator=(const scopedFd &) = delete;
        scopedFd(scopedFd &&) = delete;
        scopedFd &
        operator=(scopedFd &&) = delete;

        [[nodiscard]] int
        get() const noexcept {
            return fd_;
        }

    private:
        int fd_;
    };

} // namespace

std::string
makeStoreFileName(int64_t pid, uint64_t start_time, uint64_t instance) {
    return std::string(MP_STORE_FILE_PREFIX) + std::to_string(pid) + "." +
        std::to_string(start_time) + "." + std::to_string(instance) +
        std::string(MP_STORE_FILE_SUFFIX);
}

uint64_t
readProcessStartTime(int64_t pid) {
    std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
    if (!stat.is_open()) {
        return 0;
    }
    std::string content((std::istreambuf_iterator<char>(stat)), std::istreambuf_iterator<char>());

    // comm (field 2) is wrapped in parentheses and may itself contain spaces or
    // ')', so split on the LAST ')': everything after it starts at field 3.
    const auto close = content.rfind(')');
    if (close == std::string::npos) {
        return 0;
    }

    std::istringstream rest(content.substr(close + 1));
    std::vector<std::string> tokens{std::istream_iterator<std::string>(rest),
                                    std::istream_iterator<std::string>()};
    // starttime is field 22; tokens[0] is field 3, so index 22 - 3 = 19.
    constexpr std::size_t kStartTimeIndex = 19;
    if (tokens.size() <= kStartTimeIndex) {
        return 0;
    }
    try {
        return static_cast<uint64_t>(std::stoull(tokens[kStartTimeIndex]));
    }
    catch (const std::exception &) {
        return 0;
    }
}

storeWriter::storeWriter(std::filesystem::path path,
                         const std::string &agent_name,
                         const std::string &hostname,
                         const std::string &local_rank,
                         uint64_t instance,
                         const std::vector<double> &histogram_buckets)
    : path_(std::move(path)),
      mappingSize_(sizeof(storeLayout)) {
    if (histogram_buckets.size() > MP_STORE_MAX_BUCKETS) {
        throw std::runtime_error("prometheus_mp: NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US has " +
                                 std::to_string(histogram_buckets.size()) +
                                 " bounds, more than the " + std::to_string(MP_STORE_MAX_BUCKETS) +
                                 " a multi-process store can hold");
    }

    const scopedFd fd(::open(path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (fd.get() < 0) {
        throw std::runtime_error("prometheus_mp: cannot open telemetry store '" + path_.string() +
                                 "': " + std::strerror(errno));
    }

    if (::ftruncate(fd.get(), static_cast<off_t>(mappingSize_)) != 0) {
        throw std::runtime_error("prometheus_mp: cannot size telemetry store '" + path_.string() +
                                 "': " + std::strerror(errno));
    }

    mapping_ = ::mmap(nullptr, mappingSize_, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        throw std::runtime_error("prometheus_mp: cannot map telemetry store '" + path_.string() +
                                 "': " + std::strerror(errno));
    }

    auto *layout = static_cast<storeLayout *>(mapping_);
    std::memset(layout, 0, mappingSize_);
    layout->schemaVersion = MP_STORE_SCHEMA_VERSION;
    layout->slotCount = static_cast<uint32_t>(MP_STORE_SLOT_COUNT);
    layout->pid = static_cast<int64_t>(::getpid());
    layout->startTime = readProcessStartTime(layout->pid);
    layout->instance = instance;
    layout->bucketCount = histogram_buckets.size();
    std::copy(histogram_buckets.begin(), histogram_buckets.end(), layout->bucketBounds);
    copyField(layout->agentName, MP_MAX_AGENT_NAME, agent_name, "agent name");
    copyField(layout->hostname, MP_MAX_HOSTNAME, hostname, "hostname");
    copyField(layout->localRank, MP_MAX_LOCAL_RANK, local_rank, "local_rank");
    __atomic_store_n(&layout->lastUpdateNs, nixlTime::getNs(), __ATOMIC_RELAXED);
    // Publish the magic last so a concurrent reader never validates a
    // half-initialized header.
    __atomic_store_n(&layout->magic, MP_STORE_MAGIC, __ATOMIC_RELEASE);
}

storeWriter::~storeWriter() {
    if (mapping_ != nullptr) {
        ::munmap(mapping_, mappingSize_);
        mapping_ = nullptr;
    }
    std::error_code ec;
    std::filesystem::remove(path_, ec);
}

void
storeWriter::refreshHeartbeat() noexcept {
    auto *layout = static_cast<storeLayout *>(mapping_);
    __atomic_store_n(&layout->lastUpdateNs, nixlTime::getNs(), __ATOMIC_RELAXED);
}

void
storeWriter::addCounter(nixl_telemetry_event_type_t type, uint64_t delta) noexcept {
    const auto idx = static_cast<std::size_t>(type);
    if (idx >= MP_STORE_SLOT_COUNT) {
        return;
    }
    auto *layout = static_cast<storeLayout *>(mapping_);
    __atomic_fetch_add(&layout->counters[idx], delta, __ATOMIC_RELAXED);
}

void
storeWriter::setGauge(nixl_telemetry_event_type_t type, uint64_t value) noexcept {
    const auto idx = static_cast<std::size_t>(type);
    if (idx >= MP_STORE_SLOT_COUNT) {
        return;
    }
    auto *layout = static_cast<storeLayout *>(mapping_);
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
    auto *layout = static_cast<storeLayout *>(mapping_);
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
        NIXL_WARN << "prometheus_mp: ignoring telemetry store '" << path.string()
                  << "' owned by uid " << st.st_uid;
        return {std::nullopt, false};
    }

    void *mapping = ::mmap(nullptr, sizeof(storeLayout), PROT_READ, MAP_SHARED, fd.get(), 0);
    if (mapping == MAP_FAILED) {
        // Transient (e.g. ENOMEM): the file may be a healthy peer's, so do not
        // mark it reapable.
        NIXL_WARN << "prometheus_mp: cannot map telemetry store '" << path.string()
                  << "': " << std::strerror(errno);
        return {std::nullopt, false};
    }

    const std::unique_ptr<void, void (*)(void *)> guard(
        mapping, [](void *p) noexcept { ::munmap(p, sizeof(storeLayout)); });

    const auto *layout = static_cast<const storeLayout *>(mapping);

    const uint64_t magic = __atomic_load_n(&layout->magic, __ATOMIC_ACQUIRE);
    if (magic == 0) {
        // Zeroed header: either a store still being initialized by a live process,
        // or an orphan left by a process that died mid-creation. Skip quietly (no
        // WARN); the collector reaps stale orphans by file age.
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
    snap.startTime = layout->startTime;
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
