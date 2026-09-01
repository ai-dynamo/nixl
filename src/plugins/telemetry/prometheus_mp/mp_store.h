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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_STORE_H
#define NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_STORE_H

#include "common/mapped_region.h"
#include "common/scoped_fd.h"
#include "telemetry_event.h"

#include <ctime>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nixl::telemetry::mp {

// On-disk schema version for the per-process metric-state store. Independent of
// the event-buffer TELEMETRY_VERSION: this is a different file format. Bump on
// any layout change so a reader rejects incompatible files.
inline constexpr uint32_t MP_STORE_SCHEMA_VERSION = 1;

// Store file naming shared by the writer (exporter) and the collector so the
// collector can discover peer files by globbing "<prefix>*<suffix>".
inline constexpr std::string_view MP_STORE_FILE_PREFIX = "nixl.";
inline constexpr std::string_view MP_STORE_FILE_SUFFIX = ".mmap";

// Name a store carries only while it is being created, on the filesystems that
// cannot hold a nameless one (see storeWriter). Outside the <prefix>*<suffix>
// pattern, so the collector never reads or reaps a store mid-creation.
inline constexpr std::string_view MP_STORE_STAGING_SUFFIX = ".staging";

// Number of value slots in each of the counter and gauge arrays. Indexed
// directly by nixl_telemetry_event_type_t, so every event type has a reserved
// counter slot and a reserved gauge slot (unused ones stay zero). Derived from
// the highest enum value (AGENT_TELEMETRY_EVENTS_DROPPED must stay last); keep in
// sync if the enum is extended.
inline constexpr std::size_t MP_STORE_SLOT_COUNT =
    static_cast<std::size_t>(nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED) + 1;

// Most histogram bucket boundaries the fixed layout can hold (the built-in set
// uses 18). A NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US override with more bounds than
// this is rejected rather than silently truncated -- the single-process exporter,
// which keeps its buckets on the heap, has no such limit.
inline constexpr std::size_t MP_STORE_MAX_BUCKETS = 32;

/**
 * @brief Reads the clock the store's timestamps are expressed in.
 *
 * CLOCK_MONOTONIC directly, not nixlTime::getNs(): a store timestamp is written
 * by one process and compared by another, so the epoch has to be the kernel's
 * boot time rather than whatever epoch an implementation's steady_clock happens
 * to use. This is part of the on-disk contract.
 */
[[nodiscard]] inline uint64_t
monotonicNs() noexcept {
    struct timespec ts{};
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

namespace detail {
    // Highest slot index the collector will index into the counter/gauge arrays,
    // across every event type it publishes.
    [[nodiscard]] constexpr std::size_t
    maxTelemetrySlot() {
        std::size_t max_slot = 0;
        for (const auto type : telemetry_metric_event_types) {
            max_slot = std::max(max_slot, static_cast<std::size_t>(type));
        }
        for (const auto type : telemetry_error_event_types) {
            max_slot = std::max(max_slot, static_cast<std::size_t>(type));
        }
        return max_slot;
    }

    [[nodiscard]] constexpr std::size_t
    histogramSlotCount() {
        std::size_t count = 0;
        for (const auto type : telemetry_metric_event_types) {
            if (nixlEnumStrings::telemetryMetricDescriptor(type).histogramName != nullptr) {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] constexpr std::array<std::size_t, histogramSlotCount()>
    histogramSlots() {
        std::array<std::size_t, histogramSlotCount()> slots{};
        std::size_t next = 0;
        for (const auto type : telemetry_metric_event_types) {
            if (nixlEnumStrings::telemetryMetricDescriptor(type).histogramName != nullptr) {
                slots[next++] = static_cast<std::size_t>(type);
            }
        }
        return slots;
    }
} // namespace detail

// The slots that can ever hold histogram data, derived at compile time from the
// shared descriptor. Every other slot is provably zero, so the reader skips it
// instead of loading MP_STORE_MAX_BUCKETS + 2 words per slot.
inline constexpr auto MP_STORE_HISTOGRAM_SLOTS = detail::histogramSlots();

// Compile-time guard: if the enum is extended past AGENT_TELEMETRY_EVENTS_DROPPED
// (so it is no longer last), the fixed-slot store would be indexed out of bounds
// by the collector. Fail the build instead, forcing MP_STORE_SLOT_COUNT to be
// updated.
static_assert(detail::maxTelemetrySlot() < MP_STORE_SLOT_COUNT,
              "MP_STORE_SLOT_COUNT must cover every telemetry event type the collector indexes; "
              "keep AGENT_TELEMETRY_EVENTS_DROPPED last or update MP_STORE_SLOT_COUNT");

/**
 * @brief A point-in-time copy of one process's metric-state store file.
 *
 * Produced by readStoreSnapshot(). Values are plain (already loaded from the
 * shared file); @c counters are cumulative running totals and @c gauges hold the
 * last-operation value, both indexed by nixl_telemetry_event_type_t.
 */
struct storeSnapshot {
    int64_t pid = 0;
    // Monotonic nanoseconds (monotonicNs(), CLOCK_MONOTONIC -- host-wide, so
    // comparable across processes) of the last writer update; used for TTL staleness.
    uint64_t lastUpdateNs = 0;
    // Per-process instance counter, distinguishing multiple agents that share the
    // same name (and thus pid/hostname) within one process so their series differ.
    uint64_t instance = 0;
    std::string agentName;
    std::string hostname;
    // Optional local (per-GPU/TP) rank label; empty when no rank env was set.
    std::string localRank;
    std::array<uint64_t, MP_STORE_SLOT_COUNT> counters{};
    std::array<uint64_t, MP_STORE_SLOT_COUNT> gauges{};
    // Histogram bucket upper bounds this process was configured with; only the
    // first bucketCount entries are meaningful. Carried per store because each
    // process resolves them from its own environment.
    uint32_t bucketCount = 0;
    std::array<double, MP_STORE_MAX_BUCKETS> bucketBounds{};
    // Per-bucket observation counts, indexed by nixl_telemetry_event_type_t then
    // by bucket, where index bucketCount collects values above the last bound.
    // Deliberately NOT cumulative: the reader accumulates, so a scrape racing a
    // writer can never expose non-monotonic buckets.
    std::array<std::array<uint64_t, MP_STORE_MAX_BUCKETS + 1>, MP_STORE_SLOT_COUNT> histBuckets{};
    // Sum of all observed values per event type, in the event's own units.
    std::array<uint64_t, MP_STORE_SLOT_COUNT> histSums{};
};

/**
 * @class storeWriter
 * @brief Owns one process's metric-state mmap file and updates it in place.
 *
 * Each NIXL process (writer or exporter mode) owns exactly one store. Updates
 * are lock-free atomic operations directly on the mapped file, so the exporter
 * process can read peers' files concurrently without coordination. The file has
 * a fixed size (fixed slot layout), so it never needs to grow.
 *
 * The store is flock-ed before it is given a name and stays locked for as long
 * as this object lives, which is what lets a reader tell a live writer from a
 * dead one by trying the lock alone -- see storeWriterAlive(). Creating it
 * nameless (O_TMPFILE) and linking it in afterwards is what makes that "before"
 * airtight: no reader can ever find a store that is initialized but unlocked, so
 * there is no window in which a live writer's store looks abandoned. A
 * filesystem without O_TMPFILE gets a MP_STORE_STAGING_SUFFIX name instead,
 * renamed into place once locked and initialized; a process killed inside that
 * window leaks one staging file, which is the price of not having O_TMPFILE.
 */
class storeWriter {
public:
    /**
     * @brief Creates (or truncates) and maps the store file at @p path.
     * @param path Full path to this process's store file.
     * @param agent_name Per-process agent name (unique; drives the series label).
     * @param hostname Host name label.
     * @param local_rank Optional rank label; pass empty to omit it.
     * @param instance Per-process instance counter; disambiguates multiple
     *        same-named agents in one process so their series stay distinct.
     * @param histogram_buckets Histogram bucket upper bounds, at most
     *        MP_STORE_MAX_BUCKETS of them.
     * @throws std::runtime_error on create/ftruncate/mmap/publish failure, or
     *         when @p histogram_buckets does not fit the store.
     */
    storeWriter(std::filesystem::path path,
                const std::string &agent_name,
                const std::string &hostname,
                const std::string &local_rank,
                uint64_t instance,
                const std::vector<double> &histogram_buckets);

    /**
     * @brief Unmaps the store and releases its lock, leaving the file in place.
     *
     * The last values a process recorded are typically not scraped yet when it
     * exits, so the file must outlive it: releasing the lock is what tells the
     * collector to go on publishing them until they age past the stale TTL, and
     * to reap the file then. Unlinking here would drop everything produced since
     * the previous scrape.
     */
    ~storeWriter() = default;

    storeWriter(const storeWriter &) = delete;
    storeWriter &
    operator=(const storeWriter &) = delete;
    storeWriter(storeWriter &&) = delete;
    storeWriter &
    operator=(storeWriter &&) = delete;

    // Adds @p delta to the cumulative counter slot for @p type. Out-of-range
    // types are ignored.
    void
    addCounter(nixl_telemetry_event_type_t type, uint64_t delta) noexcept;

    // Stores @p value as the last-operation gauge for @p type. Out-of-range
    // types are ignored.
    void
    setGauge(nixl_telemetry_event_type_t type, uint64_t value) noexcept;

    // Records @p value in the histogram bucket it falls into (upper bounds are
    // inclusive, matching Prometheus) and adds it to the sum for @p type.
    // Out-of-range types are ignored.
    void
    observeHistogram(nixl_telemetry_event_type_t type, uint64_t value) noexcept;

    /**
     * @brief Republishes this process's liveness timestamp.
     * @return The timestamp written, so a caller needing the current time (the
     *         exporter throttles its re-elections on it) reads the clock once.
     *
     * Deliberately separate from the mutators: it costs a CLOCK_MONOTONIC read
     * (~24ns, several times a metric update's atomic), and the only consumer is
     * the collector's staleness check, which has a 30s default TTL and applies
     * to dead processes only. The exporter therefore refreshes once per event,
     * after the slot updates, instead of from each mutator -- a duration event
     * touches three slots. Skipping refreshes never hides a live process, it
     * only shortens how long a dead one's last values linger.
     */
    [[nodiscard]] uint64_t
    refreshHeartbeat() noexcept;

    [[nodiscard]] const std::filesystem::path &
    path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
    // Held open for the writer's lifetime: closing it releases the lock, which is
    // how every reader learns this process is done with the store.
    scopedFd fd_;
    mappedRegion mapping_;
};

/**
 * @brief Whether the process that created the store at @p path still holds it.
 *
 * A store is locked before it has a name and unlocked only when its writer dies
 * or destroys it, and no process ever locks a store it did not create. So a lock
 * this succeeds in taking means the writer is gone for good and the file will
 * never change again -- which is the whole liveness test, with no pids, no
 * /proc, and no need to share a PID namespace with the writer.
 *
 * Conservative on failure: a store that cannot be opened or locked for any other
 * reason (a filesystem without flock, EMFILE) reads as held, so nothing is ever
 * reaped on the strength of a probe that did not work.
 */
[[nodiscard]] bool
storeWriterAlive(const std::filesystem::path &path);

/**
 * @brief Outcome of readStoreSnapshot().
 *
 * @c contentInvalid is true only when the file was read and its content is
 * unusable (too small, bad/zero magic, incompatible schema) -- i.e. a genuine
 * orphan safe to reap. It stays false when the file could not be opened or
 * mapped (missing, or a transient error such as EMFILE/ENOMEM), so a live peer
 * is never mistaken for an orphan.
 */
struct storeReadResult {
    std::optional<storeSnapshot> snapshot;
    bool contentInvalid = false;
};

/**
 * @brief Reads a consistent snapshot of a store file written by another process.
 * @param path Path to a peer's store file.
 * @return The snapshot when the store was read and is compatible, otherwise an
 *         empty snapshot and the reason category (see storeReadResult).
 */
[[nodiscard]] storeReadResult
readStoreSnapshot(const std::filesystem::path &path);

/**
 * @brief A value fixed for the life of this process.
 * @return The same value on every call, in this process.
 *
 * Only two runs sharing a pid have to tell their store files apart, so any
 * per-process constant serves: this is the wall clock at first call.
 */
[[nodiscard]] uint64_t
processRunMarker() noexcept;

/**
 * @brief Builds a store file name (MP_STORE_FILE_PREFIX / suffix).
 * @param pid Process id.
 * @param run_marker processRunMarker() of the writing process (disambiguates PID
 *        reuse across restarts).
 * @param instance Per-process instance counter (disambiguates multiple agents in
 *        the same process so their store files never collide).
 * @return File name of the form "nixl.<pid>.<run_marker>.<instance>.mmap".
 */
[[nodiscard]] std::string
makeStoreFileName(int64_t pid, uint64_t run_marker, uint64_t instance);

} // namespace nixl::telemetry::mp

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_STORE_H
