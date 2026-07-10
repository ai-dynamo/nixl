/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "telemetry_event.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace nixl::telemetry::mp {

// On-disk schema version for the per-process metric-state store. Independent of
// the event-buffer TELEMETRY_VERSION: this is a different file format. Bump on
// any layout change so a reader rejects incompatible files.
inline constexpr uint32_t MP_STORE_SCHEMA_VERSION = 1;

// Store file naming shared by the writer (exporter) and the collector so the
// collector can discover peer files by globbing "<prefix>*<suffix>".
inline constexpr std::string_view MP_STORE_FILE_PREFIX = "nixl.";
inline constexpr std::string_view MP_STORE_FILE_SUFFIX = ".mmap";

// Number of value slots in each of the counter and gauge arrays. Indexed
// directly by nixl_telemetry_event_type_t, so every event type has a reserved
// counter slot and a reserved gauge slot (unused ones stay zero). Derived from
// the highest enum value (AGENT_TELEMETRY_EVENTS_DROPPED must stay last); keep in
// sync if the enum is extended.
inline constexpr std::size_t MP_STORE_SLOT_COUNT =
    static_cast<std::size_t>(nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED) + 1;

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
} // namespace detail

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
struct mpStoreSnapshot {
    int64_t pid = 0;
    // Process start time in clock ticks (/proc/<pid>/stat field 22); pairs with
    // pid to survive PID reuse when checking liveness.
    uint64_t startTime = 0;
    // Monotonic nanoseconds (nixlTime::getNs, CLOCK_MONOTONIC -- host-wide, so
    // comparable across processes) of the last writer update; used for TTL staleness.
    uint64_t lastUpdateNs = 0;
    std::string agentName;
    std::string hostname;
    // Optional local (per-GPU/TP) rank label; empty when no rank env was set.
    std::string localRank;
    std::array<uint64_t, MP_STORE_SLOT_COUNT> counters{};
    std::array<uint64_t, MP_STORE_SLOT_COUNT> gauges{};
};

/**
 * @class mpStoreWriter
 * @brief Owns one process's metric-state mmap file and updates it in place.
 *
 * Each NIXL process (writer or exporter mode) owns exactly one store. Updates
 * are lock-free atomic operations directly on the mapped file, so the exporter
 * process can read peers' files concurrently without coordination. The file has
 * a fixed size (fixed slot layout), so it never needs to grow.
 */
class mpStoreWriter {
public:
    /**
     * @brief Creates (or truncates) and maps the store file at @p path.
     * @param path Full path to this process's store file.
     * @param agent_name Per-process agent name (unique; drives the series label).
     * @param hostname Host name label.
     * @param local_rank Optional rank label; pass empty to omit it.
     * @throws std::runtime_error on open/ftruncate/mmap failure.
     */
    mpStoreWriter(std::filesystem::path path,
                  const std::string &agent_name,
                  const std::string &hostname,
                  const std::string &local_rank);
    ~mpStoreWriter();

    mpStoreWriter(const mpStoreWriter &) = delete;
    mpStoreWriter &
    operator=(const mpStoreWriter &) = delete;
    mpStoreWriter(mpStoreWriter &&) = delete;
    mpStoreWriter &
    operator=(mpStoreWriter &&) = delete;

    // Adds @p delta to the cumulative counter slot for @p type and refreshes the
    // heartbeat. Out-of-range types are ignored.
    void
    addCounter(nixl_telemetry_event_type_t type, uint64_t delta) noexcept;

    // Stores @p value as the last-operation gauge for @p type and refreshes the
    // heartbeat. Out-of-range types are ignored.
    void
    setGauge(nixl_telemetry_event_type_t type, uint64_t value) noexcept;

    // Refreshes the last-update timestamp without changing any metric (keeps the
    // process from looking stale during idle periods).
    void
    refreshHeartbeat() noexcept;

    [[nodiscard]] const std::filesystem::path &
    path() const noexcept {
        return path_;
    }

private:
    void
    touch() noexcept;

    std::filesystem::path path_;
    void *mapping_ = nullptr;
    std::size_t mappingSize_ = 0;
};

/**
 * @brief Reads a consistent snapshot of a store file written by another process.
 * @param path Path to a peer's store file.
 * @return The snapshot, or std::nullopt if the file is missing, too small, or
 *         has a bad magic / incompatible schema version (a WARN is logged for a
 *         present-but-invalid file).
 */
[[nodiscard]] std::optional<mpStoreSnapshot>
readStoreSnapshot(const std::filesystem::path &path);

/**
 * @brief Reads a process's start time (/proc/<pid>/stat field 22, clock ticks).
 * @param pid Process id.
 * @return The start time, or 0 if it could not be read.
 */
[[nodiscard]] uint64_t
readProcessStartTime(int64_t pid);

/**
 * @brief Builds a store file name (MP_STORE_FILE_PREFIX / suffix).
 * @param pid Process id.
 * @param start_time Process start time (disambiguates PID reuse across restarts).
 * @param instance Per-process instance counter (disambiguates multiple agents in
 *        the same process so their store files never collide).
 * @return File name of the form "nixl.<pid>.<start_time>.<instance>.mmap".
 */
[[nodiscard]] std::string
makeStoreFileName(int64_t pid, uint64_t start_time, uint64_t instance);

} // namespace nixl::telemetry::mp

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_STORE_H
