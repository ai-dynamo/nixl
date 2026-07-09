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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_COLLECTOR_H
#define NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_COLLECTOR_H

#include "mp_store.h"

#include <prometheus/collectable.h>
#include <prometheus/metric_family.h>

#include <chrono>
#include <filesystem>
#include <vector>

namespace nixl::telemetry::mp {

// Default staleness window: a store whose owning process is gone and whose last
// update is older than this is dropped (and reaped) by the collector.
inline constexpr std::chrono::seconds MP_DEFAULT_STALE_TTL{30};

/**
 * @brief Whether a process is still running, guarding against PID reuse.
 * @param pid Process id from a store header.
 * @param start_time Process start time from the same header (0 = unknown/skip check).
 * @return True if the pid exists and (when known) its start time still matches.
 */
[[nodiscard]] bool
isProcessAlive(int64_t pid, uint64_t start_time);

/**
 * @brief Whether a store snapshot should still be published.
 *
 * Live if the owning process is alive, or (to smooth over a just-exited process)
 * if its last update is within @p ttl.
 * @param snap Store snapshot.
 * @param ttl Freshness window for a process that is no longer alive.
 */
[[nodiscard]] bool
isSnapshotLive(const mpStoreSnapshot &snap, std::chrono::nanoseconds ttl);

/**
 * @brief Converts per-process snapshots into Prometheus metric families.
 *
 * Emits one series per (metric, process): cumulative counters and last-operation
 * gauges keyed by nixl_telemetry_event_type_t, plus the agent_errors_total family
 * with a status label. Series are labeled by hostname, agent_name and (when
 * present) dp_rank. No cross-process aggregation -- each process is its own
 * series. Returns empty when @p snapshots is empty.
 * @param snapshots Live per-process snapshots.
 */
[[nodiscard]] std::vector<prometheus::MetricFamily>
buildMetricFamilies(const std::vector<mpStoreSnapshot> &snapshots);

/**
 * @class nixlMultiprocessCollector
 * @brief prometheus-cpp Collectable that aggregates all peer store files.
 *
 * Registered with the exporter process's Exposer. On each scrape it globs the
 * shared directory for store files, reads a snapshot of each, drops (and
 * optionally reaps) stale ones, and returns per-process metric families.
 */
class nixlMultiprocessCollector final : public prometheus::Collectable {
public:
    /**
     * @param dir Shared telemetry directory holding peer store files.
     * @param stale_ttl Freshness window for a process that has exited.
     * @param reap_stale When true, unlink store files whose process is gone and
     *        whose data is older than @p stale_ttl.
     */
    explicit nixlMultiprocessCollector(std::filesystem::path dir,
                                       std::chrono::nanoseconds stale_ttl = MP_DEFAULT_STALE_TTL,
                                       bool reap_stale = true);

    [[nodiscard]] std::vector<prometheus::MetricFamily>
    Collect() const override;

private:
    std::filesystem::path dir_;
    std::chrono::nanoseconds staleTtl_;
    bool reapStale_;
};

} // namespace nixl::telemetry::mp

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_COLLECTOR_H
