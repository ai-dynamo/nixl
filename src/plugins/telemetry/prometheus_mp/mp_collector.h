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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_COLLECTOR_H
#define NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_COLLECTOR_H

#include "mp_store.h"

#include <prometheus/collectable.h>
#include <prometheus/metric_family.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace nixl::telemetry::mp {

// Default staleness window: a store whose owning process is gone and whose last
// update is older than this is dropped (and reaped) by the collector.
inline constexpr std::chrono::seconds MP_DEFAULT_STALE_TTL{30};

/**
 * @brief Whether a store snapshot may still be published.
 * @param snap Store snapshot.
 * @param ttl Freshness window for a store whose writer is gone.
 * @param writer_alive Whether the writer still holds the store
 *        (storeWriterAlive()).
 *
 * A live writer is always published. A departed one is kept until its last
 * values age out of @p ttl, and published meanwhile unless another store with
 * its hostname, agent_name and local_rank is live or was updated more recently.
 */
[[nodiscard]] bool
isSnapshotLive(const storeSnapshot &snap, std::chrono::nanoseconds ttl, bool writer_alive);

/** @brief A store that survived the scan, with whether its writer still holds it. */
struct scannedStore {
    storeSnapshot snapshot;
    bool writerAlive = false;
};

/** @brief A store selected for a scrape; @c withProcessLabels adds pid and agent_instance. */
struct publishedStore {
    storeSnapshot snapshot;
    bool withProcessLabels = false;
};

/**
 * @brief Converts per-process snapshots into Prometheus metric families.
 *
 * Emits one series per (metric, process): cumulative counters and last-operation
 * gauges keyed by nixl_telemetry_event_type_t, plus the agent_errors_total family
 * with a status label. Series are labeled by hostname, agent_name and (when
 * present) local_rank, plus pid and agent_instance for stores with
 * @c withProcessLabels set. No cross-process aggregation -- each process is its
 * own series. Returns empty when @p stores is empty.
 * @param stores Live per-process snapshots.
 */
[[nodiscard]] std::vector<prometheus::MetricFamily>
buildMetricFamilies(const std::vector<publishedStore> &stores);

/**
 * @class nixlMultiprocessCollector
 * @brief prometheus-cpp Collectable that aggregates all peer store files.
 *
 * Registered with the exporter process's Exposer. On each scrape it globs the
 * shared directory for store files, reads a snapshot of each, drops (and
 * optionally reaps) stale ones, picks the ones to publish (selectStores()), and
 * returns their metric families.
 */
class nixlMultiprocessCollector final : public prometheus::Collectable {
public:
    /**
     * @param dir Shared telemetry directory holding peer store files.
     * @param stale_ttl Freshness window for a process that has exited.
     * @param reap_stale When true, unlink store files that no writer holds any
     *        more and whose data is older than @p stale_ttl.
     */
    explicit nixlMultiprocessCollector(std::filesystem::path dir,
                                       std::chrono::nanoseconds stale_ttl = MP_DEFAULT_STALE_TTL,
                                       bool reap_stale = true);

    [[nodiscard]] std::vector<prometheus::MetricFamily>
    Collect() const override;

    /**
     * @brief Picks the stores one scrape publishes.
     * @param stores The stores of one scan, in any order.
     *
     * Groups @p stores by hostname, agent_name and local_rank. A group with at
     * most one live writer yields one store: the live one, else the most recently
     * updated, ties going to the higher pid, then the higher instance. A group
     * with two or more live writers yields every live store with
     * @c withProcessLabels set. Only the first such group a collector sees is
     * reported.
     */
    [[nodiscard]] std::vector<publishedStore>
    selectStores(std::vector<scannedStore> stores) const;

private:
    // Scans the shared directory once, dropping (and optionally reaping) stores
    // that are stale or unusable.
    [[nodiscard]] std::vector<scannedStore>
    scanLiveStores() const;

    void
    reportSharedIdentity(const storeSnapshot &store, std::size_t live_writers) const;

    const std::filesystem::path dir_;
    const std::chrono::nanoseconds staleTtl_;
    const bool reapStale_;
    mutable std::atomic_flag duplicateReported_;
};

} // namespace nixl::telemetry::mp

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_COLLECTOR_H
