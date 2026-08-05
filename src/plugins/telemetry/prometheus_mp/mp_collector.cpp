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
#include "mp_collector.h"

#include "common/nixl_log.h"
#include "telemetry_event.h"

#include <prometheus/client_metric.h>
#include <prometheus/metric_type.h>

#include <chrono>
#include <exception>
#include <limits>
#include <system_error>

namespace nixl::telemetry::mp {

namespace {

    using prometheus::ClientMetric;
    using prometheus::MetricFamily;
    using prometheus::MetricType;

    [[nodiscard]] std::vector<ClientMetric::Label>
    baseLabels(const storeSnapshot &s) {
        std::vector<ClientMetric::Label> labels;
        labels.push_back({"hostname", s.hostname});
        labels.push_back({"agent_name", s.agentName});
        // pid guarantees per-process series uniqueness even if agent names are
        // not unique across processes; avoids duplicate-series scrape errors. Not
        // named "instance" (a reserved Prometheus target label).
        labels.push_back({"pid", std::to_string(s.pid)});
        labels.push_back({"agent_instance", std::to_string(s.instance)});
        if (!s.localRank.empty()) {
            labels.push_back({"local_rank", s.localRank});
        }
        return labels;
    }

    [[nodiscard]] ClientMetric
    counterMetric(std::vector<ClientMetric::Label> labels, uint64_t value) {
        ClientMetric m;
        m.label = std::move(labels);
        m.counter.value = static_cast<double>(value);
        return m;
    }

    [[nodiscard]] ClientMetric
    gaugeMetric(std::vector<ClientMetric::Label> labels, uint64_t value) {
        ClientMetric m;
        m.label = std::move(labels);
        m.gauge.value = static_cast<double>(value);
        return m;
    }

    // Turns the store's per-bucket counts into the cumulative form Prometheus
    // expects, ending with the explicit +Inf bucket that prometheus-cpp's own
    // Histogram::Collect() emits.
    [[nodiscard]] ClientMetric
    histogramMetric(std::vector<ClientMetric::Label> labels,
                    const storeSnapshot &s,
                    std::size_t slot) {
        ClientMetric m;
        m.label = std::move(labels);
        m.histogram.bucket.reserve(s.bucketCount + 1);
        uint64_t cumulative = 0;
        for (std::size_t i = 0; i <= s.bucketCount; ++i) {
            cumulative += s.histBuckets[slot][i];
            ClientMetric::Bucket bucket;
            bucket.cumulative_count = cumulative;
            bucket.upper_bound =
                (i == s.bucketCount) ? std::numeric_limits<double>::infinity() : s.bucketBounds[i];
            m.histogram.bucket.push_back(bucket);
        }
        m.histogram.sample_count = cumulative;
        m.histogram.sample_sum = static_cast<double>(s.histSums[slot]);
        return m;
    }

    [[nodiscard]] bool
    nameMatchesStore(const std::string &name) {
        return name.size() > MP_STORE_FILE_PREFIX.size() + MP_STORE_FILE_SUFFIX.size() &&
            name.starts_with(MP_STORE_FILE_PREFIX) && name.ends_with(MP_STORE_FILE_SUFFIX);
    }

} // namespace

bool
isSnapshotLive(const storeSnapshot &snap, std::chrono::nanoseconds ttl, bool writer_alive) {
    if (writer_alive) {
        return true;
    }
    const uint64_t now = monotonicNs();
    const auto ttl_ns = static_cast<uint64_t>(ttl.count() < 0 ? 0 : ttl.count());
    return now >= snap.lastUpdateNs && (now - snap.lastUpdateNs) <= ttl_ns;
}

std::vector<MetricFamily>
buildMetricFamilies(const std::vector<storeSnapshot> &snapshots) {
    std::vector<MetricFamily> families;
    if (snapshots.empty()) {
        return families;
    }

    for (const auto type : telemetry_metric_event_types) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(type);
        if (descriptor.counterName == nullptr) {
            continue;
        }
        MetricFamily family;
        family.name = descriptor.counterName;
        family.help = descriptor.counterHelp;
        family.type = MetricType::Counter;
        const auto slot = static_cast<std::size_t>(type);
        for (const auto &snap : snapshots) {
            family.metric.push_back(counterMetric(baseLabels(snap), snap.counters[slot]));
        }
        families.push_back(std::move(family));
    }

    for (const auto type : telemetry_metric_event_types) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(type);
        if (descriptor.gaugeName == nullptr) {
            continue;
        }
        MetricFamily family;
        family.name = descriptor.gaugeName;
        family.help = descriptor.gaugeHelp;
        family.type = MetricType::Gauge;
        const auto slot = static_cast<std::size_t>(type);
        for (const auto &snap : snapshots) {
            family.metric.push_back(gaugeMetric(baseLabels(snap), snap.gauges[slot]));
        }
        families.push_back(std::move(family));
    }

    for (const auto type : telemetry_metric_event_types) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(type);
        if (descriptor.histogramName == nullptr) {
            continue;
        }
        MetricFamily family;
        family.name = descriptor.histogramName;
        family.help = descriptor.histogramHelp;
        family.type = MetricType::Histogram;
        const auto slot = static_cast<std::size_t>(type);
        for (const auto &snap : snapshots) {
            family.metric.push_back(histogramMetric(baseLabels(snap), snap, slot));
        }
        families.push_back(std::move(family));
    }

    MetricFamily errors;
    errors.name = telemetry_error_family_name;
    errors.help = telemetry_error_family_help;
    errors.type = MetricType::Counter;
    for (const auto &snap : snapshots) {
        for (const auto type : telemetry_error_event_types) {
            auto labels = baseLabels(snap);
            labels.push_back({"status", nixlEnumStrings::telemetryErrorStatusLabel(type)});
            errors.metric.push_back(
                counterMetric(std::move(labels), snap.counters[static_cast<std::size_t>(type)]));
        }
    }
    families.push_back(std::move(errors));

    return families;
}

nixlMultiprocessCollector::nixlMultiprocessCollector(std::filesystem::path dir,
                                                     std::chrono::nanoseconds stale_ttl,
                                                     bool reap_stale)
    : dir_(std::move(dir)),
      staleTtl_(stale_ttl),
      reapStale_(reap_stale) {}

std::vector<MetricFamily>
nixlMultiprocessCollector::Collect() const {
    // prometheus-cpp calls this on its HTTP handler thread; an exception escaping
    // into that handler (std::bad_alloc from the allocations below, say) would
    // fail the scrape at best and terminate the process at worst. Telemetry must
    // never take down the data plane, so degrade to an empty scrape instead.
    try {
        return buildMetricFamilies(scanLiveStores());
    }
    catch (const std::exception &e) {
        NIXL_WARN << "prometheus_mp: telemetry collection failed: " << e.what();
    }
    catch (...) {
        NIXL_WARN << "prometheus_mp: telemetry collection failed";
    }
    return {};
}

std::vector<storeSnapshot>
nixlMultiprocessCollector::scanLiveStores() const {
    std::vector<storeSnapshot> live;

    std::error_code ec;
    std::filesystem::directory_iterator it(dir_, ec);
    if (ec) {
        NIXL_DEBUG << "prometheus_mp: cannot scan telemetry dir '" << dir_.string()
                   << "': " << ec.message();
        return {};
    }

    // Iterate with the non-throwing increment: peer writers and this collector's
    // own reaping mutate the directory concurrently, and the range-for's
    // operator++ would throw on a mid-iteration filesystem error.
    for (const std::filesystem::directory_iterator end; it != end; it.increment(ec)) {
        if (ec) {
            NIXL_DEBUG << "prometheus_mp: telemetry dir iteration stopped early: " << ec.message();
            break;
        }
        const auto &entry = *it;
        if (!entry.is_regular_file(ec) || !nameMatchesStore(entry.path().filename().string())) {
            continue;
        }
        const bool writer_alive = storeWriterAlive(entry.path());
        auto [snap, content_invalid] = readStoreSnapshot(entry.path());
        if (!snap) {
            // Reap only genuinely bad content (bad/zero magic, wrong schema,
            // truncated) that nobody holds. A transient open/mmap failure leaves
            // contentInvalid false, so a healthy peer we simply failed to read is
            // never unlinked.
            if (reapStale_ && content_invalid && !writer_alive) {
                std::error_code rm_ec;
                std::filesystem::remove(entry.path(), rm_ec);
            }
            continue;
        }
        if (isSnapshotLive(*snap, staleTtl_, writer_alive)) {
            live.push_back(std::move(*snap));
        } else if (reapStale_) {
            std::error_code rm_ec;
            std::filesystem::remove(entry.path(), rm_ec);
        }
    }

    return live;
}

} // namespace nixl::telemetry::mp
