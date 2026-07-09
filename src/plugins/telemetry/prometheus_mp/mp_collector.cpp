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
#include "mp_collector.h"

#include "common/nixl_log.h"
#include "telemetry_event.h"

#include <prometheus/client_metric.h>
#include <prometheus/metric_type.h>

#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <ctime>
#include <system_error>

namespace nixl::telemetry::mp {

namespace {

    using prometheus::ClientMetric;
    using prometheus::MetricFamily;
    using prometheus::MetricType;

    [[nodiscard]] uint64_t
    nowNs() noexcept {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count());
    }

    [[nodiscard]] std::vector<ClientMetric::Label>
    baseLabels(const mpStoreSnapshot &s) {
        std::vector<ClientMetric::Label> labels;
        labels.push_back({"hostname", s.hostname});
        labels.push_back({"agent_name", s.agentName});
        // pid guarantees per-process series uniqueness even if agent names are
        // not unique across processes; avoids duplicate-series scrape errors. Not
        // named "instance" (a reserved Prometheus target label).
        labels.push_back({"pid", std::to_string(s.pid)});
        if (!s.dpRank.empty()) {
            labels.push_back({"dp_rank", s.dpRank});
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

    // Minimum age before an unparseable file is reaped, even when the TTL is 0.
    // Protects a store a live process is actively creating (a sub-second window)
    // from being deleted out from under it.
    constexpr long kInvalidFileFloorSeconds = 2;

    // Whether an unparseable store file (bad/zero magic, wrong schema, truncated)
    // is old enough to be an orphan worth removing, rather than a live process's
    // store mid-creation.
    [[nodiscard]] bool
    invalidFileReapable(const std::filesystem::path &path, std::chrono::nanoseconds ttl) {
        struct stat st{};
        if (::stat(path.c_str(), &st) != 0) {
            return false;
        }
        const long ttl_seconds =
            static_cast<long>(std::chrono::duration_cast<std::chrono::seconds>(ttl).count());
        const long grace = std::max<long>(ttl_seconds, kInvalidFileFloorSeconds);
        const long age = static_cast<long>(::time(nullptr) - st.st_mtime);
        return age > grace;
    }

    [[nodiscard]] bool
    nameMatchesStore(const std::string &name) {
        return name.size() > MP_STORE_FILE_PREFIX.size() + MP_STORE_FILE_SUFFIX.size() &&
            name.compare(0, MP_STORE_FILE_PREFIX.size(), MP_STORE_FILE_PREFIX) == 0 &&
            name.compare(name.size() - MP_STORE_FILE_SUFFIX.size(),
                         MP_STORE_FILE_SUFFIX.size(),
                         MP_STORE_FILE_SUFFIX) == 0;
    }

} // namespace

bool
isProcessAlive(int64_t pid, uint64_t start_time) {
    if (pid <= 0) {
        return false;
    }
    if (::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH) {
        return false;
    }
    // Process exists (kill succeeded, or failed with EPERM). Guard against PID
    // reuse: if we recorded a start time and can read the current one, they must
    // match.
    if (start_time != 0) {
        const uint64_t current = readProcessStartTime(pid);
        if (current != 0 && current != start_time) {
            return false;
        }
    }
    return true;
}

bool
isSnapshotLive(const mpStoreSnapshot &snap, std::chrono::nanoseconds ttl) {
    if (isProcessAlive(snap.pid, snap.startTime)) {
        return true;
    }
    const uint64_t now = nowNs();
    const auto ttl_ns = static_cast<uint64_t>(ttl.count() < 0 ? 0 : ttl.count());
    return now >= snap.lastUpdateNs && (now - snap.lastUpdateNs) <= ttl_ns;
}

std::vector<MetricFamily>
buildMetricFamilies(const std::vector<mpStoreSnapshot> &snapshots) {
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

    MetricFamily errors;
    errors.name = "agent_errors_total";
    errors.help = "Cumulative error count by status";
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
    std::vector<mpStoreSnapshot> live;

    std::error_code ec;
    std::filesystem::directory_iterator it(dir_, ec);
    if (ec) {
        NIXL_DEBUG << "prometheus_mp: cannot scan telemetry dir '" << dir_.string()
                   << "': " << ec.message();
        return {};
    }

    for (const auto &entry : it) {
        if (!entry.is_regular_file(ec) || !nameMatchesStore(entry.path().filename().string())) {
            continue;
        }
        auto snap = readStoreSnapshot(entry.path());
        if (!snap) {
            // Unparseable (mid-init, orphaned, or incompatible): reap only if it is
            // old enough to not be a store a live process is currently creating.
            if (reapStale_ && invalidFileReapable(entry.path(), staleTtl_)) {
                std::error_code rm_ec;
                std::filesystem::remove(entry.path(), rm_ec);
            }
            continue;
        }
        if (isSnapshotLive(*snap, staleTtl_)) {
            live.push_back(std::move(*snap));
        } else if (reapStale_) {
            std::error_code rm_ec;
            std::filesystem::remove(entry.path(), rm_ec);
        }
    }

    return buildMetricFamilies(live);
}

} // namespace nixl::telemetry::mp
