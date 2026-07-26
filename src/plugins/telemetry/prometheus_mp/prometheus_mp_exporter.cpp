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
#include "prometheus_mp_exporter.h"

#include "common/configuration.h"
#include "common/hostname.h"
#include "common/nixl_log.h"
#include "histogram_buckets.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

using nixl::telemetry::mp::makeStoreFileName;
using nixl::telemetry::mp::MP_DEFAULT_STALE_TTL;
using nixl::telemetry::mp::storeWriter;
using nixl::telemetry::mp::nixlMultiprocessCollector;
using nixl::telemetry::mp::ownerElection;
using nixl::telemetry::mp::readProcessStartTime;

constexpr uint16_t defaultPort = 9090;
constexpr char defaultRankEnvName[] = "LOCAL_RANK";

constexpr char prometheusPortVar[] = "NIXL_TELEMETRY_PROMETHEUS_PORT";
constexpr char prometheusLocalVar[] = "NIXL_TELEMETRY_PROMETHEUS_LOCAL";
constexpr char multiprocDirVar[] = "NIXL_TELEMETRY_MULTIPROC_DIR";
constexpr char rankEnvVar[] = "NIXL_TELEMETRY_RANK_ENV";
constexpr char staleTtlVar[] = "NIXL_TELEMETRY_MP_STALE_TTL";

const std::string localAddress = "127.0.0.1";
const std::string publicAddress = "0.0.0.0";

// How often a writer re-runs the election to notice that the owner died. The
// election is one non-blocking flock, but it sits on the export path, so it is
// throttled; the endpoint is down for at most this plus the scrape interval.
// Writers start at 0, so the first exported event re-checks immediately -- an
// owner that died between this process's election and its first event is caught
// at once.
constexpr uint64_t electionRetryIntervalNs = 200000000ULL;

// Once a re-election has won and still failed to bind, the port is held from
// outside the run rather than by a rank of it, which is a condition that lasts.
// Retrying an Exposer five times a second against it is the one costly case, so
// back off; a transient conflict then clears in seconds instead of milliseconds.
constexpr uint64_t electionBackoffNs = 5000000000ULL;

// civetweb reports a failed port bind with this exact text (as used by the
// single-process prometheus exporter). Only this case is treated as a benign
// bind collision; any other Exposer failure is a genuine error.
constexpr char bindFailureMarker[] = "Failed to setup server ports";

// Resolves the optional local_rank label value: NIXL_TELEMETRY_RANK_ENV names
// which env var holds the rank (default LOCAL_RANK); the value of that env var is
// the rank. Empty when the named env var is unset -- rank is a best-effort label
// only. This is the local/per-GPU (TP) rank, distinct from Dynamo's dp_rank.
[[nodiscard]] std::string
resolveLocalRank() {
    const std::string rank_source =
        nixl::config::getValueDefaulted<std::string>(rankEnvVar, defaultRankEnvName);
    if (rank_source.empty()) {
        return {};
    }
    return nixl::config::getValueOptional<std::string>(rank_source).value_or(std::string());
}

[[nodiscard]] std::chrono::nanoseconds
resolveStaleTtl() {
    const uint64_t configured = nixl::config::getValueDefaulted<uint64_t>(
        staleTtlVar, static_cast<uint64_t>(MP_DEFAULT_STALE_TTL.count()));
    constexpr uint64_t max_seconds =
        static_cast<uint64_t>(std::chrono::nanoseconds::max().count()) / 1000000000ULL;
    const auto seconds = std::chrono::seconds(std::min(configured, max_seconds));
    return std::chrono::duration_cast<std::chrono::nanoseconds>(seconds);
}

[[nodiscard]] std::filesystem::path
resolveMultiprocDir() {
    const auto dir = nixl::config::getValueOptional<std::string>(multiprocDirVar);
    if (!dir || dir->empty()) {
        throw std::runtime_error(
            "prometheus_mp exporter requires NIXL_TELEMETRY_MULTIPROC_DIR to be set");
    }
    std::filesystem::path path(*dir);
    std::error_code ec;
    const bool created = std::filesystem::create_directories(path, ec);
    if (ec) {
        throw std::runtime_error("prometheus_mp exporter: cannot create telemetry dir '" +
                                 path.string() + "': " + ec.message());
    }
    if (created) {
        // The umask default (typically 0755) would leave the store files readable
        // by every user on the host; the O_NOFOLLOW/0600/uid checks defend the
        // files, but only 0700 keeps a co-tenant out of the directory itself.
        std::filesystem::permissions(
            path, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace, ec);
        if (ec) {
            NIXL_WARN << "prometheus_mp: cannot restrict telemetry dir '" << path.string()
                      << "' to 0700: " << ec.message();
        }
    }
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0 && (st.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
        NIXL_WARN << "prometheus_mp: telemetry dir '" << path.string()
                  << "' is writable by group or other; another user can plant store and lock "
                  << "files there. Use a private directory owned by this user (mode 0700)";
    }
    return path;
}

// Per-process instance counter so multiple agents in one process get distinct
// store files.
std::atomic<uint64_t> s_instanceSeq{0};

} // namespace

nixlTelemetryPrometheusMpExporter::nixlTelemetryPrometheusMpExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params),
      dir_(resolveMultiprocDir()),
      staleTtl_(resolveStaleTtl()),
      retryIntervalNs_(electionRetryIntervalNs) {
    const int64_t pid = static_cast<int64_t>(::getpid());
    const uint64_t start_time = readProcessStartTime(pid);
    const uint64_t instance = s_instanceSeq.fetch_add(1, std::memory_order_relaxed);
    const std::filesystem::path store_path = dir_ / makeStoreFileName(pid, start_time, instance);

    store_ = std::make_unique<storeWriter>(store_path,
                                           init_params.agentName,
                                           nixl::getHostname().value_or("unknown"),
                                           resolveLocalRank(),
                                           instance,
                                           nixl::telemetry::resolveHistogramBucketsUs());

    const bool local = nixl::config::getValueDefaulted(prometheusLocalVar, false);
    const uint16_t port = nixl::config::getValueDefaulted(prometheusPortVar, defaultPort);
    bindAddress_ = (local ? localAddress : publicAddress) + ":" + std::to_string(port);

    election_ = ownerElection(dir_);
    if (election_.won() && startServing()) {
        NIXL_INFO << "prometheus_mp exporter (owner) serving " << bindAddress_
                  << ", aggregating telemetry dir " << dir_.string();
        return;
    }

    if (election_.won()) {
        // Elected, so no sibling can be serving: the port belongs to something
        // outside this run and nothing will aggregate this directory. The
        // election was conceded in startServing(), so a process starting once the
        // port frees -- a conflict as short as a previous run still shutting down
        // -- can take the endpoint over.
        NIXL_WARN << "prometheus_mp: elected to serve telemetry dir " << dir_.string() << " but "
                  << bindAddress_ << " is held by a process outside this run (a foreign service, "
                  << "or a rank pointed at a different " << multiprocDirVar
                  << "); nothing aggregates this directory";
    } else {
        const std::string owner_endpoint = election_.publishedEndpoint();
        if (!owner_endpoint.empty() && owner_endpoint != bindAddress_) {
            NIXL_WARN << "prometheus_mp: this rank asks for " << bindAddress_
                      << " but telemetry dir " << dir_.string() << " is already served on "
                      << owner_endpoint << "; ranks disagree on " << prometheusPortVar << '/'
                      << prometheusLocalVar << ", and only " << owner_endpoint << " is scrapeable";
        }
        election_.release();
    }
    NIXL_INFO << "prometheus_mp exporter (writer): endpoint " << bindAddress_
              << " owned by another process; agent '" << init_params.agentName << "' writing to "
              << store_path.string();
}

nixlTelemetryPrometheusMpExporter::~nixlTelemetryPrometheusMpExporter() = default;

bool
nixlTelemetryPrometheusMpExporter::startServing() {
    try {
        auto exposer = std::make_shared<prometheus::Exposer>(bindAddress_);
        auto collector = std::make_shared<nixlMultiprocessCollector>(dir_, staleTtl_);
        exposer->RegisterCollectable(collector);
        collector_ = std::move(collector);
        exposer_ = std::move(exposer);
        election_.publishEndpoint(bindAddress_);
        return true;
    }
    catch (const std::exception &e) {
        if (std::string(e.what()).find(bindFailureMarker) == std::string::npos) {
            throw;
        }
        // Hold the lock and the port stays unreachable for the whole run, since
        // no other rank can then be elected to try it.
        election_.release();
        return false;
    }
}

void
nixlTelemetryPrometheusMpExporter::retryElection(uint64_t now_ns) noexcept {
    if (now_ns - lastElectionNs_ < retryIntervalNs_) {
        return;
    }
    lastElectionNs_ = now_ns;
    try {
        election_ = ownerElection(dir_, false);
        if (!election_.won()) {
            election_.release();
            return;
        }
        if (!startServing()) {
            election_.release();
            retryIntervalNs_ = electionBackoffNs;
            return;
        }
        NIXL_INFO << "prometheus_mp exporter: no process owns telemetry dir " << dir_.string()
                  << " any more; this one now serves " << bindAddress_;
    }
    catch (const std::exception &e) {
        // Telemetry must not break the export path it is called from.
        NIXL_DEBUG << "prometheus_mp: cannot take over " << bindAddress_ << ": " << e.what();
    }
}

nixl_status_t
nixlTelemetryPrometheusMpExporter::exportEvent(const nixlTelemetryEvent &event) {
    const auto type = event.eventType_;
    const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(type);
    const bool is_error = nixlEnumStrings::telemetryErrorStatusLabel(type) != nullptr;

    if (descriptor.counterName != nullptr || is_error) {
        store_->addCounter(type, event.value_);
    }
    if (descriptor.gaugeName != nullptr) {
        store_->setGauge(type, event.value_);
    }
    if (descriptor.histogramName != nullptr) {
        store_->observeHistogram(type, event.value_);
    }
    // Once per event, not once per slot updated: a duration event touches three
    // slots, and the clock read costs several times the atomics it would follow.
    const uint64_t now = store_->refreshHeartbeat();
    if (!exposer_) {
        retryElection(now);
    }
    return NIXL_SUCCESS;
}
