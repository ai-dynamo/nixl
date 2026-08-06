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

#include <absl/strings/str_join.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nixl::telemetry::mp::heldAddressesExcept;
using nixl::telemetry::mp::makeStoreFileName;
using nixl::telemetry::mp::MP_DEFAULT_STALE_TTL;
using nixl::telemetry::mp::processRunMarker;
using nixl::telemetry::mp::scrapeEndpoint;
using nixl::telemetry::mp::storeWriter;

constexpr uint16_t defaultPort = 9090;
constexpr char defaultRankEnvName[] = "LOCAL_RANK";

constexpr char prometheusPortVar[] = "NIXL_TELEMETRY_PROMETHEUS_PORT";
constexpr char prometheusLocalVar[] = "NIXL_TELEMETRY_PROMETHEUS_LOCAL";
constexpr char multiprocDirVar[] = "NIXL_TELEMETRY_MULTIPROC_DIR";
constexpr char rankEnvVar[] = "NIXL_TELEMETRY_RANK_ENV";
constexpr char staleTtlVar[] = "NIXL_TELEMETRY_MP_STALE_TTL";

const std::string localAddress = "127.0.0.1";
const std::string publicAddress = "0.0.0.0";

[[nodiscard]] std::string
resolveBindAddress() {
    const bool local = nixl::config::getValueDefaulted(prometheusLocalVar, false);
    const uint16_t port = nixl::config::getValueDefaulted(prometheusPortVar, defaultPort);
    return (local ? localAddress : publicAddress) + ":" + std::to_string(port);
}

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
    if (::stat(path.c_str(), &st) != 0) {
        return path;
    }
    // Follows symlinks, deliberately: a configured path that is a symlink into
    // another user's directory is exactly the case worth reporting, and rejecting
    // symlinks outright would break the legitimate ones.
    if (st.st_uid != ::geteuid()) {
        NIXL_WARN << "prometheus_mp: telemetry dir '" << path.string() << "' is owned by uid "
                  << st.st_uid << "; this process writes its stores into a directory another "
                  << "user controls. Use a private directory owned by this user (mode 0700)";
    }
    if ((st.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
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
      endpoint_(dir_, resolveBindAddress(), resolveStaleTtl()) {
    const int64_t pid = static_cast<int64_t>(::getpid());
    const uint64_t instance = s_instanceSeq.fetch_add(1, std::memory_order_relaxed);
    const std::filesystem::path store_path =
        dir_ / makeStoreFileName(pid, processRunMarker(), instance);

    store_ = std::make_unique<storeWriter>(store_path,
                                           init_params.agentName,
                                           nixl::getHostname().value_or("unknown"),
                                           resolveLocalRank(),
                                           instance,
                                           nixl::telemetry::resolveHistogramBucketsUs());

    const std::string &bind_address = endpoint_.bindAddress();
    const auto claimed = endpoint_.claim();
    // Elections are per address, so ranks that disagree about the port each own
    // one instead of contending. Reported from whichever side notices, since the
    // ranks start in no particular order.
    const std::vector<std::string> others = claimed == scrapeEndpoint::status::SIBLING_OWNS ?
        std::vector<std::string>{} :
        heldAddressesExcept(dir_, bind_address);
    if (!others.empty()) {
        NIXL_WARN << "prometheus_mp: ranks of telemetry dir " << dir_.string() << " disagree on "
                  << prometheusPortVar << '/' << prometheusLocalVar << ": " << bind_address
                  << " is not the only address serving it (" << absl::StrJoin(others, ", ")
                  << "). Each of them exports every rank, so Prometheus scraping more than one "
                  << "sees the same series twice";
    }

    switch (claimed) {
    case scrapeEndpoint::status::SERVING:
        NIXL_INFO << "prometheus_mp exporter (owner) serving " << bind_address
                  << ", aggregating telemetry dir " << dir_.string();
        return;

    case scrapeEndpoint::status::PORT_TAKEN:
        // Elected for this address, so no sibling asking for the same one can be
        // serving: the port belongs to something outside the run, or to a rank
        // that asked for the same port on a different address. The election is
        // conceded rather than held, so the next rank to win takes the address
        // over once the port frees.
        NIXL_WARN << "prometheus_mp: elected to serve telemetry dir " << dir_.string() << " but "
                  << bind_address << " is held by a process outside this run (a foreign service, "
                  << "or a rank pointed at a different " << multiprocDirVar
                  << "); nothing aggregates this directory on " << bind_address;
        break;

    case scrapeEndpoint::status::SIBLING_OWNS:
        break;
    }
    NIXL_INFO << "prometheus_mp exporter (writer): address " << bind_address
              << " owned by another process; agent '" << init_params.agentName << "' writing to "
              << store_path.string();
}

nixlTelemetryPrometheusMpExporter::~nixlTelemetryPrometheusMpExporter() = default;

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
    if (!endpoint_.serving()) {
        endpoint_.reclaim(std::chrono::nanoseconds(now));
    }
    return NIXL_SUCCESS;
}
