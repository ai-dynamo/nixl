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
#include "prometheus_mp_exporter.h"

#include "common/configuration.h"
#include "common/nixl_log.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

using nixl::telemetry::mp::makeStoreFileName;
using nixl::telemetry::mp::mpStoreWriter;
using nixl::telemetry::mp::nixlMultiprocessCollector;
using nixl::telemetry::mp::readProcessStartTime;

constexpr uint16_t defaultPort = 9090;
constexpr uint64_t defaultStaleTtlSeconds = 30;
constexpr char defaultRankEnvName[] = "LOCAL_RANK";

constexpr char prometheusPortVar[] = "NIXL_TELEMETRY_PROMETHEUS_PORT";
constexpr char prometheusLocalVar[] = "NIXL_TELEMETRY_PROMETHEUS_LOCAL";
constexpr char multiprocDirVar[] = "NIXL_TELEMETRY_MULTIPROC_DIR";
constexpr char rankEnvVar[] = "NIXL_TELEMETRY_RANK_ENV";
constexpr char staleTtlVar[] = "NIXL_TELEMETRY_MP_STALE_TTL";

const std::string localAddress = "127.0.0.1";
const std::string publicAddress = "0.0.0.0";

// civetweb reports a failed port bind with this exact text (as used by the
// single-process prometheus exporter). Only this case is treated as a benign
// bind collision; any other Exposer failure is a genuine error.
constexpr char bindFailureMarker[] = "Failed to setup server ports";

[[nodiscard]] std::string
getHostname() {
    char hostname[HOST_NAME_MAX + 1];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[HOST_NAME_MAX] = '\0';
        return std::string(hostname);
    }
    return "unknown";
}

// Resolves the optional dp_rank label value: NIXL_TELEMETRY_RANK_ENV names which
// env var holds the rank (default LOCAL_RANK); the value of that env var is the
// rank. Empty when the named env var is unset -- rank is a best-effort label only.
[[nodiscard]] std::string
resolveDpRank() {
    const std::string rank_source =
        nixl::config::getValueDefaulted<std::string>(rankEnvVar, defaultRankEnvName);
    if (rank_source.empty()) {
        return {};
    }
    const char *value = std::getenv(rank_source.c_str());
    return value != nullptr ? std::string(value) : std::string();
}

[[nodiscard]] std::chrono::nanoseconds
resolveStaleTtl() {
    const uint64_t seconds =
        nixl::config::getValueDefaulted<uint64_t>(staleTtlVar, defaultStaleTtlSeconds);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(seconds));
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
    std::filesystem::create_directories(path, ec);
    if (ec) {
        throw std::runtime_error("prometheus_mp exporter: cannot create telemetry dir '" +
                                 path.string() + "': " + ec.message());
    }
    return path;
}

// Per-process instance counter so multiple agents in one process get distinct
// store files.
std::atomic<uint64_t> s_instanceSeq{0};

} // namespace

nixlTelemetryPrometheusMpExporter::nixlTelemetryPrometheusMpExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params) {
    const std::filesystem::path dir = resolveMultiprocDir();

    const int64_t pid = static_cast<int64_t>(::getpid());
    const uint64_t start_time = readProcessStartTime(pid);
    const uint64_t instance = s_instanceSeq.fetch_add(1, std::memory_order_relaxed);
    const std::filesystem::path store_path = dir / makeStoreFileName(pid, start_time, instance);

    store_ = std::make_unique<mpStoreWriter>(
        store_path, init_params.agentName, getHostname(), resolveDpRank());

    const bool local = nixl::config::getValueDefaulted(prometheusLocalVar, false);
    const uint16_t port = nixl::config::getValueDefaulted(prometheusPortVar, defaultPort);
    const std::string bind_address =
        (local ? localAddress : publicAddress) + ":" + std::to_string(port);

    try {
        auto exposer = std::make_shared<prometheus::Exposer>(bind_address);
        collector_ = std::make_shared<nixlMultiprocessCollector>(dir, resolveStaleTtl());
        exposer->RegisterCollectable(collector_);
        exposer_ = std::move(exposer);
        owner_ = true;
        NIXL_INFO << "prometheus_mp exporter (owner) serving " << bind_address
                  << ", aggregating telemetry dir " << dir.string();
    }
    catch (const std::exception &e) {
        if (std::string(e.what()).find(bindFailureMarker) == std::string::npos) {
            throw;
        }
        owner_ = false;
        NIXL_INFO << "prometheus_mp exporter (writer): endpoint " << bind_address
                  << " owned by another process; agent '" << init_params.agentName
                  << "' writing to " << store_path.string();
    }
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
    return NIXL_SUCCESS;
}
