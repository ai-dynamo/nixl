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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_PROMETHEUS_MP_EXPORTER_H
#define NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_PROMETHEUS_MP_EXPORTER_H

#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "mp_collector.h"
#include "mp_store.h"

#include <prometheus/exposer.h>

#include <memory>

/**
 * @class nixlTelemetryPrometheusMpExporter
 * @brief Multi-process Prometheus exporter with bind-race owner election.
 *
 * Every process writes its own metric state to a per-process store in the shared
 * NIXL_TELEMETRY_MULTIPROC_DIR. On construction each process races to bind the
 * scrape port: the winner ("owner") runs a prometheus-cpp Exposer plus a
 * nixlMultiprocessCollector that aggregates all peers' stores on each scrape; the
 * losers run in writer-only mode (no HTTP server). A bind collision is therefore
 * benign -- it never throws nixlTelemetryBindFailed -- so every process gets a
 * valid telemetry sink and all ranks are exported behind the single owner port.
 */
class nixlTelemetryPrometheusMpExporter final : public nixlTelemetryExporter {
public:
    explicit nixlTelemetryPrometheusMpExporter(const nixlTelemetryExporterInitParams &init_params);
    ~nixlTelemetryPrometheusMpExporter() override;

    nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

    // True if this process won the bind race and serves the scrape endpoint.
    [[nodiscard]] bool
    isExporter() const noexcept {
        return owner_;
    }

private:
    // Declared so destruction is exposer_ -> collector_ -> store_: stop serving
    // before dropping the collector it weak-references, then unmap the store.
    std::unique_ptr<nixl::telemetry::mp::mpStoreWriter> store_;
    std::shared_ptr<nixl::telemetry::mp::nixlMultiprocessCollector> collector_;
    std::shared_ptr<prometheus::Exposer> exposer_;
    bool owner_ = false;
};

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_PROMETHEUS_MP_EXPORTER_H
