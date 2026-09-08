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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_PROMETHEUS_MP_EXPORTER_H
#define NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_PROMETHEUS_MP_EXPORTER_H

#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "mp_store.h"
#include "scrape_endpoint.h"

#include <filesystem>
#include <memory>

/**
 * @class nixlTelemetryPrometheusMpExporter
 * @brief Multi-process Prometheus exporter with locked owner election.
 *
 * Every process writes its own metric state to a per-process store in the shared
 * NIXL_TELEMETRY_MULTIPROC_DIR, and one of them additionally serves the scrape
 * endpoint that publishes all of those stores (see scrapeEndpoint for how that
 * one is picked, and re-picked when it dies). Not being the one that serves is
 * benign -- it never throws nixlTelemetryBindFailed -- so every process gets a
 * valid telemetry sink and all ranks are exported behind the single owner port.
 *
 * This class is therefore the mapping from telemetry events to store slots, plus
 * the reporting of what the election meant for this particular rank; a rank that
 * is not serving keeps trying to take the endpoint over as it exports, so a rank
 * that exports nothing stays a writer.
 */
class nixlTelemetryPrometheusMpExporter final : public nixlTelemetryExporter {
public:
    explicit nixlTelemetryPrometheusMpExporter(const nixlTelemetryExporterInitParams &init_params);
    ~nixlTelemetryPrometheusMpExporter() override;

    nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

    // True if this process won the election and serves the scrape endpoint.
    [[nodiscard]] bool
    isExporter() const noexcept {
        return endpoint_.serving();
    }

private:
    std::filesystem::path dir_;
    std::unique_ptr<nixl::telemetry::mp::storeWriter> store_;
    // Declared last so it is destroyed first: serving stops before the store it
    // publishes is unmapped.
    nixl::telemetry::mp::scrapeEndpoint endpoint_;
};

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_PROMETHEUS_MP_EXPORTER_H
