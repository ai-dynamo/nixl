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
#ifndef NIXL_SRC_API_CPP_TELEMETRY_TELEMETRY_EXPORTER_H
#define NIXL_SRC_API_CPP_TELEMETRY_TELEMETRY_EXPORTER_H

#include "nixl_types.h"
#include "telemetry_event.h"

#include <stdexcept>
#include <string>
#include <utility>

inline constexpr char telemetryExporterVar[] = "NIXL_TELEMETRY_EXPORTER";

/**
 * @brief Thrown by a telemetry exporter when its scrape endpoint could not bind
 *        its port -- typically a benign multi-process collision. Callers may
 *        catch it to continue without a telemetry sink instead of treating the
 *        failure as fatal.
 */
class nixlTelemetryBindFailed : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * @struct nixlTelemetryExporterInitParams
 * @brief Initialization parameters for telemetry exporters
 */
struct nixlTelemetryExporterInitParams {
    std::string agentName;
    size_t maxEventsBuffered;
};

/**
 * @class nixlTelemetryExporter
 * @brief Abstract base class for telemetry exporters
 *
 * This class defines the interface that all telemetry exporters must implement.
 * It provides the core functionality for reading telemetry events and exporting
 * them to various destinations.
 */
class nixlTelemetryExporter {
public:
    explicit nixlTelemetryExporter(const nixlTelemetryExporterInitParams &init_params) noexcept
        : maxEventsBuffered_(init_params.maxEventsBuffered) {};
    nixlTelemetryExporter(nixlTelemetryExporter &&) = delete;
    nixlTelemetryExporter(const nixlTelemetryExporter &) = delete;

    void
    operator=(nixlTelemetryExporter &&) = delete;
    void
    operator=(const nixlTelemetryExporter &) = delete;

    virtual ~nixlTelemetryExporter() = default;

    [[nodiscard]] size_t
    getMaxEventsBuffered() const noexcept {
        return maxEventsBuffered_;
    }

    virtual nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) = 0;

    /**
     * @brief Runs @p body with a batch open around it.
     *
     * Exporters may use the batch boundary to share per-batch work across the
     * exportEvent() calls made inside @p body (e.g. a single timestamp). The
     * batch begins before @p body runs and ends when it returns (including on
     * exception). Scopes may nest; the lifecycle hooks fire only for the
     * outermost scope, so a nested call reuses the surrounding batch.
     *
     * @tparam Body Callable invocable with no arguments.
     * @param body Callable run once while the batch is open.
     */
    template<typename Body>
    void
    withBatch(Body &&body) {
        const batchScope scope{*this};
        std::forward<Body>(body)();
    }

private:
    class batchScope {
    public:
        explicit batchScope(nixlTelemetryExporter &exporter) noexcept : exporter_(exporter) {
            if (exporter_.batchDepth_++ == 0) {
                exporter_.onBatchBegin();
            }
        }

        ~batchScope() {
            if (--exporter_.batchDepth_ == 0) {
                exporter_.onBatchEnd();
            }
        }

        batchScope(const batchScope &) = delete;
        batchScope(batchScope &&) = delete;
        batchScope &
        operator=(const batchScope &) = delete;
        batchScope &
        operator=(batchScope &&) = delete;

    private:
        nixlTelemetryExporter &exporter_;
    };

    virtual void
    onBatchBegin() noexcept {}

    virtual void
    onBatchEnd() noexcept {}

    unsigned batchDepth_ = 0;
    const size_t maxEventsBuffered_;
};

#endif // NIXL_SRC_API_CPP_TELEMETRY_TELEMETRY_EXPORTER_H
