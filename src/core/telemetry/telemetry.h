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
#ifndef NIXL_SRC_CORE_TELEMETRY_TELEMETRY_H
#define NIXL_SRC_CORE_TELEMETRY_TELEMETRY_H

#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "telemetry_staging_queue.h"
#include "mem_section.h"
#include "nixl_types.h"

#include <string>
#include <memory>
#include <chrono>
#include <functional>
#include <atomic>

#include <asio.hpp>

struct periodicTask {
    asio::steady_timer timer_;
    std::function<bool()> callback_;
    std::chrono::milliseconds interval_;
    std::atomic<bool> enabled_;

    periodicTask(const asio::any_io_executor &executor,
                 std::chrono::milliseconds interval,
                 bool enabled = false)
        : timer_(executor),
          callback_(nullptr),
          interval_(interval),
          enabled_(enabled) {}
};

class nixlTelemetry {
public:
    /**
     * @brief Creates a telemetry instance that records transfer events.
     *
     * Only called when telemetry is explicitly requested (NIXL_TELEMETRY_ENABLE
     * or the agent's captureTelemetry config). With an output sink configured it
     * exports events there; without one it falls back to the collect-only NOP
     * sink, which still records events in process (so getXferTelemetry() returns
     * data) but writes nothing.
     * @param agent_name Non-empty agent name.
     * @return A telemetry instance, or nullptr if the exporter's scrape endpoint
     *         could not bind its port (a benign multi-process collision): that
     *         rank then runs without a telemetry sink instead of failing.
     * @throws std::invalid_argument / std::runtime_error on genuine
     *         configuration or plugin-load errors.
     */
    [[nodiscard]] static std::unique_ptr<nixlTelemetry>
    create(const std::string &agent_name);

    nixlTelemetry(const std::string &agent_name, const std::string &exporter_name);

    ~nixlTelemetry();

    void
    updateTxBytes(uint64_t tx_bytes);
    void
    updateRxBytes(uint64_t rx_bytes);
    void
    updateTxRequestsNum(uint32_t num);
    void
    updateRxRequestsNum(uint32_t num);
    void
    updateErrorCount(nixl_status_t error_type);
    void
    updateMemoryRegistered(uint64_t memory_registered);
    void
    updateMemoryDeregistered(uint64_t memory_deregistered);
    /**
     * @brief Records one completed transfer's stats as a single telemetry batch.
     *
     * Appends the activated subset of the four per-transfer events (transfer
     * time, bytes, request count, post time) under one lock; deactivated metrics
     * are skipped before batching. The batch is all-or-none: if the buffer cannot
     * hold the filtered batch, none of its events are recorded.
     * @param xfer_time Start-to-complete transfer duration.
     * @param is_write True for TX events (agent_tx_*), false for RX (agent_rx_*).
     * @param bytes Bytes transferred by the request.
     * @param post_time Start-to-post (backend submit) duration.
     */
    void
    addXferStats(std::chrono::microseconds xfer_time,
                 bool is_write,
                 uint64_t bytes,
                 std::chrono::microseconds post_time);

private:
    // Load the named telemetry plugin and create its exporter. Throws on a
    // genuine plugin-load / exporter-creation failure. Used to initialize the
    // const exporter_ from the member-initializer list, so a constructed
    // nixlTelemetry always has a valid exporter (create() decides up front
    // whether telemetry should exist at all).
    [[nodiscard]] std::unique_ptr<nixlTelemetryExporter>
    makeExporter(const std::string &exporter_name) const;
    void
    startExportTask();
    void
    registerPeriodicTask(periodicTask &task);
    void
    updateData(nixl_telemetry_event_type_t event_type, uint64_t value);

    // Whether the given event type is exported. Deactivated types are dropped at
    // the source (before the staging queue), so they cost no lock/append. This is
    // a lock-free read of an immutable, construction-time mask -- safe on the
    // multi-producer hot path.
    [[nodiscard]] bool
    isMetricEnabled(nixl_telemetry_event_type_t event_type) const noexcept {
        return metricEnabled_[static_cast<size_t>(event_type)];
    }
    bool
    flushPendingEvents();
    // Emits the staging-queue drops accumulated since the last flush as a
    // synthetic AGENT_TELEMETRY_EVENTS_DROPPED event.
    void
    exportDroppedEvents();

    // Declared in initialization order: agentName_ and maxBufferedEvents_ are
    // consumed by makeExporter() when constructing exporter_.
    const std::string agentName_;
    const size_t maxBufferedEvents_;
    const std::unique_ptr<nixlTelemetryExporter> exporter_;
    // Per-event-type export allowlist resolved once from NIXL_TELEMETRY_ENABLED_METRICS.
    // Indexed by nixl_telemetry_event_type_t; a deactivated event is skipped at
    // the source before it enters the staging queue. All-true when the variable
    // is unset (backward compatible).
    const nixl_telemetry_metric_mask_t metricEnabled_;
    // Bounded producer/consumer staging queue: owns event storage, the capacity
    // reserve, the producer mutex, capacity enforcement, single/batch insertion,
    // the swap-drain, and the staging-drop counter. Its drops do not track BUFFER
    // cyclic-ring loss (a separate, downstream condition); each flush takes and
    // resets the drop count and publishes it as an AGENT_TELEMETRY_EVENTS_DROPPED
    // event.
    nixlTelemetryStagingQueue stagingQueue_;
    asio::thread_pool pool_;
    periodicTask writeTask_;
};

#endif
