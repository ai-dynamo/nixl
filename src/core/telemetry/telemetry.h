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
#include "mem_section.h"
#include "nixl_types.h"

#include <string>
#include <vector>
#include <mutex>
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
     * or the agent's captureTelemetry config), so it always returns a non-null
     * instance. With an output sink configured it exports events there; without
     * one it falls back to the collect-only NOP sink, which still records events
     * in process (so getXferTelemetry() returns data) but writes nothing.
     * @param agent_name Non-empty agent name.
     * @return A non-null telemetry instance.
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
     * Appends the four per-transfer events (transfer time, bytes, request count,
     * post time) under one lock. The batch is all-or-none: if the buffer cannot
     * hold all four, none are recorded.
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
    bool
    flushPendingEvents();

    // Declared in initialization order: agentName_ and maxBufferedEvents_ are
    // consumed by makeExporter() when constructing exporter_.
    const std::string agentName_;
    const size_t maxBufferedEvents_;
    const std::unique_ptr<nixlTelemetryExporter> exporter_;
    std::vector<nixlTelemetryEvent> events_;
    std::mutex mutex_;
    asio::thread_pool pool_;
    periodicTask writeTask_;
};

#endif
