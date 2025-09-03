/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _TELEMETRY_H
#define _TELEMETRY_H

#include "common/cyclic_buffer.h"
#include "telemetry_event.h"
#include "mem_section.h"
#include "nixl_types.h"

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <chrono>
#include <functional>

#include <asio.hpp>

struct periodicTask {
    asio::steady_timer timer_;
    std::function<bool()> callback_;
    std::chrono::milliseconds interval_;

    periodicTask(const asio::any_io_executor &executor, std::chrono::milliseconds interval)
        : timer_(executor),
          interval_(interval) {}
};

class nixlTelemetry {
public:
    nixlTelemetry(const std::string &file_path, backend_map_t &backend_map);

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
    void
    addXferTime(std::chrono::microseconds transaction_time, bool is_write, uint64_t bytes);
    void
    addPostTime(std::chrono::microseconds post_time);

private:
    void
    initializeTelemetry();
    void
    registerPeriodicTask(periodicTask &task);
    void
    updateData(const std::string &event_name, nixl_telemetry_category_t category, uint64_t value);
    bool
    writeEventHelper();
    std::unique_ptr<sharedRingBuffer<nixlTelemetryEvent>> buffer_;
    std::vector<nixlTelemetryEvent> events_;
    std::mutex mutex_;
    asio::thread_pool pool_;
    periodicTask writeTask_;
    std::string file_;
    backend_map_t &backendMap_;
};

#endif // _TELEMETRY_H
