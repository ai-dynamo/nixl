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

#include <iostream>
#include <signal.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>
#include <filesystem>
#include <string>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <errno.h>


namespace fs = std::filesystem;

#include "common/cyclic_buffer.h"
#include "telemetry_event.h"

volatile sig_atomic_t g_running = true;

// Signal handler for Ctrl+C
void
signal_handler(int signal) {
    if (signal == SIGINT) {
        g_running = false;
    }
}

std::string
format_timestamp(uint64_t timestamp_us) {
    auto time_point =
        std::chrono::system_clock::time_point(std::chrono::microseconds(timestamp_us));
    auto time_t = std::chrono::system_clock::to_time_t(time_point);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");

    auto microseconds = timestamp_us % 1000000;
    ss << "." << std::setfill('0') << std::setw(6) << microseconds;

    return ss.str();
}

std::string
format_bytes(uint64_t bytes) {
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double value = static_cast<double>(bytes);

    while (value >= 1024.0 && unit_index < 4) {
        value /= 1024.0;
        unit_index++;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value << " " << units[unit_index];
    return ss.str();
}

void
print_telemetry_event(const nixlTelemetryEvent &event) {
    // Can be extended to more general ostream if needed
    // friend std::ostream &operator<<(std::ostream &os, const nixlTelemetryEvent &event)
    std::cout << "\n=== NIXL Telemetry Event ===" << std::endl;
    std::cout << "Timestamp: " << format_timestamp(event.timestampUs_) << std::endl;
    std::cout << "Category: " << nixlEnumStrings::telemetryCategoryStr(event.category_)
              << std::endl;
    std::cout << "Event name: " << event.eventName_ << std::endl;
    std::cout << "Value: " << event.value_ << std::endl;

    std::cout << "===========================" << std::endl;
}

void
usage() {
    std::cout << "Usage: telemetry_reader <telemetry_file_path>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  <telemetry_file_path>    Path to the telemetry file" << std::endl;
    exit(0);
}

int
main(int argc, char *argv[]) {
    if (argc < 2 || argv[1] == std::string("-h") || argv[1] == std::string("--help")) {
        usage();
    }

    std::cout << "Telemetry path: " << argv[1] << std::endl;
    auto telemetry_path = argv[1];

    if (!fs::exists(telemetry_path)) {
        std::cerr << "Telemetry file " << telemetry_path << " does not exist" << std::endl;
        return 1;
    }

    signal(SIGINT, signal_handler);

    try {
        std::cout << "Opening telemetry buffer: " << telemetry_path << std::endl;
        std::cout << "Press Ctrl+C to stop reading telemetry..." << std::endl;

        sharedRingBuffer<nixlTelemetryEvent> buffer(telemetry_path, false, TELEMETRY_VERSION);

        std::cout << "Successfully opened telemetry buffer (version: " << buffer.version() << ")"
                  << std::endl;
        std::cout << "Buffer capacity: " << buffer.capacity() << " events" << std::endl;

        nixlTelemetryEvent event;
        uint64_t event_count = 0;

        while (g_running) {
            if (buffer.pop(event)) {
                event_count++;
                print_telemetry_event(event);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }

        std::cout << "\nTotal events read: " << event_count << std::endl;
        std::cout << "Final buffer size: " << buffer.size() << " events" << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
