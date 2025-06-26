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
#include "nixl_types.h"

volatile bool g_running = true;

// Signal handler for Ctrl+C
void
signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nReceived Ctrl+C, shutting down..." << std::endl;
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
    std::cout << "\n=== NIXL Telemetry Event ===" << std::endl;
    std::cout << "Timestamp: " << format_timestamp(event.timestamp_us) << std::endl;
    std::cout << "Transfer Statistics:" << std::endl;
    std::cout << "  TX Bytes: " << format_bytes(event.tx_bytes) << std::endl;
    std::cout << "  RX Bytes: " << format_bytes(event.rx_bytes) << std::endl;
    std::cout << "  TX Requests: " << event.tx_requests_num << std::endl;
    std::cout << "  RX Requests: " << event.rx_requests_num << std::endl;
    std::cout << "  Memory Registered: " << format_bytes(event.memory_registered) << " bytes"
              << std::endl;
    std::cout << "  Total Transaction Time: " << event.total_transaction_time_us << " us"
              << std::endl;
    std::cout << "  Error Counts:" << std::endl;

    for (int i = 0; i < std::abs(NIXL_ERR_LAST); i++) {
        // skip success since not captured
        if (i != 1) {
            std::cout << "    "
                      << nixlEnumStrings::statusStr(static_cast<nixl_status_t>((i - 1) * -1))
                      << ": " << event.error_counts[i] << std::endl;
        }
    }

    if (event.num_backends > 0) {
        std::cout << "Plugin Telemetry (" << event.num_backends << " backends):" << std::endl;
        for (uint16_t i = 0; i < event.num_backends; i++) {
            const auto &backend = event.backend_telemetry[i];
            std::cout << "  Backend: " << backend.plugin_name << std::endl;
            std::cout << "    Metrics (" << backend.num_metrics << "):" << std::endl;
            for (uint32_t j = 0; j < backend.num_metrics; j++) {
                const auto &metric = backend.metrics[j];
                std::cout << "      " << metric.name << ": " << metric.value << std::endl;
            }
        }
    }
    std::cout << "===========================" << std::endl;
}

bool
check_if_process_running(pid_t pid) {
    if (kill(pid, 0) == 0) {
        return true;
    }
    if (errno == EPERM) {
        return true;
    }
    return false;
}

std::string
look_for_stat_active_telemetry_files(const std::string &telemetry_path, bool read_any_file) {
    constexpr size_t module_name_size = std::strlen(TELEMETRY_PREFIX);
    constexpr size_t pid_offset = module_name_size + 1; // file name is like TELEMETRY_PREFIX.pid

    fs::path stats_path(telemetry_path);
    if (!fs::exists(stats_path) || !fs::is_directory(stats_path)) {
        std::cerr << "Cannot open directory " << stats_path.string() << std::endl;
        return "";
    }

    for (const auto &entry : fs::directory_iterator(stats_path)) {
        const auto &filename = entry.path().filename().string();
        // check file name starts with TELEMETRY_PREFIX
        if (filename.compare(0, module_name_size, TELEMETRY_PREFIX) == 0) {
            if (read_any_file) {
                return entry.path().string();
            }
            // get pid from file name
            auto pid_str = filename.substr(pid_offset);
            auto pid = std::stoi(pid_str);

            if (!check_if_process_running(pid)) continue;
            return entry.path().string();
        }
    }

    return "";
}

void
usage() {
    std::cout << "Usage: telemetry_reader_example <telemetry_folder_path> <read any file>"
              << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  <telemetry_folder_path>    Path to the telemetry folder" << std::endl;
    std::cout << "  <read any file>            Read telemetry data from any file (not only active "
                 "process) in the folder (0 - false, 1 - true, default: 0)"
              << std::endl;
    exit(0);
}

int
main(int argc, char *argv[]) {
    if (argc < 2 || argv[1] == std::string("-h") || argv[1] == std::string("--help")) {
        usage();
    }

    std::cout << "Telemetry path: " << argv[1] << std::endl;
    auto read_any_file = false;
    if (argc > 2) {
        read_any_file = std::stoi(argv[2]);
    }
    auto telemetry_path = argv[1];
    auto telemetry_file_name = look_for_stat_active_telemetry_files(telemetry_path, read_any_file);
    if (telemetry_file_name.empty()) {
        std::cerr << "No active telemetry files found" << std::endl;
        return 1;
    }

    // Set up signal handler for Ctrl+C
    signal(SIGINT, signal_handler);

    try {
        std::cout << "Opening telemetry buffer: " << telemetry_file_name << std::endl;
        std::cout << "Press Ctrl+C to stop reading telemetry..." << std::endl;

        // Open the shared memory buffer for reading
        SharedRingBuffer<nixlTelemetryEvent, TELEMETRY_BUFFER_SIZE> buffer(
            telemetry_file_name.c_str(), false, TELEMETRY_VERSION);

        std::cout << "Successfully opened telemetry buffer (version: " << buffer.get_version()
                  << ")" << std::endl;
        std::cout << "Buffer size: " << buffer.size() << " events" << std::endl;

        nixlTelemetryEvent event = {};
        uint64_t event_count = 0;

        while (g_running) {
            // Try to read an event from the buffer
            if (buffer.pop(event)) {
                event_count++;
                print_telemetry_event(event);
            } else {
                // No events available, sleep briefly
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
