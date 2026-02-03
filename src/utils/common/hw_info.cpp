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

#include "hw_info.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "nixl_log.h"

namespace nixl {

namespace {

    constexpr const char *kPciDevicePath = "/sys/bus/pci/devices";
    constexpr unsigned long kPciVendorMellanox = 0x15b3;
    constexpr unsigned long kPciVendorNvidia = 0x10de;
    constexpr unsigned long kPciClassIb = 0x0207;
    constexpr unsigned long kPciClassGpuDisplay = 0x0300;
    constexpr unsigned long kPciClassGpu3d = 0x0302;

    [[nodiscard]] std::optional<unsigned long>
    readSysfsUlong(const std::filesystem::path &sysfs_path,
                   std::string_view file_name,
                   std::string_view device_name) noexcept {
        std::ifstream file(sysfs_path / file_name);
        if (!file.is_open()) {
            NIXL_TRACE << "Failed to open " << file_name << " for device " << device_name;
            return std::nullopt;
        }

        std::string value;
        if (!std::getline(file, value)) {
            NIXL_TRACE << "Failed to read " << file_name << " for device " << device_name;
            return std::nullopt;
        }

        try {
            return std::stoul(value, nullptr, 0);
        }
        catch (const std::exception &e) {
            NIXL_TRACE << "Failed to parse " << file_name << " for device " << device_name << ": "
                       << e.what();
            return std::nullopt;
        }
    }

} // namespace

hwInfo::hwInfo() {
    std::error_code ec;
    std::filesystem::directory_iterator dir_iter(kPciDevicePath, ec);
    if (ec) {
        throw std::runtime_error("Failed to scan PCI devices directory: " + ec.message());
    }

    for (const auto &entry : dir_iter) {
        const std::string device_name = entry.path().filename().string();

        // Skip hidden entries
        if (device_name.empty() || device_name[0] == '.') {
            continue;
        }

        const std::filesystem::path device_path = entry.path();

        // Read vendor ID
        const auto vendor_id = readSysfsUlong(device_path, "vendor", device_name);
        if (!vendor_id) {
            continue;
        }

        // Read class ID
        auto class_id = readSysfsUlong(device_path, "class", device_name);
        if (!class_id) {
            continue;
        }
        *class_id >>= 8;

        // Check for InfiniBand device
        if ((*vendor_id == kPciVendorMellanox) && (*class_id == kPciClassIb)) {
            numIbDevices++;
            NIXL_DEBUG << "Found IB device #" << numIbDevices << ": " << device_name << " vendor=0x"
                       << std::hex << *vendor_id << " class=0x" << *class_id << std::dec;
        }

        // Check for GPU
        if ((*vendor_id == kPciVendorNvidia) &&
            ((*class_id == kPciClassGpuDisplay) || (*class_id == kPciClassGpu3d))) {
            numNvidiaGpus++;
            NIXL_DEBUG << "Found GPU #" << numNvidiaGpus << ": " << device_name << " vendor=0x"
                       << std::hex << *vendor_id << " class=0x" << *class_id << std::dec;
        }
    }
}

} // namespace nixl
