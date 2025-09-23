/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
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

#include "libfabric_common.h"
#include "common/nixl_log.h"

#include <iomanip>
#include <sstream>
#include <atomic>
#include <cstring>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

namespace LibfabricUtils {


std::pair<std::string, std::vector<std::string>>
getAvailableEfaDevices() {
    std::unordered_map<std::string, std::vector<std::string>> provider_devices_map;
    std::vector<std::string> all_efa_devices;
    std::string fabric_name;
    struct fi_info *hints, *info;
    hints = fi_allocinfo();
    if (!hints) {
        NIXL_ERROR << "Failed to allocate fi_info for device discovery";
        return {fabric_name, all_efa_devices};
    }

    // Important to initialize this to allow differentiation between EFA and EFA-Direct
    hints->mode = ~0;

    // Set required capabilities - let libfabric select the best provider
    hints->caps = FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE |
        FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->fabric_attr->prov_name = strdup("efa");
    hints->ep_attr->type = FI_EP_RDM;

    int ret = fi_getinfo(FI_VERSION(1, 9), NULL, NULL, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo failed during device discovery: " << fi_strerror(-ret);
        fi_freeinfo(hints);
        return {fabric_name, all_efa_devices};
    }

    // Process providers and filter for EFA providers with RMA capabilities
    for (struct fi_info *cur = info; cur; cur = cur->next) {
        if (cur->domain_attr && cur->domain_attr->name && cur->fabric_attr &&
            cur->fabric_attr->name) {

            std::string device_name = cur->domain_attr->name;
            std::string provider_name = cur->fabric_attr->name;

            // Add device to the appropriate provider's vector
            provider_devices_map[provider_name].push_back(device_name);

            NIXL_TRACE << "Found EFA device: " << device_name << " with provider: " << provider_name
                       << " (caps: 0x" << std::hex << cur->caps << std::dec << ")";
        }
    }

    fi_freeinfo(info);
    fi_freeinfo(hints);

    // Extract device names from the map, prioritizing efa-direct over efa
    all_efa_devices.clear();
    if (provider_devices_map.find("efa-direct") != provider_devices_map.end()) {
        all_efa_devices = provider_devices_map["efa-direct"];
        fabric_name = "efa-direct";
        NIXL_TRACE << "Using efa-direct provider with " << all_efa_devices.size() << " devices";
    } else if (provider_devices_map.find("efa") != provider_devices_map.end()) {
        all_efa_devices = provider_devices_map["efa"];
        fabric_name = "efa";
        NIXL_TRACE << "Using efa provider with " << all_efa_devices.size() << " devices";
    }

    return {fabric_name, all_efa_devices};
}

std::string
hexdump(const void *data) {
    static constexpr uint HEXDUMP_MAX_LENGTH = 56;
    std::stringstream ss;
    ss.str().reserve(HEXDUMP_MAX_LENGTH * 3);
    const unsigned char *bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < HEXDUMP_MAX_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]) << " ";
    }
    return ss.str();
}

// Simple counter for pre-allocation only
static uint32_t g_xfer_id_counter = 1; // Start from 1, 0 reserved for special cases

std::vector<uint32_t>
preallocateXferIds(size_t count) {
    std::vector<uint32_t> xfer_ids;
    xfer_ids.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        uint32_t xfer_id = g_xfer_id_counter++;

        // Handle wraparound: 20-bit field can hold 0 to 1,048,575
        if (xfer_id > NIXL_XFER_ID_MASK) {
            // Reset counter and try again
            g_xfer_id_counter = 1;
            xfer_id = 1;
            g_xfer_id_counter = 2; // Update for next iteration
        }

        xfer_ids.push_back(xfer_id);
    }

    return xfer_ids;
}

} // namespace LibfabricUtils
