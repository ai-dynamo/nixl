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
getAvailableNetworkDevices() {
    std::vector<std::string> all_devices;
    std::string provider_name;

    std::unordered_map<std::string, std::vector<std::string>> provider_device_map;
    struct fi_info *hints, *info;
    hints = fi_allocinfo();
    if (!hints) {
        NIXL_ERROR << "Failed to allocate fi_info";
        return {"none", {}};
    }

    hints->caps = 0;
    hints->caps = FI_MSG | FI_RMA; // Basic messaging and RMA

    hints->caps |= FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->mode = FI_CONTEXT;
    hints->ep_attr->type = FI_EP_RDM;

    int ret = fi_getinfo(FI_VERSION(1, 9), NULL, NULL, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo failed " << fi_strerror(-ret);
        fi_freeinfo(hints);
        return {"none", {}};
    }

    // Process devices for this provider
    for (struct fi_info *cur = info; cur; cur = cur->next) {
        if (cur->domain_attr && cur->domain_attr->name && cur->fabric_attr &&
            cur->fabric_attr->name) {

            std::string device_name = cur->domain_attr->name;
            std::string provider_name = cur->fabric_attr->prov_name;

            NIXL_TRACE << "Found device - domain: " << device_name
                       << ", provider: " << provider_name << ", ep_type: " << cur->ep_attr->type
                       << ", caps: 0x" << std::hex << cur->caps << std::dec;

            if (provider_device_map.find(provider_name) == provider_device_map.end()) {
                provider_device_map[provider_name] = {};
            }
            provider_device_map[provider_name].push_back(device_name);
        }
    }

    fi_freeinfo(info);
    fi_freeinfo(hints);

    for (auto device_list : provider_device_map) {
        for (auto device : device_list.second) {
            NIXL_TRACE << "Provider: " << device_list.first << ", Device: " << device;
        }
    }

    if (provider_device_map.find("efa") != provider_device_map.end()) {
        return {"efa", provider_device_map["efa"]};
    } else if (provider_device_map.find("sockets") != provider_device_map.end()) {
        return {"sockets", {provider_device_map["sockets"][0]}};
    }

    NIXL_WARN << "No network devices found with any provider";
    return {"none", {}};
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

// Thread-safe atomic counters for optimized ID generation
static std::atomic<uint16_t> g_xfer_id_counter{1}; // 16-bit XFER_ID counter, start from 1
static std::atomic<uint8_t> g_seq_id_counter{0}; // 4-bit SEQ_ID counter, start from 0

uint16_t
getNextXferId() {
    uint16_t xfer_id = g_xfer_id_counter.fetch_add(1);

    // Handle wraparound: 16-bit field can hold 0 to 65,535
    if (xfer_id > NIXL_XFER_ID_MASK) {
        // Reset counter atomically and get a fresh ID
        uint16_t expected = xfer_id;
        while (expected > NIXL_XFER_ID_MASK &&
               !g_xfer_id_counter.compare_exchange_weak(expected, 1)) {
            expected = g_xfer_id_counter.load();
        }
        xfer_id = g_xfer_id_counter.fetch_add(1);
        // Ensure we don't exceed the mask after reset
        if (xfer_id > NIXL_XFER_ID_MASK) {
            xfer_id = 1;
        }
    }

    return xfer_id;
}

uint8_t
getNextSeqId() {
    uint8_t seq_id = g_seq_id_counter.fetch_add(1);

    // Handle wraparound: 4-bit field can hold 0 to 15
    if (seq_id > NIXL_SEQ_ID_MASK) {
        // Reset counter atomically and get a fresh ID
        uint8_t expected = seq_id;
        while (expected > NIXL_SEQ_ID_MASK &&
               !g_seq_id_counter.compare_exchange_weak(expected, 0)) {
            expected = g_seq_id_counter.load();
        }
        seq_id = g_seq_id_counter.fetch_add(1);
        // Ensure we don't exceed the mask after reset
        if (seq_id > NIXL_SEQ_ID_MASK) {
            seq_id = 0;
        }
    }

    return seq_id;
}

void
resetSeqId() {
    // Reset SEQ_ID counter for new postXfer
    g_seq_id_counter.store(0);
}

} // namespace LibfabricUtils
