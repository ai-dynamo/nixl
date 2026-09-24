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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CUFILE_NICS_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CUFILE_NICS_H

#include <string>
#include <vector>

/**
 * Readers for the two cufile.json keys the connector reuses, so a host already
 * configured for GPUDirect Storage RDMA needs no extra settings. Kept free of
 * any JSON library: only the two keys are needed.
 */

/**
 * Strip // line comments from JSONC content (cufile.json allows them). NIC
 * values and the DC key never contain "//", and commented-out duplicate keys in
 * other sections vanish, so the real keys are found.
 */
inline std::string
stripLineComments(const std::string &jsonc) {
    std::string s;
    s.reserve(jsonc.size());
    size_t line_start = 0;
    while (line_start <= jsonc.size()) {
        size_t nl = jsonc.find('\n', line_start);
        size_t line_end = (nl == std::string::npos) ? jsonc.size() : nl;
        std::string line = jsonc.substr(line_start, line_end - line_start);
        size_t comment = line.find("//");
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        s += line;
        s += '\n';
        if (nl == std::string::npos) {
            break;
        }
        line_start = nl + 1;
    }
    return s;
}

/**
 * Extract the "rdma_dev_addr_list" entries from cufile.json content. Entries are
 * NIC IPv4 addresses or RDMA device names (e.g. "mlx5_1"). Returns an empty
 * vector if the key is absent or the file is empty.
 */
inline std::vector<std::string>
parseRdmaDevAddrList(const std::string &jsonc) {
    const std::string s = stripLineComments(jsonc);

    std::vector<std::string> out;
    size_t key = s.find("rdma_dev_addr_list");
    if (key == std::string::npos) {
        return out;
    }
    size_t lb = s.find('[', key);
    if (lb == std::string::npos) {
        return out;
    }
    size_t rb = s.find(']', lb);
    if (rb == std::string::npos) {
        return out;
    }

    // Extract quoted tokens within [ ... ].
    size_t p = lb + 1;
    while (p < rb) {
        size_t q1 = s.find('"', p);
        if (q1 == std::string::npos || q1 >= rb) {
            break;
        }
        size_t q2 = s.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 > rb) {
            break;
        }
        out.push_back(s.substr(q1 + 1, q2 - q1 - 1));
        p = q2 + 1;
    }
    return out;
}

/**
 * Extract the "rdma_dc_key" value (a quoted hex string, e.g. "0xffeeddcc") from
 * cufile.json content. Returns "" if the key is absent (commented out by
 * default in cufile.json, in which case the caller uses its own default).
 */
inline std::string
parseRdmaDcKey(const std::string &jsonc) {
    const std::string s = stripLineComments(jsonc);
    size_t key = s.find("rdma_dc_key");
    if (key == std::string::npos) {
        return std::string();
    }
    size_t colon = s.find(':', key);
    if (colon == std::string::npos) {
        return std::string();
    }
    size_t q1 = s.find('"', colon);
    if (q1 == std::string::npos) {
        return std::string();
    }
    size_t q2 = s.find('"', q1 + 1);
    if (q2 == std::string::npos) {
        return std::string();
    }
    return s.substr(q1 + 1, q2 - q1 - 1);
}

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CUFILE_NICS_H
