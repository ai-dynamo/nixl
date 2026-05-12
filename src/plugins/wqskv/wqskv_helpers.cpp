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

#include "wqskv_helpers.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <absl/strings/str_format.h>
#include <json/json.h>

#include "common/nixl_log.h"

namespace wqskv {

std::string
resolveConfigPath(const nixlBackendInitParams *init_params) {
    if (init_params != nullptr && init_params->customParams != nullptr) {
        auto it = init_params->customParams->find("config_path");
        if (it != init_params->customParams->end() && !it->second.empty()) {
            return it->second;
        }
    }
    if (const char *env = std::getenv("WDS_BACKEND_CONFIG_PATH")) {
        if (env[0] != '\0') {
            return std::string(env);
        }
    }
    return {};
}

bool
loadKVCacheOptionsFromJson(const std::string &conf_path, KVCacheOptions &opts) {
    std::ifstream json_file(conf_path, std::ifstream::binary);
    if (!json_file.is_open()) {
        NIXL_ERROR << "WQSKV: open config failed: " << conf_path << " errno=" << errno << " ("
                   << std::strerror(errno) << ")";
        return false;
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;
    if (!Json::parseFromStream(builder, json_file, &root, &errs)) {
        NIXL_ERROR << "WQSKV: parse config failed: " << conf_path << " err=" << errs;
        return false;
    }

    opts.poolid = root["poolid"].asUInt();
    opts.thread_num = root["thread_num"].asInt();

    const std::string mode = root["thread_mode"].asString();
    if (mode == "poll") {
        opts.thread_mode = KVCACHE_THREAD_POLL;
    } else if (mode == "event") {
        opts.thread_mode = KVCACHE_THREAD_EVENT;
    } else {
        NIXL_ERROR << "WQSKV: invalid thread_mode: " << mode;
        return false;
    }

    opts.conf = root["wengine_conf"].asString();
    opts.nid = root["node_id"].asUInt();
    opts.bvar_dummy_port = root["bvar_port"].asUInt();
    opts.mem_size = root["mem_size"].asUInt();

    if (root.isMember("mempool_objsz")) {
        opts.mempool_obj_sz = root["mempool_objsz"].asUInt();
    }
    if (root.isMember("mempool_objnum")) {
        opts.mempool_obj_num = root["mempool_objnum"].asUInt();
    }
    if (root.isMember("use_round_robin")) {
        opts.use_round_robin = root["use_round_robin"].asBool();
    }
    if (root.isMember("wds_debug_log")) {
        opts.debug_log = root["wds_debug_log"].asBool();
    }

    const std::string cpus = root["bind_cpus"].asString();
    std::stringstream ss(cpus);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            opts.bind_cpus.push_back(std::stoi(token));
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "WQSKV: invalid bind_cpus token '" << token << "': " << e.what();
            return false;
        }
    }

    NIXL_INFO << "WQSKV: parsed config " << conf_path << " poolid=" << opts.poolid
              << " thread_num=" << opts.thread_num << " thread_mode=" << opts.thread_mode
              << " conf=" << opts.conf << " nid=" << opts.nid
              << " bvar_port=" << opts.bvar_dummy_port << " mem_size=" << opts.mem_size
              << " bind_cpus=" << cpus;
    return true;
}

bool
parseCustomParamKeys(const std::string &blob, int expected_count, std::vector<std::string> &out) {
    if (blob.empty() || expected_count <= 0) {
        return false;
    }
    out.clear();
    out.reserve(expected_count);
    size_t pos = 0;
    while (pos <= blob.size()) {
        size_t next = blob.find('\n', pos);
        if (next == std::string::npos) {
            out.emplace_back(blob.substr(pos));
            break;
        }
        out.emplace_back(blob.substr(pos, next - pos));
        pos = next + 1;
    }
    if (static_cast<int>(out.size()) != expected_count) {
        out.clear();
        return false;
    }
    return true;
}

bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("WQSKV: invalid operation type: %d", operation);
        return false;
    }
    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("WQSKV: local mem type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }
    if (remote.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("WQSKV: remote mem type must be DRAM_SEG, got %d",
                                      remote.getType());
        return false;
    }
    return true;
}

} // namespace wqskv
