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

#include "doca_memos_backend.h"

#include "common/nixl_log.h"
#include "nixl_types.h"

#include <absl/strings/str_format.h>
#include <charconv>
#include <iomanip>
#include <limits>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cstring>

// DOCA headers
#include <doca_kvdev.h>
#include <doca_nvme_kernel_kvdev.h>
#include <doca_pe.h>
#include <doca_ctx.h>
#include <doca_error.h>

namespace {

// Metadata for OBJ_SEG memory
class nixlDocaMemosMetadata : public nixlBackendMD {
public:
    nixlDocaMemosMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, const docaMemosKey &key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(key) {}

    ~nixlDocaMemosMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    docaMemosKey objKey;
};

bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Invalid operation type: %d", operation);
        return false;
    }

    if (remote_agent != local_agent) {
        NIXL_WARN << absl::StrFormat("Remote agent doesn't match the requesting agent (%s). Got %s",
                                     local_agent,
                                     remote_agent);
    }

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Remote memory type must be OBJ_SEG, got %d",
                                      remote.getType());
        return false;
    }

    return true;
}

} // anonymous namespace

std::ostream &
operator<<(std::ostream &os, const docaMemosKey &k) {
    std::ios::fmtflags flags(os.flags());
    char fill = os.fill();
    os << std::hex << std::setfill('0');
    for (uint32_t i = 0; i < k.keyLen && i < DOCA_MEMOS_MAX_OBJECT_KEY_LEN; ++i) {
        os << std::setw(2) << static_cast<unsigned>(k.key[i]);
    }
    os.flags(flags);
    os.fill(fill);
    return os;
}

void
nixlDocaMemosEngine::cleanupDocaResources() {
    doca_error_t result;

    // doca_kvdev_stop() is only valid on a started device. _set_nguid() /
    // _set_path() failures during init leave the device in the unstarted
    // state, so guard with a started check rather than blindly stopping.
    if (kvdev_) {
        uint8_t started = 0;
        if (doca_kvdev_is_started(kvdev_, &started) == DOCA_SUCCESS && started) {
            result = doca_kvdev_stop(kvdev_);
            if (result != DOCA_SUCCESS) {
                NIXL_WARN << "Failed to stop DOCA KV device: " << doca_error_get_descr(result);
            }
        }
        kvdev_ = nullptr;
    }

    if (nvmeKvdev_) {
        result = doca_nvme_kernel_kvdev_destroy(nvmeKvdev_);
        if (result != DOCA_SUCCESS) {
            NIXL_WARN << "Failed to destroy NVMe kernel KV device: "
                      << doca_error_get_descr(result);
        }
        nvmeKvdev_ = nullptr;
    }
}

nixl_status_t
nixlDocaMemosEngine::parseInitParams(const nixl_b_params_t *params) {
    if (!params) {
        return NIXL_SUCCESS;
    }

    auto it = params->find("device_name");
    if (it != params->end()) {
        deviceName_ = it->second;
    }

    it = params->find("num_tasks");
    if (it != params->end()) {
        try {
            const unsigned long long parsed = std::stoull(it->second);
            if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
                NIXL_ERROR << "num_tasks '" << it->second << "' out of range (1.."
                           << std::numeric_limits<uint32_t>::max() << ")";
                return NIXL_ERR_INVALID_PARAM;
            }
            numTasks_ = static_cast<uint32_t>(parsed);
        }
        catch (...) {
            NIXL_ERROR << "Failed to parse num_tasks parameter";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    it = params->find("nguid");
    if (it != params->end()) {
        nguid_ = it->second;
    }

    it = params->find("ignore_read_not_found");
    if (it != params->end()) {
        if (it->second == "true") {
            ignoreReadNotFound_ = true;
        } else if (it->second == "false") {
            ignoreReadNotFound_ = false;
        } else {
            NIXL_ERROR << "Invalid ignore_read_not_found '" << it->second
                       << "', expected 'true' or 'false'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    it = params->find("query_mem_mode");
    if (it != params->end()) {
        if (it->second == "actual") {
            queryMemAssumeSuccess_ = false;
        } else if (it->second == "assume_success") {
            queryMemAssumeSuccess_ = true;
        } else {
            NIXL_ERROR << "Invalid query_mem_mode '" << it->second
                       << "', expected 'assume_success' or 'actual'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    if (!params->count("nguid")) {
        NIXL_WARN << "Using default nguid (all zeros); set 'nguid' to override";
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaMemosEngine::initDocaDevice() {
    uint32_t max_path_len = 0;
    doca_error_t result = doca_nvme_kernel_kvdev_cap_get_max_path_len(&max_path_len);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "doca_nvme_kernel_kvdev_cap_get_max_path_len failed: "
                   << doca_error_get_descr(result);
        return NIXL_ERR_BACKEND;
    }
    if (deviceName_.size() + 1 > max_path_len) {
        NIXL_ERROR << "device_name length " << deviceName_.size() << " exceeds device-reported max "
                   << (max_path_len - 1);
        return NIXL_ERR_INVALID_PARAM;
    }

    uint8_t nguid_bytes[DOCA_KVDEV_NGUID_LEN] = {0};
    if (!nguid_.empty()) {
        if (nguid_.length() != 32) {
            NIXL_ERROR << "Invalid nguid '" << nguid_ << "' (expected 32 hex chars)";
            return NIXL_ERR_INVALID_PARAM;
        }
        for (size_t i = 0; i < DOCA_KVDEV_NGUID_LEN; i++) {
            unsigned val = 0;
            const char *first = nguid_.data() + i * 2;
            auto [ptr, ec] = std::from_chars(first, first + 2, val, 16);
            if (ec != std::errc{} || ptr != first + 2) {
                NIXL_ERROR << "Invalid nguid '" << nguid_ << "' (expected 32 hex chars)";
                return NIXL_ERR_INVALID_PARAM;
            }
            nguid_bytes[i] = static_cast<uint8_t>(val);
        }
    }

    result = doca_nvme_kernel_kvdev_create(&nvmeKvdev_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create DOCA NVMe kernel KV device: "
                   << doca_error_get_descr(result);
        return NIXL_ERR_BACKEND;
    }

    result = doca_nvme_kernel_kvdev_set_path(nvmeKvdev_, deviceName_.c_str());
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set NVMe kernel KV device path: " << doca_error_get_descr(result);
        cleanupDocaResources();
        return NIXL_ERR_BACKEND;
    }

    kvdev_ = doca_nvme_kernel_kvdev_as_kvdev(nvmeKvdev_);
    if (!kvdev_) {
        NIXL_ERROR << "Failed to convert NVMe kernel KV device to generic KV device";
        cleanupDocaResources();
        return NIXL_ERR_BACKEND;
    }

    result = doca_kvdev_set_nguid(kvdev_, nguid_bytes);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set DOCA KV device NGUID: " << doca_error_get_descr(result);
        cleanupDocaResources();
        return NIXL_ERR_BACKEND;
    }

    result = doca_kvdev_start(kvdev_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to start DOCA KV device: " << doca_error_get_descr(result);
        cleanupDocaResources();
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << "DOCA KV device initialized successfully";

    uint32_t dev_max_tasks = 0;
    result = doca_kvdev_get_max_tasks(kvdev_, &dev_max_tasks);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "doca_kvdev_get_max_tasks failed: " << doca_error_get_descr(result);
        cleanupDocaResources();
        return NIXL_ERR_BACKEND;
    }
    if (dev_max_tasks > 0 && numTasks_ > dev_max_tasks) {
        NIXL_INFO << "Clamping num_tasks from " << numTasks_ << " to device max " << dev_max_tasks;
        numTasks_ = dev_max_tasks;
    }

    // Plugin stores keys inline in a fixed-size array sized to the current
    // DOCA spec (DOCA_MEMOS_MAX_OBJECT_KEY_LEN). If the device ever reports a
    // larger maximum, that capacity must be revisited.
    uint16_t dev_max_key_len = 0;
    result = doca_kvdev_get_max_key_len(kvdev_, &dev_max_key_len);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "doca_kvdev_get_max_key_len failed: " << doca_error_get_descr(result);
        cleanupDocaResources();
        return NIXL_ERR_BACKEND;
    }
    if (dev_max_key_len > DOCA_MEMOS_MAX_OBJECT_KEY_LEN) {
        NIXL_ERROR << "Device max key length " << dev_max_key_len << " exceeds plugin capacity "
                   << DOCA_MEMOS_MAX_OBJECT_KEY_LEN;
        cleanupDocaResources();
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaMemosEngine::createProgressEngine(const nixlBackendInitParams *init_params) {
    try {
        if (init_params->enableProgTh) {
            if (init_params->pthrDelay == 0) {
                NIXL_WARN << "Progress-thread delay is 0us; thread will busy-spin";
            }
            progressEngine_ = std::make_unique<nixlThreadedProgressEngine>(
                nvmeKvdev_, numTasks_, std::chrono::microseconds(init_params->pthrDelay));
        } else {
            progressEngine_ = std::make_unique<nixlNoThreadProgressEngine>(nvmeKvdev_, numTasks_);
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create progress engine: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    if (progressEngine_->hasInitError()) {
        NIXL_ERROR << "Progress engine initialization failed";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixlDocaMemosEngine::nixlDocaMemosEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {

    NIXL_INFO << "Initializing DOCA KV backend";

    if (parseInitParams(init_params->customParams) != NIXL_SUCCESS) {
        initErr = true;
        return;
    }

    if (deviceName_.empty()) {
        NIXL_ERROR << "DOCA KV backend requires 'device_name' parameter to be set";
        NIXL_ERROR << "Example: params[\"device_name\"] = \"/dev/nvme0n1\"";
        initErr = true;
        return;
    }

    NIXL_INFO << "Initializing DOCA KV with device_name=" << deviceName_
              << ", num_tasks=" << numTasks_ << ", nguid=" << nguid_
              << ", query_mem_mode=" << (queryMemAssumeSuccess_ ? "assume_success" : "actual")
              << ", ignore_read_not_found=" << (ignoreReadNotFound_ ? "true" : "false");

    if (initDocaDevice() != NIXL_SUCCESS) {
        initErr = true;
        return;
    }

    NIXL_INFO << "Creating progress engine";
    if (createProgressEngine(init_params) != NIXL_SUCCESS) {
        cleanupDocaResources();
        initErr = true;
        return;
    }

    NIXL_INFO << "DOCA KV backend initialized successfully";
}

nixlDocaMemosEngine::~nixlDocaMemosEngine() {
    NIXL_INFO << "Destroying DOCA KV backend";

    progressEngine_.reset();

    cleanupDocaResources();

    NIXL_INFO << "DOCA KV backend destroyed";
}

bool
nixlDocaMemosEngine::convertToMemosKey(const std::string &meta_info, docaMemosKey &key) {
    if (meta_info.empty()) {
        return false;
    }
    if (meta_info.size() > DOCA_MEMOS_MAX_OBJECT_KEY_LEN * 2) {
        return false;
    }
    if (meta_info.size() & 1) {
        return false;
    }
    auto minfo = meta_info.data();
    key.keyLen = meta_info.size() / 2;
    for (uint32_t i = 0; i < key.keyLen; i++) {
        unsigned val = 0;
        auto [ptr, ec] = std::from_chars(minfo + i * 2, minfo + i * 2 + 2, val, 16);
        if (ec != std::errc{} || ptr != minfo + i * 2 + 2) {
            return false;
        }
        key.key[i] = static_cast<uint8_t>(val);
    }
    return true;
}

// Resolves meta_info to a device key via one of three paths. The hex-first /
// raw-fallback behaviour means the same meta_info string may take different
// paths depending on its contents, so log the chosen path at DEBUG level.
bool
nixlDocaMemosEngine::resolveMemosKey(uint64_t dev_id,
                                     const std::string &meta_info,
                                     docaMemosKey &key) {
    if (meta_info.empty()) {
        key.keyLen = sizeof(dev_id);
        memcpy(key.key, &dev_id, key.keyLen);
        NIXL_DEBUG << "resolveMemosKey: empty meta_info, using dev_id bytes";
        return true;
    }
    if (convertToMemosKey(meta_info, key)) {
        NIXL_DEBUG << "resolveMemosKey: parsed meta_info as hex (" << key.keyLen << " bytes)";
        return true;
    }
    if (meta_info.size() <= DOCA_MEMOS_MAX_OBJECT_KEY_LEN) {
        key.keyLen = meta_info.size();
        memcpy(key.key, meta_info.data(), key.keyLen);
        NIXL_DEBUG << "resolveMemosKey: using meta_info as raw bytes (" << key.keyLen << " bytes)";
        return true;
    }
    return false;
}

nixl_status_t
nixlDocaMemosEngine::registerMem(const nixlBlobDesc &mem,
                                 const nixl_mem_t &nixl_mem,
                                 nixlBackendMD *&out) {
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (nixl_mem == OBJ_SEG) {
        docaMemosKey key;
        if (!resolveMemosKey(mem.devId, mem.metaInfo, key)) {
            NIXL_ERROR << "Failed to convert metaInfo to docaMemosKey: " << mem.metaInfo;
            return NIXL_ERR_INVALID_PARAM;
        }

        auto kv_md = std::make_unique<nixlDocaMemosMetadata>(nixl_mem, mem.devId, key);

        NIXL_DEBUG << "Registered OBJ_SEG memory with devId: " << mem.devId;
        out = kv_md.release();
    } else {
        out = nullptr;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaMemosEngine::deregisterMem(nixlBackendMD *meta) {
    nixlDocaMemosMetadata *kv_md = static_cast<nixlDocaMemosMetadata *>(meta);
    if (kv_md) {
        NIXL_DEBUG << "Deregistered memory with devId: " << kv_md->devId;
    }
    delete kv_md;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaMemosEngine::queryMem(const nixl_reg_dlist_t &descs,
                              std::vector<nixl_query_resp_t> &resp) const {
    if (queryMemAssumeSuccess_) {
        resp.reserve(descs.descCount());
        for (int i = 0; i < descs.descCount(); i++) {
            resp.emplace_back(nixl_b_params_t{});
        }
        return NIXL_SUCCESS;
    }

    return progressEngine_->queryMem(descs, resp);
}

nixl_status_t
nixlDocaMemosEngine::prepXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH *&handle,
                              const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    int desc_count = local.descCount();
    if (desc_count == 0) {
        NIXL_ERROR << "Empty descriptor lists";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (remote.descCount() != desc_count) {
        NIXL_ERROR << "Descriptor count mismatch: local=" << desc_count
                   << " remote=" << remote.descCount();
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *req_h = new nixlDocaMemosBackendReqH(desc_count);
    req_h->valueIovecs_.resize(desc_count);
    req_h->taskContexts_.resize(desc_count);
    if (operation == NIXL_READ) {
        req_h->ignoreNotFound_ = ignoreReadNotFound_;
    }

    for (int i = 0; i < desc_count; i++) {
        const auto &local_desc = local[i];
        req_h->valueIovecs_[i] = {reinterpret_cast<void *>(local_desc.addr), local_desc.len};
    }

    req_h->objectKeys_.clear();
    req_h->objectKeys_.reserve(desc_count);

    for (int i = 0; i < desc_count; i++) {
        const auto &remote_desc = remote[i];
        auto *kv_md = static_cast<nixlDocaMemosMetadata *>(remote_desc.metadataP);
        if (!kv_md) {
            NIXL_ERROR << "No metadata for remote descriptor at index " << i;
            delete req_h;
            return NIXL_ERR_INVALID_PARAM;
        }
        req_h->objectKeys_.push_back(kv_md->objKey);
    }

    handle = req_h;

    NIXL_DEBUG << "Prepared transfer: operation=" << operation << ", " << desc_count << " tasks";

    return NIXL_SUCCESS;
}

nixl_status_t
nixlDocaMemosEngine::postXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH *&handle,
                              const nixl_opt_b_args_t *opt_args) const {
    auto req_h = static_cast<nixlDocaMemosBackendReqH *>(handle);
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << "Unsupported operation type: " << operation;
        return NIXL_ERR_INVALID_PARAM;
    }

    return progressEngine_->postXfer(req_h, operation, local, remote);
}

nixl_status_t
nixlDocaMemosEngine::checkXfer(nixlBackendReqH *handle) const {
    auto req_h = static_cast<nixlDocaMemosBackendReqH *>(handle);
    return progressEngine_->checkXfer(req_h);
}

nixl_status_t
nixlDocaMemosEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto req_h = static_cast<nixlDocaMemosBackendReqH *>(handle);
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_DEBUG << "Releasing request handle with " << req_h->totalTasks_ << " tasks";

    progressEngine_->cancelRequest(req_h);
    return NIXL_SUCCESS;
}
