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

#include "obj_backend.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include <absl/strings/str_format.h>
#include <memory>
#include <future>
#include <optional>
#include <vector>
#include <chrono>
#include <algorithm>

namespace {

std::size_t
getNumThreads(nixl_b_params_t *custom_params) {
    return custom_params && custom_params->count("num_threads") > 0 ?
        std::stoul(custom_params->at("num_threads")) :
        std::max(1u, std::thread::hardware_concurrency() / 2);
}

struct MultipartParams {
    std::string upload_id;
    int part_number = 0;
    std::string object_key;
    bool is_multipart = false;
};

/**
 * Parse multipart upload parameters from descriptor metaInfo.
 * Expected format: "<object_key>;upload_id=<id>;part_number=<num>"
 * Or just: "<object_key>" for regular uploads
 * Returns a MultipartParams struct with is_multipart=true if both parameters are found.
 */
MultipartParams
parseMultipartParams(const std::string &metaInfo) {
    MultipartParams params;

    if (metaInfo.empty()) {
        return params;
    }

    // First element before semicolon is the object key
    size_t first_semi = metaInfo.find(';');
    if (first_semi == std::string::npos) {
        // No multipart params, just object key
        params.object_key = metaInfo;
        return params;
    }

    params.object_key = metaInfo.substr(0, first_semi);

    size_t upload_id_pos = metaInfo.find("upload_id=");
    if (upload_id_pos != std::string::npos) {
        size_t start = upload_id_pos + 10; // Length of "upload_id="
        size_t end = metaInfo.find(';', start);
        if (end == std::string::npos) {
            end = metaInfo.length();
        }
        params.upload_id = metaInfo.substr(start, end - start);
    }

    size_t part_number_pos = metaInfo.find("part_number=");
    if (part_number_pos != std::string::npos) {
        size_t start = part_number_pos + 12; // Length of "part_number="
        size_t end = metaInfo.find(';', start);
        if (end == std::string::npos) {
            end = metaInfo.length();
        }
        try {
            params.part_number = std::stoi(metaInfo.substr(start, end - start));
        }
        catch (const std::exception &e) {
            NIXL_WARN << "Failed to parse part_number from metaInfo: " << e.what();
            return params;
        }
    }

    // Valid multipart upload requires both upload_id and part_number
    if (!params.upload_id.empty() && params.part_number > 0 && params.part_number <= 10000) {
        params.is_multipart = true;
        NIXL_DEBUG << absl::StrFormat(
            "Parsed multipart params from metaInfo: object_key=%s, upload_id=%s, part_number=%d",
            params.object_key,
            params.upload_id,
            params.part_number);
    }

    return params;
}

bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (remote_agent != local_agent)
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be OBJ_SEG, got %d",
                                      remote.getType());
        return false;
    }

    return true;
}

class nixlObjBackendReqH : public nixlBackendReqH {
public:
    nixlObjBackendReqH() = default;
    ~nixlObjBackendReqH() = default;

    std::vector<std::future<nixl_status_t>> statusFutures_;

    nixl_status_t
    getOverallStatus() {
        while (!statusFutures_.empty()) {
            if (statusFutures_.back().wait_for(std::chrono::seconds(0)) ==
                std::future_status::ready) {
                auto current_status = statusFutures_.back().get();
                if (current_status != NIXL_SUCCESS) {
                    statusFutures_.clear();
                    return current_status;
                }
                statusFutures_.pop_back();
            } else {
                return NIXL_IN_PROG;
            }
        }
        return NIXL_SUCCESS;
    }
};

class nixlObjMetadata : public nixlBackendMD {
public:
    nixlObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(obj_key) {}

    ~nixlObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string objKey;
};

} // namespace

// -----------------------------------------------------------------------------
// Obj Engine Implementation
// -----------------------------------------------------------------------------

nixlObjEngine::nixlObjEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      executor_(std::make_shared<asioThreadPoolExecutor>(getNumThreads(init_params->customParams))),
      s3Client_(std::make_shared<awsS3Client>(init_params->customParams, executor_)) {

    NIXL_INFO << "Object storage backend initialized with S3 client wrapper";
}

// Used for testing to inject a mock S3 client dependency
nixlObjEngine::nixlObjEngine(const nixlBackendInitParams *init_params,
                             std::shared_ptr<iS3Client> s3_client)
    : nixlBackendEngine(init_params),
      executor_(std::make_shared<asioThreadPoolExecutor>(std::thread::hardware_concurrency())),
      s3Client_(s3_client) {
    s3Client_->setExecutor(executor_);
    NIXL_INFO << "Object storage backend initialized with injected S3 client";
}

nixlObjEngine::~nixlObjEngine() {
    executor_->WaitUntilStopped();
}

nixl_status_t
nixlObjEngine::registerMem(const nixlBlobDesc &mem,
                           const nixl_mem_t &nixl_mem,
                           nixlBackendMD *&out) {
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end())
        return NIXL_ERR_NOT_SUPPORTED;

    if (nixl_mem == OBJ_SEG) {
        std::string meta_info = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;

        std::unique_ptr<nixlObjMetadata> obj_md =
            std::make_unique<nixlObjMetadata>(nixl_mem, mem.devId, meta_info);

        size_t semi_pos = meta_info.find(';');
        std::string obj_key_only =
            (semi_pos != std::string::npos) ? meta_info.substr(0, semi_pos) : meta_info;
        devIdToObjKey_[mem.devId] = obj_key_only;

        out = obj_md.release();
    } else {
        out = nullptr;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlObjEngine::deregisterMem(nixlBackendMD *meta) {
    nixlObjMetadata *obj_md = static_cast<nixlObjMetadata *>(meta);
    if (obj_md) {
        std::unique_ptr<nixlObjMetadata> obj_md_ptr = std::unique_ptr<nixlObjMetadata>(obj_md);
        devIdToObjKey_.erase(obj_md->devId);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlObjEngine::queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const {
    resp.reserve(descs.descCount());

    try {
        for (auto &desc : descs) {
            // Check if metaInfo contains an upload_id query
            std::string upload_id_query;
            size_t upload_id_pos = desc.metaInfo.find("upload_id=");
            if (upload_id_pos != std::string::npos) {
                size_t start = upload_id_pos + 10;
                size_t end = desc.metaInfo.find(';', start);
                if (end == std::string::npos) {
                    end = desc.metaInfo.length();
                }
                upload_id_query = desc.metaInfo.substr(start, end - start);
            }

            // If querying for ETags by upload_id, return them without checking object existence
            if (!upload_id_query.empty()) {
                NIXL_DEBUG << "Querying ETags for upload_id: " << upload_id_query;
                nixl_b_params_t params;
                std::lock_guard<std::mutex> lock(etagsMutex_);
                NIXL_DEBUG << "Total upload_ids stored: " << uploadIdToETags_.size();
                auto etag_it = uploadIdToETags_.find(upload_id_query);
                if (etag_it != uploadIdToETags_.end() && !etag_it->second.empty()) {
                    std::string etags_str;
                    for (size_t i = 0; i < etag_it->second.size(); ++i) {
                        if (i > 0) etags_str += ",";
                        etags_str += etag_it->second[i];
                    }
                    params["etags"] = etags_str;
                    params["etag_count"] = std::to_string(etag_it->second.size());
                    resp.emplace_back(nixl_query_resp_t{params});
                } else {
                    resp.emplace_back(std::nullopt);
                }
                continue;
            }
            bool exists = s3Client_->checkObjectExists(desc.metaInfo);
            if (!exists) {
                resp.emplace_back(std::nullopt);
                continue;
            }

            resp.emplace_back(nixl_query_resp_t{nixl_b_params_t{}});
        }
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << "Failed to query memory: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlObjEngine::prepXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent))
        return NIXL_ERR_INVALID_PARAM;

    auto req_h = std::make_unique<nixlObjBackendReqH>();
    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlObjEngine::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    nixlObjBackendReqH *req_h = static_cast<nixlObjBackendReqH *>(handle);

    for (int i = 0; i < local.descCount(); ++i) {
        const auto &local_desc = local[i];
        const auto &remote_desc = remote[i];

        // Parse multipart upload parameters from remote descriptor's metaInfo
        nixlObjMetadata *obj_md = static_cast<nixlObjMetadata *>(remote_desc.metadataP);
        if (!obj_md) {
            NIXL_ERROR << "Remote descriptor metadata is null";
            return NIXL_ERR_INVALID_PARAM;
        }

        MultipartParams multipart_params = parseMultipartParams(obj_md->objKey);

        // Use the object key from parsed params (handles both regular and multipart)
        std::string obj_key =
            multipart_params.object_key.empty() ? obj_md->objKey : multipart_params.object_key;

        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->statusFutures_.push_back(status_promise->get_future());

        uintptr_t data_ptr = local_desc.addr;
        size_t data_len = local_desc.len;
        size_t offset = remote_desc.addr;

        // S3 client interface signals completion via a callback, but NIXL API polls request handle
        // for the status code. Use future/promise pair to bridge the gap.
        if (operation == NIXL_WRITE) {
            // Use multipart upload if parameters are provided
            if (multipart_params.is_multipart) {
                s3Client_->uploadPartAsync(
                    obj_key,
                    multipart_params.upload_id,
                    multipart_params.part_number,
                    data_ptr,
                    data_len,
                    [this,
                     status_promise,
                     key = obj_key,
                     upload_id = multipart_params.upload_id,
                     part_num = multipart_params.part_number,
                     len = data_len](bool success, const std::string &etag) {
                        if (success) {
                            NIXL_DEBUG << absl::StrFormat(
                                "OBJ MULTIPART WRITE SUCCESS: key=%s, upload_id=%s, part=%d, "
                                "size=%zu bytes, etag=%s",
                                key,
                                upload_id,
                                part_num,
                                len,
                                etag);
                            // Store ETag for later retrieval via queryMem, indexed by upload_id
                            {
                                std::lock_guard<std::mutex> lock(etagsMutex_);
                                uploadIdToETags_[upload_id].push_back(etag);
                                NIXL_DEBUG << absl::StrFormat("Stored ETag for upload_id=%s, total "
                                                              "ETags for this upload: %zu",
                                                              upload_id,
                                                              uploadIdToETags_[upload_id].size());
                            }
                            status_promise->set_value(NIXL_SUCCESS);
                        } else {
                            // Error details already logged in obj_s3_client.cpp callback
                            NIXL_ERROR << absl::StrFormat(
                                "OBJ MULTIPART WRITE CALLBACK: key=%s, upload_id=%s, part=%d, "
                                "size=%zu bytes - Transfer failed (see AWS SDK error above)",
                                key,
                                upload_id,
                                part_num,
                                len);
                            status_promise->set_value(NIXL_ERR_BACKEND);
                        }
                    });
            } else {
                // Use regular PutObject
                s3Client_->putObjectAsync(
                    obj_key,
                    data_ptr,
                    data_len,
                    offset,
                    [status_promise, key = obj_key, len = data_len](bool success) {
                        if (success) {
                            NIXL_DEBUG << absl::StrFormat(
                                "OBJ WRITE SUCCESS: key=%s, size=%zu bytes", key, len);
                            status_promise->set_value(NIXL_SUCCESS);
                        } else {
                            // Error details already logged in obj_s3_client.cpp callback
                            NIXL_ERROR
                                << absl::StrFormat("OBJ WRITE CALLBACK: key=%s, size=%zu bytes - "
                                                   "Transfer failed (see AWS SDK error above)",
                                                   key,
                                                   len);
                            status_promise->set_value(NIXL_ERR_BACKEND);
                        }
                    });
            }
        } else {
            // Read operation - multipart upload doesn't apply here
            s3Client_->getObjectAsync(
                obj_key,
                data_ptr,
                data_len,
                offset,
                [status_promise, key = obj_key, len = data_len](bool success) {
                    if (success) {
                        NIXL_DEBUG << absl::StrFormat(
                            "OBJ READ SUCCESS: key=%s, size=%zu bytes", key, len);
                        status_promise->set_value(NIXL_SUCCESS);
                    } else {
                        // Error details already logged in obj_s3_client.cpp callback
                        NIXL_ERROR << absl::StrFormat("OBJ READ CALLBACK: key=%s, size=%zu bytes - "
                                                      "Transfer failed (see AWS SDK error above)",
                                                      key,
                                                      len);
                        status_promise->set_value(NIXL_ERR_BACKEND);
                    }
                });
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlObjEngine::checkXfer(nixlBackendReqH *handle) const {
    nixlObjBackendReqH *req_h = static_cast<nixlObjBackendReqH *>(handle);
    return req_h->getOverallStatus();
}

nixl_status_t
nixlObjEngine::releaseReqH(nixlBackendReqH *handle) const {
    nixlObjBackendReqH *req_h = static_cast<nixlObjBackendReqH *>(handle);
    delete req_h;
    return NIXL_SUCCESS;
}
