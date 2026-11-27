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

#include "valkey_client.h"

#include "common/backend.h"
#include "common/exception.h"
#include "common/nixl_log.h"
#include "connection_request.pb.h" // generated from GLIDE's connection_request.proto

#include <array>
#include <chrono>
#include <thread>

// GLIDE's cbindgen header: no extern "C" guard and hundreds of unscoped
// enumerators, so it gets its own namespace.
namespace glide {
extern "C" {
#include <glide_ffi.h>
}
} // namespace glide

namespace {

const char *
errStr(const char *s) {
    return s ? s : "unknown error";
}

long long
blobLenReply(const glide::CommandResponse *r, const std::string &key) {
    using namespace nixl::valkey;
    if (!r || r->response_type == glide::Null) {
        return BLOB_MISSING;
    }
    if (r->response_type == glide::Int) {
        return r->int_value;
    }
    if (r->response_type == glide::Error) {
        const std::string msg(r->string_value, r->string_value_len);
        if (msg.find("WRONGTYPE") != std::string::npos) {
            return BLOB_MISSING;
        }
        NIXL_ERROR << "VALKEY: BLOB.INFO " << key << " failed: " << msg;
        return BLOB_ERROR;
    }
    NIXL_ERROR << "VALKEY: BLOB.INFO " << key << " returned reply type " << r->response_type;
    return BLOB_ERROR;
}

} // namespace

nixlValkeyConfig
nixlValkeyConfig::fromBackendParams(const nixl_b_params_t *params) {
    nixlValkeyConfig c;
    c.host = nixl::getBackendParamDefaulted(params, "host", c.host);
    c.port = nixl::getBackendParamDefaulted(params, "port", c.port);
    c.tls = nixl::getBackendParamDefaulted(params, "tls", c.tls);
    c.clusterMode = nixl::getBackendParamDefaulted(params, "cluster_mode", c.clusterMode);
    c.username = nixl::getBackendParamDefaulted(params, "username", c.username);
    c.password = nixl::getBackendParamDefaulted(params, "password", c.password);
    c.rdmaProvider = nixl::getBackendParamDefaulted(params, "rdma_provider", c.rdmaProvider);
    c.rdmaInterface = nixl::getBackendParamDefaulted(params, "rdma_interface", c.rdmaInterface);
    c.keyPrefix = nixl::getBackendParamDefaulted(params, "key_prefix", c.keyPrefix);
    c.timeoutMs = nixl::getBackendParamDefaulted(params, "timeout_ms", c.timeoutMs);
    c.transferTimeoutMs =
        nixl::getBackendParamDefaulted(params, "transfer_timeout_ms", c.transferTimeoutMs);
    c.numWorkers = nixl::getBackendParamDefaulted(params, "num_workers", c.numWorkers);
    c.writeChecksum = nixl::getBackendParamDefaulted(params, "write_checksum", c.writeChecksum);

    if (c.port <= 0 || c.port > 65535) {
        nixl::throwRuntimeError("VALKEY: invalid port ", c.port);
    }
    if (c.tls != "off" && c.tls != "on" && c.tls != "insecure") {
        nixl::throwRuntimeError("VALKEY: tls must be off, on or insecure, got ", c.tls);
    }
    if (c.rdmaProvider != "efa-direct" && c.rdmaProvider != "tcp") {
        nixl::throwRuntimeError("VALKEY: rdma_provider must be efa-direct or tcp, got ",
                                c.rdmaProvider);
    }
    if (!c.username.empty() && c.password.empty()) {
        nixl::throwRuntimeError("VALKEY: username requires a password");
    }
    if (c.timeoutMs <= 0) {
        nixl::throwRuntimeError("VALKEY: timeout_ms must be positive");
    }
    if (c.transferTimeoutMs < 0) {
        nixl::throwRuntimeError("VALKEY: transfer_timeout_ms must be 0 or positive");
    }
    if (c.numWorkers <= 0) {
        nixl::throwRuntimeError("VALKEY: num_workers must be positive");
    }
    return c;
}

namespace nixl::valkey {

bool
rdmaUsable() {
    return glide::rdma_usable();
}

client *
client::open(const nixlValkeyConfig &config, bool rdma) {
    connection_request::ConnectionRequest req;
    auto *addr = req.add_addresses();
    addr->set_host(config.host);
    addr->set_port(config.port);
    if (config.tls != "off") {
        req.set_tls_mode(config.tls == "on" ? connection_request::SecureTls :
                                              connection_request::InsecureTls);
    }
    req.set_cluster_mode_enabled(config.clusterMode);
    req.set_request_timeout(config.timeoutMs); // RDMA transfers have none
    req.set_connection_timeout(config.timeoutMs);
    if (!config.password.empty()) {
        auto *auth = req.mutable_authentication_info();
        auth->set_password(config.password);
        auth->set_username(config.username);
    }
    if (rdma) {
        auto *rdma_config = req.mutable_rdma_config();
        if (config.rdmaProvider == "tcp") {
            rdma_config->mutable_tcp();
        } else {
            rdma_config->mutable_efa_direct();
        }
        if (!config.rdmaInterface.empty()) {
            rdma_config->set_interface(config.rdmaInterface);
        }
    }
    const std::string bytes = req.SerializeAsString();

    glide::ClientType type = {};
    type.tag = glide::SyncClient;
    const glide::ConnectionResponse *resp = glide::create_client(
        reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size(), &type, nullptr, nullptr, 0);
    const void *handle = resp ? resp->conn_ptr : nullptr;
    if (!handle) {
        NIXL_ERROR << "VALKEY: connect to " << config.host << ":" << config.port
                   << " failed: " << errStr(resp ? resp->connection_error_message : nullptr);
    }
    if (resp) {
        glide::free_connection_response(const_cast<glide::ConnectionResponse *>(resp));
    }
    return handle ? new client(handle) : nullptr;
}

void
client::close() {
    glide::close_client(handle_);
    delete this;
}

std::vector<long long>
client::blobLens(const std::vector<std::string> &keys) {
    static const char cmd[] = "BLOB.INFO", field[] = "LEN";
    std::vector<long long> lens(keys.size(), BLOB_ERROR);
    if (keys.empty()) {
        return lens;
    }
    std::vector<std::array<const uint8_t *, 3>> args(keys.size());
    std::vector<std::array<uintptr_t, 3>> arg_lens(keys.size());
    std::vector<glide::CmdInfo> cmds(keys.size());
    std::vector<const glide::CmdInfo *> cmd_ptrs(keys.size());
    for (size_t i = 0; i < keys.size(); i++) {
        args[i] = {reinterpret_cast<const uint8_t *>(cmd),
                   reinterpret_cast<const uint8_t *>(keys[i].data()),
                   reinterpret_cast<const uint8_t *>(field)};
        arg_lens[i] = {sizeof(cmd) - 1, keys[i].size(), sizeof(field) - 1};
        cmds[i] = {glide::CustomCommand, args[i].data(), 3, arg_lens[i].data()};
        cmd_ptrs[i] = &cmds[i];
    }
    const glide::BatchInfo batch = {keys.size(), cmd_ptrs.data(), false};
    // raise_on_error=false: a per-key error comes back as that key's reply.
    glide::CommandResult *r = glide::batch(handle_, 0, &batch, false, nullptr, 0);

    if (!r) {
        NIXL_ERROR << "VALKEY: BLOB.INFO pipeline returned no result";
    } else if (r->command_error) {
        NIXL_ERROR << "VALKEY: BLOB.INFO pipeline failed: "
                   << errStr(r->command_error->command_error_message);
    } else if (!r->response || r->response->response_type != glide::Array ||
               r->response->array_value_len != (long)keys.size()) {
        NIXL_ERROR << "VALKEY: BLOB.INFO pipeline returned an unexpected reply";
    } else {
        for (size_t i = 0; i < keys.size(); i++) {
            lens[i] = blobLenReply(&r->response->array_value[i], keys[i]);
        }
    }
    glide::free_command_result(r);
    return lens;
}

long long
client::blobLen(const std::string &key) {
    return blobLens({key})[0];
}

glide::RdmaRegion *
client::registerHost(uintptr_t addr, size_t len) {
    glide::RdmaRegistration *r =
        glide::register_rdma_region(handle_, reinterpret_cast<uint8_t *>(addr), len);
    glide::RdmaRegion *region = r ? r->region : nullptr;
    if (!region) {
        NIXL_ERROR << "VALKEY: RDMA registration of " << len
                   << " bytes failed: " << errStr(r ? r->error_message : nullptr);
    }
    glide::free_rdma_registration(r);
    return region;
}

nixl_status_t
client::transfer(bool write,
                 const std::string &key,
                 glide::RdmaRegion *region,
                 size_t offset,
                 size_t len,
                 bool checksum,
                 bool &closed) {
    const auto *k = reinterpret_cast<const uint8_t *>(key.data());
    glide::RdmaResult *r = write ? glide::rdma_set(handle_,
                                                   k,
                                                   key.size(),
                                                   region,
                                                   offset,
                                                   len,
                                                   checksum ? glide::Compute : glide::Skip,
                                                   0) :
                                   glide::rdma_get(handle_, k, key.size(), region, offset, len);
    nixl_status_t st = NIXL_SUCCESS;

    if (r->error_message) {
        NIXL_ERROR << "VALKEY: " << (write ? "write " : "read ") << key
                   << " failed: " << r->error_message;
        st = NIXL_ERR_BACKEND;
    } else if (!write && !r->found) {
        st = NIXL_ERR_NOT_FOUND;
    } else if (!write && r->bytes_written != len) {
        NIXL_ERROR << "VALKEY: read " << key << " returned " << r->bytes_written
                   << " bytes, expected " << len;
        st = NIXL_ERR_MISMATCH;
    }
    closed = r->region_closed;
    glide::free_rdma_result(r);
    return st;
}

void
client::closeRegions() {
    if (char *err = glide::rdma_close_regions(handle_)) {
        // A region libfabric could not close keeps its transfer waiting.
        NIXL_ERROR << "VALKEY: cancelling transfers failed: " << err;
        glide::free_c_string(err);
    }
}

void
releaseRegion(glide::RdmaRegion *region) {
    // Only a transfer that ended early leaves a hold, and that drops the whole
    // client (closeWithRegions), so an idle region never has one.
    if (glide::RdmaMemoryHold *hold = glide::free_rdma_region(region)) {
        NIXL_ERROR << "VALKEY: GLIDE still holds the memory of a freed RDMA region"
                   << (glide::rdma_memory_leaked(hold) ? " (the region failed to close)" : "");
        glide::free_rdma_memory_hold(hold);
    }
}

void
closeWithRegions(client *c, const std::vector<glide::RdmaRegion *> &regions, int timeout_ms) {
    std::vector<glide::RdmaMemoryHold *> holds;
    for (auto *region : regions) {
        if (auto *h = glide::free_rdma_region(region)) {
            holds.push_back(h);
        }
    }
    if (c) {
        c->close();
    }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (auto *h : holds) {
        while (!glide::rdma_memory_released(h) && !glide::rdma_memory_leaked(h) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!glide::rdma_memory_released(h)) {
            NIXL_ERROR << "VALKEY: GLIDE still holds registered memory after closing the client"
                       << (glide::rdma_memory_leaked(h) ? " (a region failed to close)" : "");
        }
        glide::free_rdma_memory_hold(h);
    }
}

} // namespace nixl::valkey
