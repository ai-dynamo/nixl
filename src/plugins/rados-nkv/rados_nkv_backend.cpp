/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
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

#include "rados_nkv_backend.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

#include "common/nixl_log.h"

extern "C" {
#include "kv_host_shim.h"
}

// -----------------------------------------------------------------------------
// Per-descriptor metadata for an OBJ_SEG (remote KV-key) registration.
// -----------------------------------------------------------------------------
namespace {

class nixlRadosNkvMetadata : public nixlBackendMD {
public:
    explicit nixlRadosNkvMetadata(std::vector<uint8_t> key)
        : nixlBackendMD(true),
          key(std::move(key)) {}

    ~nixlRadosNkvMetadata() override = default;

    std::vector<uint8_t> key;
};

// Synchronous request handle: the shim ops complete inline, so we just stash
// the final status produced by postXfer and report it back in checkXfer.
class nixlRadosNkvBackendReqH : public nixlBackendReqH {
public:
    nixlRadosNkvBackendReqH() = default;
    ~nixlRadosNkvBackendReqH() override = default;

    nixl_status_t status = NIXL_IN_PROG;
};

// Parse "true"/"1"/"yes"/"on" (case-insensitive) as boolean true.
bool
parseBool(const std::string &v) {
    std::string s;
    s.reserve(v.size());
    for (char c : v) {
        s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return s == "true" || s == "1" || s == "yes" || s == "on";
}

} // namespace

// -----------------------------------------------------------------------------
// nixlRadosNkvEngine
// -----------------------------------------------------------------------------

nixlRadosNkvEngine::nixlRadosNkvEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    // The vfio-user socket directory is taken from the backend's custom params.
    // Accept a couple of common spellings for convenience.
    std::string vfu_addr;
    for (const char *k : {"vfu_addr", "socket", "vfio_user_path", "device"}) {
        if (getInitParam(k, vfu_addr) == NIXL_SUCCESS && !vfu_addr.empty()) {
            break;
        }
        vfu_addr.clear();
    }

    if (vfu_addr.empty()) {
        NIXL_ERROR << "RADOS_NKV: missing required custom param 'vfu_addr' "
                      "(the VFIOUSER transport directory)";
        initErr = true;
        return;
    }

    std::string nsid_str;
    uint32_t nsid = 0; // 0 selects the first CSI==KV namespace
    if (getInitParam("nsid", nsid_str) == NIXL_SUCCESS && !nsid_str.empty()) {
        try {
            nsid = static_cast<uint32_t>(std::stoul(nsid_str));
        }
        catch (const std::exception &e) {
            NIXL_WARN << "RADOS_NKV: bad nsid '" << nsid_str << "', using auto-select";
            nsid = 0;
        }
    }

    // init_env defaults to false: in production the host/agent owns the SPDK
    // env, which lets multiple engines coexist in one process (the future
    // nixlAgent path). Standalone tests with no host env pass init_env=true so
    // the shim brings up its own (no-hugepage, single-instance) SPDK env.
    bool init_env = false;
    std::string init_env_str;
    if (getInitParam("init_env", init_env_str) == NIXL_SUCCESS && !init_env_str.empty()) {
        init_env = parseBool(init_env_str);
    }

    struct kv_host_shim_opts opts = {};
    opts.opts_size = sizeof(opts);
    opts.name = "nixl_rados_nkv";
    opts.vfu_addr = vfu_addr.c_str();
    opts.nsid = nsid;
    opts.init_env = init_env;

    int rc = kv_host_shim_open(&opts, &shim_);
    if (rc != 0 || shim_ == nullptr) {
        NIXL_ERROR << "RADOS_NKV: kv_host_shim_open(" << vfu_addr << ") failed: rc=" << rc;
        shim_ = nullptr;
        initErr = true;
        return;
    }

    // Clamp the derived-key length to the namespace-advertised kvkml so the
    // hashed key (radosNkvDeriveKey) always fits the device key space. A
    // namespace that advertises kvkml==0 reports no usable key-length limit;
    // fall back to kMaxKeyLen rather than clamping to 0 (which would reject
    // every key).
    uint32_t shim_kvkml = kv_host_shim_max_key_len(shim_);
    if (shim_kvkml == 0) {
        NIXL_WARN << "RADOS_NKV: namespace advertised kvkml=0 (no key-length limit "
                     "reported); using default max key length "
                  << static_cast<unsigned>(kMaxKeyLen);
        shim_kvkml = kMaxKeyLen;
    }
    maxKeyLen_ = static_cast<uint8_t>(std::min<uint32_t>(kMaxKeyLen, shim_kvkml));

    NIXL_INFO << "RADOS_NKV: opened SPDK KV shim on " << vfu_addr
              << " (init_env=" << (init_env ? "true" : "false") << ", max_key=" << shim_kvkml
              << ", effective_max_key=" << static_cast<unsigned>(maxKeyLen_)
              << ", max_value=" << kv_host_shim_max_value_len(shim_) << ")";
}

nixlRadosNkvEngine::~nixlRadosNkvEngine() {
    if (shim_) {
        kv_host_shim_close(shim_);
        shim_ = nullptr;
    }
}

nixl_mem_list_t
nixlRadosNkvEngine::getSupportedMems() const {
    // Mirror obj: local DRAM source, remote OBJ-style key-addressed dest.
    return {DRAM_SEG, OBJ_SEG};
}

nixl_status_t
nixlRadosNkvEngine::registerMem(const nixlBlobDesc &mem,
                                const nixl_mem_t &nixl_mem,
                                nixlBackendMD *&out) {
    if (nixl_mem != DRAM_SEG && nixl_mem != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (nixl_mem == OBJ_SEG) {
        // The remote descriptor carries the token sequence in metaInfo; derive
        // the fixed-length KV key by hashing it. The key lives in the per-desc
        // metadata and is read back from the descriptor at transfer time.
        std::vector<uint8_t> key;
        if (!radosNkvDeriveKey(mem.metaInfo, maxKeyLen_, key)) {
            NIXL_ERROR << "RADOS_NKV: empty token sequence (metaInfo); cannot derive a KV key";
            return NIXL_ERR_INVALID_PARAM;
        }
        out = new nixlRadosNkvMetadata(std::move(key));
    } else {
        // Local DRAM registration: nothing to do for the staging skeleton (we
        // stage through a per-request shim DMA buffer in postXfer). No backend
        // metadata is needed for the DRAM side.
        out = nullptr;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosNkvEngine::deregisterMem(nixlBackendMD *meta) {
    delete static_cast<nixlRadosNkvMetadata *>(meta);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosNkvEngine::queryMem(const nixl_reg_dlist_t &descs,
                             std::vector<nixl_query_resp_t> &resp) const {
    // Mirror the OBJ backend's queryMem result/absence convention exactly:
    //   - resp is sized to descCount() and defaulted to std::nullopt (absent).
    //   - present  => resp[i] = nixl_query_resp_t{nixl_b_params_t{}} (engaged).
    //   - absent   => resp[i] = std::nullopt (left as the default).
    //   - on a backend/transport error we return an error status (NIXL_ERR_*)
    //     rather than encoding the failure as "absent", so callers never mask a
    //     transport failure as a cache miss.
    resp.assign(descs.descCount(), std::nullopt);

    if (!shim_) {
        NIXL_ERROR << "RADOS_NKV: shim not initialized";
        return NIXL_ERR_BACKEND;
    }

    for (int i = 0; i < descs.descCount(); ++i) {
        const auto &desc = descs[i];

        // Derive the KV key from the OBJ_SEG descriptor's token sequence using
        // the SAME hash path as registerMem.
        std::vector<uint8_t> key;
        if (!radosNkvDeriveKey(desc.metaInfo, maxKeyLen_, key)) {
            NIXL_ERROR << "RADOS_NKV: empty token sequence (metaInfo) in queryMem descriptor " << i;
            return NIXL_ERR_INVALID_PARAM;
        }

        int rc = kv_host_shim_exist(shim_, key.data(), static_cast<uint8_t>(key.size()));
        if (rc == 0) {
            // Present.
            resp[i] = nixl_query_resp_t{nixl_b_params_t{}};
        } else if (rc == 0x87) {
            // KEY_DOES_NOT_EXIST: absent. Leave resp[i] as std::nullopt.
            resp[i] = std::nullopt;
        } else {
            // rc < 0: a submit-/transport-level error. Surface it as an error,
            // NOT as a miss, so failures are not silently masked as absent.
            NIXL_ERROR << "RADOS_NKV: kv_host_shim_exist failed for descriptor " << i
                       << ": rc=" << rc;
            return NIXL_ERR_BACKEND;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosNkvEngine::prepXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << "RADOS_NKV: invalid operation " << operation;
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << "RADOS_NKV: local memory type must be DRAM_SEG, got " << local.getType();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << "RADOS_NKV: remote memory type must be OBJ_SEG, got " << remote.getType();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "RADOS_NKV: local/remote descriptor count mismatch";
        return NIXL_ERR_INVALID_PARAM;
    }

    handle = new nixlRadosNkvBackendReqH();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosNkvEngine::postXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    if (!handle) {
        NIXL_ERROR << "RADOS_NKV: transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (!shim_) {
        NIXL_ERROR << "RADOS_NKV: shim not initialized";
        return NIXL_ERR_BACKEND;
    }
    auto *req_h = static_cast<nixlRadosNkvBackendReqH *>(handle);

    for (int i = 0; i < local.descCount(); ++i) {
        const auto &local_desc = local[i];
        const auto &remote_desc = remote[i];

        // Cross-check the local and remote descriptor lengths. The op uses the
        // local length for the DMA staging buffer and the KV value size; a
        // mismatch means the caller's view of the value size disagrees, so fail
        // rather than silently store/retrieve a different number of bytes.
        if (local_desc.len != remote_desc.len) {
            NIXL_ERROR << "RADOS_NKV: descriptor " << i
                       << " length mismatch: local=" << local_desc.len
                       << " remote=" << remote_desc.len;
            req_h->status = NIXL_ERR_INVALID_PARAM;
            return NIXL_ERR_INVALID_PARAM;
        }

        // The KV key lives in the descriptor's registration metadata (set by
        // registerMem). Read it from there rather than re-deriving or keying on
        // devId, so the transfer uses exactly the key the descriptor registered.
        auto *md = static_cast<nixlRadosNkvMetadata *>(remote_desc.metadataP);
        if (!md) {
            NIXL_ERROR << "RADOS_NKV: remote descriptor " << i
                       << " has no registered KV-key metadata";
            req_h->status = NIXL_ERR_INVALID_PARAM;
            return NIXL_ERR_INVALID_PARAM;
        }
        const std::vector<uint8_t> &key = md->key;

        const auto data_ptr = reinterpret_cast<void *>(local_desc.addr);
        const size_t data_len = local_desc.len;

        // Stage through an SPDK-DMA buffer (see header for the rationale).
        void *dma = kv_host_shim_dma_alloc(data_len);
        if (!dma) {
            NIXL_ERROR << "RADOS_NKV: DMA buffer alloc failed (" << data_len << " bytes)";
            req_h->status = NIXL_ERR_BACKEND;
            return NIXL_ERR_BACKEND;
        }

        int rc = 0;
        if (operation == NIXL_WRITE) {
            std::memcpy(dma, data_ptr, data_len);
            rc = kv_host_shim_store(shim_,
                                    key.data(),
                                    static_cast<uint8_t>(key.size()),
                                    dma,
                                    static_cast<uint32_t>(data_len));
            if (rc != 0) {
                NIXL_ERROR << "RADOS_NKV: kv_host_shim_store failed: rc=" << rc;
            }
        } else { // NIXL_READ
            uint32_t value_len_out = 0;
            rc = kv_host_shim_retrieve(shim_,
                                       key.data(),
                                       static_cast<uint8_t>(key.size()),
                                       dma,
                                       static_cast<uint32_t>(data_len),
                                       &value_len_out);
            if (rc == 0) {
                // value_len_out is the device's TRUE value length and may exceed
                // buf_len (data truncated). The transfer is sized to data_len
                // (local==remote, cross-checked above), so any size difference
                // means the request cannot be satisfied exactly: a larger value
                // would be truncated (silent data loss) and a shorter one would
                // leave the tail of the caller's buffer undefined. Surface that
                // as an error instead of copying a partial value.
                if (value_len_out != data_len) {
                    NIXL_ERROR << "RADOS_NKV: stored value length " << value_len_out
                               << " != requested transfer length " << data_len << " for descriptor "
                               << i << "; refusing partial copy";
                    rc = -1;
                } else {
                    std::memcpy(data_ptr, dma, data_len);
                }
            } else {
                NIXL_ERROR << "RADOS_NKV: kv_host_shim_retrieve failed: rc=" << rc;
            }
        }

        kv_host_shim_dma_free(dma);

        if (rc != 0) {
            req_h->status = NIXL_ERR_BACKEND;
            return NIXL_ERR_BACKEND;
        }
    }

    // The shim ops are synchronous, so the transfer is already complete.
    req_h->status = NIXL_SUCCESS;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosNkvEngine::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        NIXL_ERROR << "RADOS_NKV: transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    return static_cast<nixlRadosNkvBackendReqH *>(handle)->status;
}

nixl_status_t
nixlRadosNkvEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        NIXL_ERROR << "RADOS_NKV: transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    delete static_cast<nixlRadosNkvBackendReqH *>(handle);
    return NIXL_SUCCESS;
}
