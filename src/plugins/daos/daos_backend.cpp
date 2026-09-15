/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "daos_backend.h"

#include <atomic>
#include <cerrno>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/backend.h"
#include "common/nixl_log.h"

namespace {

struct daosOperation {
    uintptr_t data_ptr;
    size_t data_len;
    uint64_t offset;
    std::string path;
    std::shared_ptr<iDfsObject> object;
};

class nixlDaosMetadata final : public nixlBackendMD {
public:
    nixlDaosMetadata(uint64_t dev_id, std::string path)
        : nixlBackendMD(true),
          devId(dev_id),
          path(std::move(path)) {}

    uint64_t devId;
    std::string path;
};

class nixlDaosBackendReqH final : public nixlBackendReqH {
public:
    nixlDaosBackendReqH(nixl_xfer_op_t operation, std::vector<daosOperation> operations)
        : operation_(operation),
          operations_(std::move(operations)) {}

    nixl_status_t
    poll() {
        const State state = state_.load(std::memory_order_acquire);
        if (state == State::Ready) {
            return NIXL_SUCCESS;
        }
        if (state == State::Done) {
            return static_cast<nixl_status_t>(final_status_.load(std::memory_order_acquire));
        }
        if (state == State::Submitting) {
            return NIXL_IN_PROG;
        }
        if (remaining_.load(std::memory_order_acquire) != 0) {
            return NIXL_IN_PROG;
        }
        state_.store(State::Done, std::memory_order_release);
        return static_cast<nixl_status_t>(final_status_.load(std::memory_order_acquire));
    }

    void
    complete(nixl_status_t status) {
        if (status != NIXL_SUCCESS) {
            final_status_.store(status, std::memory_order_relaxed);
        }
        remaining_.fetch_sub(1, std::memory_order_release);
    }

    enum class State { Ready, Submitting, InProgress, Done };
    std::atomic<State> state_{State::Ready};
    std::atomic<size_t> remaining_{0};
    std::atomic<int> final_status_{NIXL_SUCCESS};
    std::mutex submissionMutex_;
    size_t nextOperation_ = 0;
    nixl_xfer_op_t operation_;
    std::vector<daosOperation> operations_;
};

std::string
objectPath(const nixlBlobDesc &mem) {
    return mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
}

nixl_status_t
validateTransfer(const nixl_xfer_op_t operation,
                 const nixl_meta_dlist_t &local,
                 const nixl_meta_dlist_t &remote,
                 const std::string &remote_agent,
                 const std::string &local_agent) {
    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote_agent != local_agent) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.getType() != DRAM_SEG || remote.getType() != OBJ_SEG) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.descCount() == 0 || local.descCount() != remote.descCount()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    for (int i = 0; i < local.descCount(); ++i) {
        if (local[i].len != remote[i].len || remote[i].metadataP == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }
    return NIXL_SUCCESS;
}

nixlDaosBackendReqH *
castHandle(nixlBackendReqH *handle) {
    return static_cast<nixlDaosBackendReqH *>(handle);
}

void
submitPending(iDfsClient &client, nixlDaosBackendReqH *request) {
    std::lock_guard<std::mutex> lock(request->submissionMutex_);
    while (request->nextOperation_ < request->operations_.size()) {
        const auto &op = request->operations_[request->nextOperation_];
        auto completion =
            [request, path = op.path, expected_size = op.data_len](int rc, size_t bytes_read) {
                if (rc == 0 && bytes_read != expected_size) {
                    rc = ENODATA;
                }
                if (rc != 0) {
                    NIXL_ERROR << "libdfs transfer failed for '" << path << "': " << rc;
                }
                request->complete(rc == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
            };

        int rc;
        try {
            if (request->operation_ == NIXL_WRITE) {
                rc = client.submitWrite(op.object, op.data_ptr, op.data_len, op.offset, completion);
            } else {
                rc = client.submitRead(op.object, op.data_ptr, op.data_len, op.offset, completion);
            }
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "libdfs submission threw for '" << op.path << "': " << e.what();
            rc = EIO;
        }
        catch (...) {
            NIXL_ERROR << "libdfs submission threw for '" << op.path << "'";
            rc = EIO;
        }

        if (rc == EAGAIN) {
            return;
        }
        ++request->nextOperation_;
        if (rc != 0) {
            NIXL_ERROR << "libdfs submission failed for '" << op.path << "': " << rc;
            request->complete(NIXL_ERR_BACKEND);
        }
    }
}

} // namespace

nixl_b_params_t
nixlDaosEngine::getPluginParams() {
    return {{"pool", ""},
            {"container", ""},
            {"system", ""},
            {"read_only", "false"},
            {"create_container", "false"},
            {"chunk_size", "0"},
            {"oclass_id", "0"},
            {"object_class", ""},
            {"object_class_hint", ""},
            {"num_event_queues", "1"},
            {"max_inflight_per_queue", "1024"},
            {"submission_batch_size", "32"},
            {"completion_batch_size", "128"},
            {"progress_poll_timeout_us", "1000"},
            {"progress_cpu_affinity", ""}};
}

nixlDaosEngine::nixlDaosEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      client_(makeLibDfsClient(init_params->customParams)) {
    NIXL_INFO << "DAOS backend initialized with native libdfs events";
}

nixlDaosEngine::nixlDaosEngine(const nixlBackendInitParams *init_params,
                               std::shared_ptr<iDfsClient> client)
    : nixlBackendEngine(init_params),
      client_(std::move(client)) {
    if (!client_) {
        throw std::invalid_argument("DAOS client must not be null");
    }
}

nixlDaosEngine::~nixlDaosEngine() = default;

nixl_status_t
nixlDaosEngine::registerMem(const nixlBlobDesc &mem,
                            const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) {
    out = nullptr;
    if (nixl_mem == DRAM_SEG) {
        return NIXL_SUCCESS;
    }
    if (nixl_mem != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        out = new nixlDaosMetadata(mem.devId, objectPath(mem));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "DAOS object registration failed: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlDaosEngine::deregisterMem(nixlBackendMD *meta) {
    delete static_cast<nixlDaosMetadata *>(meta);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDaosEngine::queryMem(const nixl_reg_dlist_t &descs,
                         std::vector<nixl_query_resp_t> &resp) const {
    if (descs.getType() != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    resp.assign(descs.descCount(), std::nullopt);

    bool failed = false;
    for (int i = 0; i < descs.descCount(); ++i) {
        bool exists = false;
        const int rc = client_->exists(objectPath(descs[i]), exists);
        if (rc != 0) {
            NIXL_ERROR << "DAOS query failed for '" << objectPath(descs[i]) << "': " << rc;
            failed = true;
        } else if (exists) {
            resp[i] = nixl_b_params_t{};
        }
    }
    return failed ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t
nixlDaosEngine::prepXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *) const {
    handle = nullptr;
    const nixl_status_t status =
        validateTransfer(operation, local, remote, remote_agent, localAgent);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    std::vector<daosOperation> operations;
    operations.reserve(local.descCount());
    for (int i = 0; i < local.descCount(); ++i) {
        const auto *metadata = static_cast<const nixlDaosMetadata *>(remote[i].metadataP);
        std::shared_ptr<iDfsObject> object;
        const int rc = client_->open(metadata->path, operation == NIXL_WRITE, object);
        if (rc != 0 || !object) {
            NIXL_ERROR << "libdfs open failed for '" << metadata->path << "': " << rc;
            return NIXL_ERR_BACKEND;
        }
        operations.push_back(
            {local[i].addr, local[i].len, remote[i].addr, metadata->path, std::move(object)});
    }
    handle = new nixlDaosBackendReqH(operation, std::move(operations));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDaosEngine::postXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &,
                         const nixl_meta_dlist_t &,
                         const std::string &,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *request = castHandle(handle);
    if (operation != request->operation_) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto expected = nixlDaosBackendReqH::State::Ready;
    if (!request->state_.compare_exchange_strong(
            expected, nixlDaosBackendReqH::State::Submitting, std::memory_order_acq_rel)) {
        expected = nixlDaosBackendReqH::State::Done;
        if (!request->state_.compare_exchange_strong(
                expected, nixlDaosBackendReqH::State::Submitting, std::memory_order_acq_rel)) {
            return NIXL_ERR_NOT_ALLOWED;
        }
    }
    {
        std::lock_guard<std::mutex> lock(request->submissionMutex_);
        request->nextOperation_ = 0;
    }
    request->remaining_.store(request->operations_.size(), std::memory_order_relaxed);
    request->final_status_.store(NIXL_SUCCESS, std::memory_order_relaxed);
    request->state_.store(nixlDaosBackendReqH::State::InProgress, std::memory_order_release);

    submitPending(*client_, request);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlDaosEngine::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *request = castHandle(handle);
    if (request->state_.load(std::memory_order_acquire) == nixlDaosBackendReqH::State::InProgress) {
        submitPending(*client_, request);
    }
    return request->poll();
}

nixl_status_t
nixlDaosEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *request = castHandle(handle);
    // daos_event_abort does not currently cancel internal DAOS operations. Keep
    // the request, object handles, and caller buffers alive through completion.
    if (request->state_.load(std::memory_order_acquire) == nixlDaosBackendReqH::State::InProgress) {
        submitPending(*client_, request);
    }
    if (request->poll() == NIXL_IN_PROG) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    delete request;
    return NIXL_SUCCESS;
}
