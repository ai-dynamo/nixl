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

#ifndef NIXL_SRC_PLUGINS_UCX_REQUEST_H
#define NIXL_SRC_PLUGINS_UCX_REQUEST_H

#include "backend/backend_aux.h"
#include "common/nixl_log.h"

#include "types.h"
#include "ucx_enums.h"
#include "ucx_utils.h"

#include <optional>
#include <set>
#include <string>
#include <vector>

/****************************************
 * Backend request management
 *****************************************/

class nixlUcxBackendReqH : public nixlBackendReqH {
protected:
    std::set<ucx_connection_ptr_t> connections_;
    std::vector<nixlUcxReq> requests_;
    nixlUcxWorker *worker_;
    size_t workerId_;

    [[nodiscard]] nixl_status_t
    checkConnection(const nixl_status_t status = NIXL_SUCCESS) const {
        NIXL_ASSERT(!connections_.empty());
        for (const auto &conn : connections_) {
            const nixl_status_t conn_status = conn->getEp(workerId_)->checkTxState();
            if (conn_status != NIXL_SUCCESS) {
                return conn_status;
            }
        }
        return status;
    }

    void
    setWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(worker_ == nullptr || worker == nullptr);
        worker_ = worker;
        workerId_ = worker_id;
    }

public:
    // TODO: Separate out notification part for only read/write.
    // Notification to be sent after completion of all requests
    struct Notif {
        const std::string agent;
        const nixl_blob_t payload;

        Notif(const std::string &remote_agent, const nixl_blob_t &msg)
            : agent(remote_agent),
              payload(msg) {}
    };

    std::optional<Notif> notif;

    nixlUcxBackendReqH(nixlUcxWorker *worker, size_t worker_id)
        : worker_(worker),
          workerId_(worker_id) {}

    nixlUcxBackendReqH(nixlUcxWorker *worker, size_t worker_id, size_t size)
        : nixlUcxBackendReqH(worker, worker_id) {
        reserve(size);
    }

    void
    reserve(size_t size) {
        requests_.reserve(size);
        NIXL_ASSERT(connections_.empty());
    }

    [[nodiscard]] nixl_status_t
    append(nixl_status_t status, nixlUcxReq req, const ucx_connection_ptr_t &conn) {
        switch (status) {
        case NIXL_IN_PROG:
            requests_.push_back(req);
            connections_.insert(conn);
            break;
        case NIXL_SUCCESS:
            connections_.insert(conn);
            break;
        default:
            // Error. Release all previously initiated ops and exit:
            release();
            return status;
        }
        return NIXL_SUCCESS;
    }

    [[nodiscard]] const std::set<ucx_connection_ptr_t> &
    getConnections() const noexcept {
        return connections_;
    }

    [[nodiscard]] virtual bool
    isComposite() const noexcept {
        return false;
    }

    virtual void
    release() {
        // TODO: Error log: uncompleted requests found! Cancelling ...
        for (nixlUcxReq req : requests_) {
            const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (ret == NIXL_IN_PROG) {
                // TODO: Need process this properly.
                // it may not be enough to cancel UCX request
                worker_->reqCancel(req);
            }
            worker_->reqRelease(req);
        }
        requests_.clear();
        connections_.clear();
    }

    [[nodiscard]] virtual nixl_status_t
    status() {
        if (requests_.empty()) {
            /* No pending transmissions */
            connections_.clear();
            return NIXL_SUCCESS;
        }

        worker_->progressLoop();

        /* If last request is incomplete, return NIXL_IN_PROG early without
         * checking other requests */
        nixlUcxReq req = requests_.back();
        const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
        if (ret == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        } else if (ret != NIXL_SUCCESS) {
            return checkConnection(ret);
        }

        /* Last request completed successfully, all the others must be in the
         * same state. TODO: remove extra checks? */
        size_t incomplete_reqs = 0;
        nixl_status_t out_ret = NIXL_SUCCESS;
        for (nixlUcxReq req : requests_) {
            const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (ret == NIXL_SUCCESS) [[likely]] {
                worker_->reqRelease(req);
            } else if (ret == NIXL_IN_PROG) [[likely]] {
                if (out_ret == NIXL_SUCCESS) {
                    out_ret = NIXL_IN_PROG;
                }
                requests_[incomplete_reqs++] = req;
            } else {
                // Any other ret value is ERR and will be returned
                out_ret = checkConnection(ret);
            }
        }

        requests_.resize(incomplete_reqs);
        if (requests_.empty()) {
            connections_.clear();
        }
        return out_ret;
    }

    [[nodiscard]] nixlUcxWorker *
    getWorker() const noexcept {
        return worker_;
    }

    [[nodiscard]] size_t
    getWorkerId() const noexcept {
        return workerId_;
    }
};

namespace nixl::ucx {

// TODO: Check when and where ucp_am_data_release() needs to be called!!
// (When the request is deleted after the AM CB returned UCS_INPROGRESS
// but before ucp_am_recv_data_nbx() was called...)

struct recvRequestH : public nixlUcxBackendReqH {
    const nixl_meta_dlist_t &local; // initiatorDescs from the NIXL request

    // Operations on this kind of request can be called from the Agent
    // and from the progress thread via the active message callback.
    // The recursive is needed for when status() calls progress on the
    // worker which can also (?) trigger an active message callback.
    std::recursive_mutex mutex;

    size_t recvDataCalls = 0;
    size_t recvDataErrors = 0;

    std::optional<recv_map_t::iterator> iter;

    explicit recvRequestH(nixlUcxWorker *worker, size_t worker_id, const nixl_meta_dlist_t &local)
        : nixlUcxBackendReqH(worker, worker_id, local.descCount()),
          local(local) {}

    void
    append(nixlUcxReq req) {
        requests_.push_back(req);
    }

    void
    prefill(const ucx_connection_ptr_t &conn) {
        connections_.insert(conn);
    }

    [[nodiscard]] bool
    allCallsImpl() const noexcept {
        // If we have errors we don't care about completing everything.
        return (recvDataCalls == size_t(local.descCount())) || (recvDataErrors > 0);
    }

    void
    release() override {
        const std::lock_guard lg(mutex);
        nixlUcxBackendReqH::release();
    }

    // We cannot use nixlUcxBackendReqH::status() without modifications, it
    // assumes requests_.empty() means that we are done, but for RECV there
    // might be ucp_am_recv_data_nbx() calls that we have yet to make.

    [[nodiscard]] nixl_status_t
    status() override {
        const std::lock_guard lg(mutex);
        return statusImpl(true);
    }

    [[nodiscard]] nixl_status_t
    statusImpl(const bool progress) {
        if (requests_.empty()) {
            if (allCallsImpl()) {
                connections_.clear();
                return recvDataErrors ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
            }
            return NIXL_IN_PROG;
        }

        if (progress) {
            worker_->progressLoop();
        }

        /* If last request is incomplete, return NIXL_IN_PROG early without
         * checking other requests */
        const nixlUcxReq req = requests_.back();
        const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
        if (ret == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        } else if (ret != NIXL_SUCCESS) {
            return checkConnection(ret);
        }

        /* Last request completed successfully, all the others must be in the
         * same state. TODO: remove extra checks? */
        size_t incomplete_reqs = 0;
        nixl_status_t out_ret = NIXL_SUCCESS;
        for (nixlUcxReq req : requests_) {
            const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (__builtin_expect(ret == NIXL_SUCCESS, 0)) {
                worker_->reqRelease(req);
            } else if (ret == NIXL_IN_PROG) {
                if (out_ret == NIXL_SUCCESS) {
                    out_ret = NIXL_IN_PROG;
                }
                requests_[incomplete_reqs++] = req;
            } else {
                // Any other ret value is ERR and will be returned
                out_ret = checkConnection(ret);
            }
        }

        requests_.resize(incomplete_reqs);

        if (!allCallsImpl()) {
            return NIXL_IN_PROG;
        }

        if (requests_.empty()) {
            connections_.clear();
        }
        return out_ret;
    }
};

} // namespace nixl::ucx

#endif
