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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_REQ_H
#define NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_REQ_H

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "backend/backend_engine.h"
#include "common/nixl_log.h"

#include "ucx_backend.h"
#include "ucx_sgl.h"
#include "ucx_utils.h"

class nixlUcxBackendReqH : public nixlBackendReqH {
public:
    // Notification to be sent over the bound connection after completion of all requests.
    // Empty if there is no pending notification.
    std::string notif;

#ifdef HAVE_UCX_SGL_API
    std::optional<nixl::ucx::sglXfer> sgl;
#endif

    explicit nixlUcxBackendReqH(nixlUcxWorker *worker) {
        setWorker(worker);
    }

    void
    init(const ucx_connection_ptr_t &conn, const nixlUcxEp &ep) {
        NIXL_ASSERT(requests_.empty());
        requests_.reserve(max_requests);
        conn_ = conn;
        ep_ = &ep;
    }

    [[nodiscard]] const nixlUcxEp &
    getEp() const {
        NIXL_ASSERT(ep_ != nullptr);
        return *ep_;
    }

    void
    resetStatus() {
        NIXL_ASSERT(requests_.empty());
        error_ = NIXL_SUCCESS;
        notif.clear();
    }

    void
    setError(nixl_status_t status) {
        NIXL_ASSERT(status < 0);
        if (error_ == NIXL_SUCCESS) {
            error_ = status;
        }
    }

    void
    reserve(size_t size) {
        NIXL_ASSERT(requests_.empty());
        requests_.reserve(size);
    }

    [[nodiscard]] nixl_status_t
    append(nixl_status_t status, nixlUcxReq req) {
        if (status == NIXL_IN_PROG) [[likely]] {
            requests_.push_back(req);
        } else if (status != NIXL_SUCCESS) {
            // Previously posted operations must finish before this error is reported.
            setError(status);
            return status;
        }

        return NIXL_SUCCESS;
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
        reset();
    }

    [[nodiscard]] virtual nixl_status_t
    status() {
        if (requests_.empty()) {
            /* No pending transmissions */
            return error_;
        }

        worker_->progressLoop();

        /* If last request is incomplete, return NIXL_IN_PROG early without
         * checking other requests */
        nixlUcxReq req = requests_.back();
        const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
        if (ret == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        }

        size_t incomplete_reqs = 0;
        for (nixlUcxReq req : requests_) {
            const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (ret == NIXL_IN_PROG) {
                requests_[incomplete_reqs++] = req;
            } else {
                if (ret != NIXL_SUCCESS) [[unlikely]] {
                    setError(checkConnection(ret));
                }
                worker_->reqRelease(req);
            }
        }

        requests_.resize(incomplete_reqs);
        if (!requests_.empty()) {
            return NIXL_IN_PROG;
        }
        return error_;
    }

    [[nodiscard]] nixlUcxWorker *
    getWorker() const noexcept {
        return worker_;
    }

    [[nodiscard]] size_t
    getWorkerId() const noexcept {
        return worker_->getId();
    }

protected:
    void
    setWorker(nixlUcxWorker *worker) {
        NIXL_ASSERT(worker_ == nullptr || worker == nullptr);
        worker_ = worker;
    }

private:
    // Initial reservation for SGL transfers; scalar batches reserve more.
    static constexpr size_t max_requests = 3;

    void
    reset() noexcept {
        requests_.clear();
        conn_.reset();
        ep_ = nullptr;
    }

    [[nodiscard]] nixl_status_t
    checkConnection(const nixl_status_t status = NIXL_SUCCESS) const {
        NIXL_ASSERT(ep_ != nullptr);
        const nixl_status_t conn_status = ep_->checkTxState();
        return (conn_status != NIXL_SUCCESS) ? conn_status : status;
    }

    // Keeps the connection (which owns the endpoint) alive for the lifetime
    // of the request handle.
    ucx_connection_ptr_t conn_;
    // Resolved endpoint over which data and notifications are sent.
    const nixlUcxEp *ep_ = nullptr;
    std::vector<nixlUcxReq> requests_;
    nixlUcxWorker *worker_ = nullptr;
    nixl_status_t error_ = NIXL_SUCCESS;
};

#endif // NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_REQ_H
