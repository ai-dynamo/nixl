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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"

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

    nixlBackendPostTrace trace;
    nixlTime::us_t inflightSince = 0;
    nixl_xfer_op_t operation = NIXL_WRITE;
    bool wireCompletedOnFlush = false;
    bool wireCompletedAtOnce = false;

#ifdef HAVE_UCX_SGL_API
    std::optional<nixl::ucx::sglXfer> sgl;
#endif

    explicit nixlUcxBackendReqH(nixlUcxWorker *worker) {
        setWorker(worker);
    }

    void
    init(const ucx_connection_ptr_t &conn, const nixlUcxEp &ep) {
        NIXL_ASSERT(requests_.empty());
        conn_ = conn;
        ep_ = &ep;
    }

    [[nodiscard]] const nixlUcxEp &
    getEp() const {
        NIXL_ASSERT(ep_ != nullptr);
        return *ep_;
    }

    [[nodiscard]] nixl_status_t
    append(nixl_status_t status, nixlUcxReq req) {
        if (status == NIXL_IN_PROG) [[likely]] {
            requests_.push_back(req);
        } else if (status != NIXL_SUCCESS) {
            // Error. Release all previously initiated ops and exit:
            release();
            return status;
        }

        return NIXL_SUCCESS;
    }

    /**
     * @brief Flush the bound endpoint and track the flush request
     */
    [[nodiscard]] nixl_status_t
    flush() {
        nixlUcxReq req = nullptr;
        const auto callback = wireCompletedOnFlush ? &flushCallback : nullptr;
        const nixl_status_t status = ep_->flushEp(req, callback, this);
        wireCompletedAtOnce = (status == NIXL_SUCCESS) && (callback != nullptr);
        return append(status, req);
    }

    void
    recordWireCompleted(nixlTime::us_t phase_at) const noexcept {
        const nixlTime::us_t completed_at = (phase_at != 0) ? phase_at : nixlTime::getUs();
        trace.recordAt(nixl_trace_phase_t::WIRE_COMPLETED, phase_at);
        trace.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT, operation, inflightSince, completed_at);
    }

    [[nodiscard]] nixl_status_t
    sendNotif() {
        nixlUcxReq req = nullptr;
        const nixl_status_t status = getEp().sendAm(
            nixl::ucx::am_cb_op_t::NOTIF_STR, std::move(notif), UCP_AM_SEND_FLAG_EAGER, &req);
        notif.clear();
        return append(status, req);
    }

    [[nodiscard]] virtual bool
    isComposite() const noexcept {
        return false;
    }

    virtual void
    release() {
        size_t incomplete_reqs = 0;
        for (nixlUcxReq req : requests_) {
            const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (ret == NIXL_IN_PROG) {
                // TODO: Need process this properly.
                // it may not be enough to cancel UCX request
                worker_->reqCancel(req);
                ++incomplete_reqs;
            }
            worker_->reqRelease(req);
        }
        if (incomplete_reqs != 0) {
            NIXL_WARN << "request handle " << this << " released with " << incomplete_reqs
                      << " requests in progress";
        }
        reset();
    }

    [[nodiscard]] virtual nixl_status_t
    status() {
        if (requests_.empty()) {
            /* No pending transmissions */
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
        return out_ret;
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
    // A post to a single endpoint issues at most three requests:
    // one data request, one flush request, and one notification request.
    static constexpr size_t max_requests = 3;

    static void
    flushCallback(void *request, ucs_status_t status, void *user_data) noexcept;

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
    absl::InlinedVector<nixlUcxReq, max_requests> requests_;
    nixlUcxWorker *worker_ = nullptr;
};

#endif // NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_REQ_H
