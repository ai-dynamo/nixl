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

#include <atomic>
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
    std::optional<nixl::trace::TraceContext> traceContext;
    std::uint64_t traceRequestId = 0;
    std::uint64_t traceExecution = 0;

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
        if (notif_.empty()) {
            notifStatus_.reset();
        }
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
        const bool has_notif = !notif_.empty();
        NIXL_ASSERT(has_notif == (notifStatus_ != nullptr));
        const bool need_cb = has_notif || (callback_ != nullptr);
        const nixl_status_t status = ep_->flushEp(req, need_cb ? flushCb : nullptr, this);
        if (need_cb && (status == NIXL_SUCCESS)) {
            flushCb(nullptr, UCS_OK, this);
        }

        return append(status, req);
    }

    virtual void
    setNotif(std::string &&notif) {
        NIXL_ASSERT(!hasPendingRequests());
        notif_ = std::move(notif);
        notifStatus_ =
            notif_.empty() ? nullptr : std::make_shared<nixl::ucx::amSendStatus>(NIXL_IN_PROG);
    }

    // The callback is not owned and must stay valid until the request is released
    virtual void
    setCallback(nixlBackendReqCallback *callback) {
        callback_ = callback;
    }

    [[nodiscard]] virtual bool
    isComposite() const noexcept {
        return false;
    }

    virtual void
    release() {
        size_t incomplete_reqs = 0;
        for (nixlUcxReq req : requests_) {
            incomplete_reqs += releaseRequest(req);
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
            if (notifStatus_ && (notifStatus_->load(std::memory_order_relaxed) == NIXL_IN_PROG)) {
                worker_->progressLoop();
            }
            return notifStatus();
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
        return (out_ret == NIXL_SUCCESS) ? notifStatus() : out_ret;
    }

    [[nodiscard]] bool
    hasPendingRequests() const noexcept {
        return !requests_.empty();
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

    nixl_status_t
    sendNotif() {
        NIXL_ASSERT(!notif_.empty());
        const nixl_status_t status = getEp().sendAm(nixl::ucx::am_cb_op_t::NOTIF_STR,
                                                    std::exchange(notif_, {}),
                                                    UCP_AM_SEND_FLAG_EAGER,
                                                    notifStatus_,
                                                    callback_);
        if (status != NIXL_IN_PROG) {
            notifStatus_->store(status);
        }
        return status;
    }

private:
    // A post to a single endpoint issues at most two requests:
    // one data request and one flush request.
    static constexpr size_t max_requests = 2;

    static void
    flushCb(void *, ucs_status_t status, void *user_data) {
        const auto handle = static_cast<nixlUcxBackendReqH *>(user_data);
        nixl_status_t result = nixl::ucx::ucsToNixlStatus(status);
        if ((status == UCS_OK) && !handle->notif_.empty()) {
            result = handle->sendNotif();
            if (result == NIXL_IN_PROG) {
                // The send callback calls the request callback
                return;
            }
        }
        if (handle->callback_ != nullptr) {
            handle->callback_->completed(result);
        }
    }

    [[nodiscard]] nixl_status_t
    notifStatus() {
        if (!notifStatus_) {
            return NIXL_SUCCESS;
        }

        const nixl_status_t status = notifStatus_->load(std::memory_order_acquire);
        if ((status == NIXL_SUCCESS) || (status == NIXL_IN_PROG)) {
            return status;
        }

        return checkConnection(status);
    }

    [[nodiscard]] bool
    releaseRequest(nixlUcxReq req) {
        const bool incomplete =
            nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req)) == NIXL_IN_PROG;
        if (incomplete) {
            // TODO: Need process this properly.
            // it may not be enough to cancel UCX request
            worker_->reqCancel(req);
        }
        worker_->reqRelease(req);
        return incomplete;
    }

    void
    reset() noexcept {
        requests_.clear();
        notifStatus_.reset();
        notif_.clear();
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
    std::string notif_;
    // Status of the notification send, shared with that send; null without a notification.
    std::shared_ptr<nixl::ucx::amSendStatus> notifStatus_;
    nixlBackendReqCallback *callback_ = nullptr;
};

#endif // NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_REQ_H
