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
#include <functional>
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
        NIXL_ASSERT(notifReq_ == nullptr);
        conn_ = conn;
        ep_ = &ep;
        notif_.clear();
        notifState_ = notif_state_t::NONE;
        notifStatus_ = NIXL_SUCCESS;
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
        const nixl_status_t status = ep_->flushEp(req, notifBuild_ ? flushCb : nullptr, this);
        const nixl_status_t ret = append(status, req);
        if (ret == NIXL_SUCCESS) {
            buildNotif();
        }

        return ret;
    }

    void
    setNotif(std::function<std::string(const nixlUcxBackendReqH &)> &&build) {
        notifBuild_ = std::move(build);
    }

    void
    buildNotif() {
        if (notifBuild_) {
            notif_ = notifBuild_(*this);
            notifBuild_ = nullptr;
            notifStatus_ = NIXL_IN_PROG;
            if (notifState_.exchange(notif_state_t::BUILT) == notif_state_t::FLUSHED) {
                sendNotif();
            }
        }
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
        const nixlUcxReq notif_req = notifReq_.exchange(nullptr);
        if (notif_req != nullptr) {
            incomplete_reqs += releaseRequest(notif_req);
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
            if (notifStatus_.load(std::memory_order_acquire) == NIXL_IN_PROG) {
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

private:
    enum class notif_state_t : uint8_t { NONE, BUILT, FLUSHED, SENT };

    // A post to a single endpoint issues at most two requests:
    // one data request and one flush request.
    static constexpr size_t max_requests = 2;

    static void
    flushCb(void *, ucs_status_t status, void *user_data) {
        if (status == UCS_OK) {
            static_cast<nixlUcxBackendReqH *>(user_data)->flushed();
        }
    }

    void
    flushed() {
        if (notifState_.exchange(notif_state_t::FLUSHED) == notif_state_t::BUILT) {
            sendNotif();
        }
    }

    void
    sendNotif() {
        nixlUcxReq req = nullptr;
        const nixl_status_t status = getEp().sendAm(
            nixl::ucx::am_cb_op_t::NOTIF_STR, std::move(notif_), UCP_AM_SEND_FLAG_EAGER, &req);
        notif_.clear();
        notifState_ = notif_state_t::SENT;
        if (status == NIXL_IN_PROG) {
            notifReq_ = req;
        } else {
            notifStatus_ = status;
        }
    }

    [[nodiscard]] nixl_status_t
    notifStatus() {
        flushed();

        const nixlUcxReq req = notifReq_.load(std::memory_order_acquire);
        if (req != nullptr) {
            const nixl_status_t status = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (status == NIXL_IN_PROG) {
                return status;
            }

            worker_->reqRelease(req);
            notifReq_ = nullptr;
            notifStatus_ = status;
        }

        const nixl_status_t status = notifStatus_.load(std::memory_order_acquire);
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
        notifReq_ = nullptr;
        notifBuild_ = nullptr;
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
    std::function<std::string(const nixlUcxBackendReqH &)> notifBuild_;
    std::string notif_;
    std::atomic<nixlUcxReq> notifReq_{nullptr};
    std::atomic<nixl_status_t> notifStatus_{NIXL_SUCCESS};
    std::atomic<notif_state_t> notifState_{notif_state_t::NONE};
};

#endif // NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_REQ_H
