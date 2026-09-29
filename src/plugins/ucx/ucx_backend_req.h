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
    // Notification to be sent after completion of all requests
    struct Notif {
        const std::string agent;
        std::unique_ptr<std::string> msg;

        Notif(const std::string &remote_agent, std::unique_ptr<std::string> msg)
            : agent(remote_agent),
              msg(std::move(msg)) {}
    };

    std::optional<Notif> notif;

#ifdef HAVE_UCX_SGL_API
    std::optional<nixl::ucx::sglXfer> sgl;
#endif

    explicit nixlUcxBackendReqH(nixlUcxWorker *worker) {
        setWorker(worker);
    }

    void
    reserve(size_t size) {
        requests_.reserve(size);
        NIXL_ASSERT(conn_ == nullptr);
    }

    [[nodiscard]] nixl_status_t
    append(nixl_status_t status, nixlUcxReq req, const ucx_connection_ptr_t &conn) {
        if (status == NIXL_IN_PROG) [[likely]] {
            requests_.push_back(req);
        } else if (status != NIXL_SUCCESS) {
            // Error. Release all previously initiated ops and exit:
            release();
            return status;
        }

        NIXL_ASSERT(conn_ == nullptr || conn_ == conn);
        conn_ = conn;
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
        requests_.clear();
        conn_.reset();
    }

    [[nodiscard]] virtual nixl_status_t
    status() {
        if (requests_.empty()) {
            /* No pending transmissions */
            conn_.reset();
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
        if (requests_.empty()) {
            conn_.reset();
        }
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
    [[nodiscard]] nixl_status_t
    checkConnection(const nixl_status_t status = NIXL_SUCCESS) const {
        NIXL_ASSERT(conn_ != nullptr);
        const nixl_status_t conn_status = conn_->getEp(getWorkerId())->checkTxState();
        return (conn_status != NIXL_SUCCESS) ? conn_status : status;
    }

    ucx_connection_ptr_t conn_;
    std::vector<nixlUcxReq> requests_;
    nixlUcxWorker *worker_ = nullptr;
};

#endif // NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_REQ_H
