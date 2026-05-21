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
#ifndef NIXL_SRC_PLUGINS_UCX_RECV_MAP_H
#define NIXL_SRC_PLUGINS_UCX_RECV_MAP_H

#include "ucx_enums.h"

#include "backend/backend_aux.h"
#include "common/nixl_log.h"

extern "C" {
#include <ucp/api/ucp.h>
}

#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

namespace nixl::ucx {

struct recvMapKey;

struct recvMapPostValue;
struct recvMapRndvValue;

using recv_map_value_t = std::variant<std::deque<recvMapPostValue>, std::deque<recvMapRndvValue>>;

using recv_map_t = std::map<recvMapKey, recv_map_value_t, std::less<void>>;

enum class am_recv_mode_t : bool {
    RNDV,
    EAGER
};

enum class am_data_mode_t {
    POST_RNDV,
    POST_EAGER,
    RNDV_POST,
    EAGER_POST
};

// TODO: Check when and where ucp_am_data_release() needs to be called!!
// (When the request is deleted after the AM CB returned UCS_INPROGRESS
// but before ucp_am_recv_data_nbx() was called...)

struct recvRequestH : public nixlBackendReqH {
    const nixl_meta_dlist_t &local; // initiatorDescs from the NIXL request

    std::recursive_mutex mutex;

    nixl_status_t status = NIXL_ERR_NOT_POSTED;

    void *rndv_desc = nullptr; // TODO: Lifecycle ok?
    void *data_req = nullptr; // TODO: Lifecycle is not ok yet!

    am_data_mode_t mode;

    std::optional<recv_map_t::iterator> iter;

    explicit
    recvRequestH(const nixl_meta_dlist_t &local)
        : local(local) {}
};

struct recvMapKey {
    std::string remoteAgent;
    std::string sendRecvTag;

    [[nodiscard]] auto
    tie() const noexcept {
        return std::tie(remoteAgent, sendRecvTag);
    }
};

[[nodiscard]] bool inline
operator<(const recvMapKey &l, const recvMapKey &r) noexcept {
    return l.tie() < r.tie();
}

[[nodiscard]] bool inline
operator==(const recvMapKey &l, const recvMapKey &r) noexcept {
    return l.tie() == r.tie();
}

// RECV is POSTed before AM arrives
struct recvMapPostValue {
    explicit
    recvMapPostValue(recvRequestH *handle) noexcept
        : handle(handle) {}

    recvRequestH *handle; // TODO: Lifecycle management?
};

// AM arrives before RECV is POSTed
struct recvMapRndvValue {
    recvMapRndvValue(void *rndv_desc, const std::size_t size, const am_recv_mode_t mode) noexcept
        : rndv_desc(rndv_desc),
          size(size),
          mode(mode) {}

    void *rndv_desc; // TODO: Lifecycle management?
    const std::size_t size;
    const am_recv_mode_t mode;
};

class recvMap {
public:
    recvMap() = default;

    ~recvMap() {
        if (!map_.empty()) {
            NIXL_WARN << "UCX AM RECV map is not empty";
        }
    }

    recvMap(recvMap&&) = delete;
    recvMap(const recvMap&) = delete;

    void operator=(recvMap&&) = delete;
    void operator=(const recvMap&) = delete;

    [[nodiscard]] nixl_status_t
    postRecv(const ucp_worker_h worker,
             const std::string &remote,
             const std::string &tag,
             recvRequestH *handle) {

        const std::lock_guard l1(mutex_);
        const std::lock_guard l2(handle->mutex);

        NIXL_ASSERT(handle);
        NIXL_ASSERT(!handle->iter);
        NIXL_ASSERT(handle->local.descCount() == 1); // TODO: Generalize

        const auto [iter, _] = map_.try_emplace({remote, tag}, std::in_place_type_t<std::deque<recvMapPostValue>>());

        if (auto *queue = std::get_if<std::deque<recvMapPostValue>>(&iter->second)) {
            queue->emplace_back(handle);
            NIXL_DEBUG << "UCX AM RECV POST " << remote << " tag " << tag << " handle " << handle << " queued " << queue->size();
            handle->iter = iter;
            handle->status = NIXL_IN_PROG;
            return handle->status;
        }

        if (auto *queue = std::get_if<std::deque<recvMapRndvValue>>(&iter->second)) {
            NIXL_ASSERT(!queue->empty());
            const recvMapRndvValue value = queue->front();

            queue->pop_front();
            if (queue->empty()) {
                map_.erase(iter);
            }

            const auto m = (value.mode == am_recv_mode_t::RNDV) ? am_data_mode_t::RNDV_POST : am_data_mode_t::EAGER_POST;
            const auto status = recvRndvData(worker, handle, value.rndv_desc, value.size, m);
            return ucsToNixlStatus(status);
        }

        NIXL_ERROR_FUNC << "unreachable";
        std::terminate();
    }

    // EAER AMs are handled like RNDV AMs in that ucp_am_recv_data_nbx() is called
    // so that we don't need to manually copy into different destination memory types.

    [[nodiscard]] ucs_status_t
    recvRndv(const ucp_worker_h worker,
             const std::string &remote,
             const std::string &tag,
             void *rndv_desc,
             const std::size_t size,
             const am_recv_mode_t mode) {
        const std::lock_guard l1(mutex_);
        const auto [iter, _] = map_.try_emplace({remote, tag}, std::in_place_type_t<std::deque<recvMapRndvValue>>());

        if (auto *queue = std::get_if<std::deque<recvMapRndvValue>>(&iter->second)) {
            queue->emplace_back(rndv_desc, size, mode);
            NIXL_DEBUG << "UCX AM RECV AM " << remote << " tag " << tag << " queued " << queue->size();
            return UCS_INPROGRESS;
        }

        if (auto *queue = std::get_if<std::deque<recvMapPostValue>>(&iter->second)) {
            NIXL_ASSERT(!queue->empty());
            recvRequestH *handle = queue->front().handle;
            NIXL_DEBUG << "UCX AM RECV AM " << remote << " tag " << tag << " handle " << handle << " first of " << queue->size();
            const std::lock_guard l2(handle->mutex);

            handle->iter.reset();
            queue->pop_front();
            if (queue->empty()) {
                map_.erase(iter);
            }

            const auto m = (mode == am_recv_mode_t::RNDV) ? am_data_mode_t::POST_RNDV : am_data_mode_t::POST_EAGER;
            const auto status = recvRndvData(worker, handle, rndv_desc, size, m);
            return status;
        }

        NIXL_ERROR_FUNC << "unreachable";
        std::terminate();
    }

    static void
    recvRndvDataCallback(void *request, ucs_status_t status, size_t length, void *user_data) {
        const auto handle = static_cast<recvRequestH *>(user_data);
        NIXL_DEBUG << __FUNCTION__ << " ENTER";
        {
            const std::lock_guard l2(handle->mutex);

            if (status == UCS_OK) {
                // TODO: What with length?
                NIXL_DEBUG << "UCX AM RECV DATA callback success " << length;
                handle->status = NIXL_SUCCESS;
            } else {
                NIXL_ERROR_FUNC << "UCX AM RECV DATA callback failure " << status;
                handle->status = ucsToNixlStatus(status);
            }
        }

        if (request) {
            ucp_request_free(request);
        }
        NIXL_DEBUG << __FUNCTION__ << " LEAVE";
    }

    [[nodiscard]] ucs_status_t
    recvRndvData(const ucp_worker_h worker,
                 recvRequestH *handle,
                 void *rndv_desc,
                 const std::size_t size,
                 const am_data_mode_t mode) {
        NIXL_ASSERT(handle != nullptr);

        if (handle->local[0].len != size) {
            // For now it was agreed that we only need to handle matching
            // sizes. If we were to accept less incoming data we would need
            // a way to tell the application how much was actually received?
            NIXL_ERROR_FUNC << "UCX AM RECV DATA expected " << handle->local[0].len << " incoming " << size;
            ucp_am_data_release(worker, rndv_desc); // TODO: For which modes is this needed?
            return UCS_ERR_REJECTED;
        }

        ucp_request_param_t params{0};
        params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
            | UCP_OP_ATTR_FIELD_USER_DATA;
        // UCP_OP_ATTR_FLAG_NO_IMM_CMPL
        // UCP_OP_ATTR_FIELD_USER_DATA for IOV
        params.user_data = handle;
        params.cb.recv_am = &recvRndvDataCallback;

        NIXL_DEBUG << "UCX AM RECV AM DATA handle " << handle << " desc " << rndv_desc;

        handle->mode = mode;

        const auto req = ucp_am_recv_data_nbx(worker,
                                              rndv_desc,
                                              reinterpret_cast<void *>(handle->local[0].addr),
                                              handle->local[0].len,
                                              &params);

        NIXL_DEBUG << "CHECKPOINT";

        if (req == nullptr) {
            NIXL_DEBUG << "UCX AM RECV DATA immediate";
            // TODO: Double check that the callback was called in this case?
            handle->status = NIXL_SUCCESS;
        } else if (UCS_PTR_IS_ERR(req)) {
            NIXL_ERROR_FUNC << "UCX AM RECV DATA failure " << UCS_PTR_STATUS(req);
            handle->status = ucsToNixlStatus(UCS_PTR_STATUS(req));
        } else if (UCS_PTR_IS_PTR(req)) {
            NIXL_DEBUG << "UCX AM RECV DATA success";
            handle->status = NIXL_IN_PROG;
            handle->data_req = req;
        } else {
            NIXL_ERROR_FUNC << "unreachable";
            std::terminate();
        }

        return UCS_OK;
    }

    void
    erase(recvRequestH *handle) {
        const std::lock_guard l1(mutex_);
        const std::lock_guard l2(handle->mutex); // Not strictly necessary?

        if (!handle->iter.has_value()) {
            return;
        }

        const auto &iter = *handle->iter;

        NIXL_ASSERT(std::holds_alternative<std::deque<recvMapPostValue>>(iter->second));

        auto &queue = std::get<std::deque<recvMapPostValue>>(iter->second);

        // A handle can occur only once in queue.
        // In the normal case the erase() should hit the first element? It's the oldest.

        for (auto i = queue.begin(); i != queue.end(); ++i) {
            if (i->handle == handle) {
                queue.erase(i);
                break;
            }
        }

        if (queue.empty()) {
            map_.erase(iter);
        }

        handle->iter.reset();
    }

private:
    std::mutex mutex_; // Not needed without progress thread?!
    recv_map_t map_;
};

} // namespace nixl::ucx

#endif
