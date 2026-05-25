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

#include "request.h"
#include "ucx_enums.h"

#include "backend/backend_aux.h"
#include "common/nixl_log.h"

extern "C" {
#include <ucp/api/ucp.h>
}

#include <exception>
#include <mutex>
#include <string>
#include <tuple>

namespace nixl::ucx {

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

    recvRequestH *handle; // TODO: Check lifecycle management?
};

// AM arrives before RECV is POSTed
struct recvMapRndvValue {
    recvMapRndvValue(void *rndv_desc, const std::size_t size) noexcept
        : rndv_desc(rndv_desc),
          size(size) {}

    void *rndv_desc; // TODO: Check lifecycle management?
    const std::size_t size;
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

        const auto [iter, _] = map_.try_emplace({remote, tag}, std::in_place_type_t<std::deque<recvMapPostValue>>());

        if (auto *queue = std::get_if<std::deque<recvMapPostValue>>(&iter->second)) {
            queue->emplace_back(handle);
            NIXL_DEBUG << "UCX AM RECV POST " << remote << " tag " << tag << " handle " << handle << " queued " << queue->size();
            handle->iter = iter;
            return NIXL_IN_PROG;
        }

        if (auto *queue = std::get_if<std::deque<recvMapRndvValue>>(&iter->second)) {
            while (!handle->allCallsImpl() && !queue->empty()) {
                const recvMapRndvValue value = queue->front();
                queue->pop_front();

                if (!recvRndvData(worker, handle, value.rndv_desc, value.size)) {
                    if (queue->empty()) {
                        map_.erase(iter);
                    }

                    // TODO: If not empty, subsequent queue entries might be for this handle!!
                    return NIXL_ERR_BACKEND;
                }
            }

            if (!queue->empty()) {
                NIXL_ASSERT(handle->allCallsImpl());
                NIXL_DEBUG << "UCX AM RECV satisfied from queue with remaining " << queue->size();
                return NIXL_SUCCESS;
            }

            if (handle->allCallsImpl()) {
                map_.erase(iter);
                NIXL_DEBUG << "UCX AM RECV satisfied from queue without remaining";
                return NIXL_SUCCESS;
            }

            NIXL_DEBUG << "UCX AM RECV partially satisfied -- flipping queue";
            iter->second.emplace<std::deque<recvMapPostValue>>().emplace_back(handle);
            return NIXL_SUCCESS;
        }

        NIXL_ERROR_FUNC << "unreachable";
        std::terminate();
    }

    // EAGER AMs are handled like RNDV AMs in that ucp_am_recv_data_nbx() is called
    // so that we don't need to manually copy into different destination memory types.

    [[nodiscard]] ucs_status_t
    recvRndv(const ucp_worker_h worker,
             const std::string &remote,
             const std::string &tag,
             void *rndv_desc,
             const std::size_t size) {
        const std::lock_guard l1(mutex_);
        const auto [iter, _] = map_.try_emplace({remote, tag}, std::in_place_type_t<std::deque<recvMapRndvValue>>());

        if (auto *queue = std::get_if<std::deque<recvMapRndvValue>>(&iter->second)) {
            queue->emplace_back(rndv_desc, size);
            NIXL_DEBUG << "UCX AM RECV AM " << remote << " tag " << tag << " queued " << queue->size();
            return UCS_INPROGRESS;
        }

        if (auto *queue = std::get_if<std::deque<recvMapPostValue>>(&iter->second)) {
            NIXL_ASSERT(!queue->empty());
            recvRequestH *handle = queue->front().handle;
            NIXL_DEBUG << "UCX AM RECV AM " << remote << " tag " << tag << " handle " << handle << " first of " << queue->size();
            const std::lock_guard l2(handle->mutex);

            const auto result = recvRndvData(worker, handle, rndv_desc, size) ? UCS_OK : UCS_ERR_REJECTED;

            // TODO: In the error case, future received active message might be for this handle!!

            if (handle->allCallsImpl()) {
                handle->iter.reset();
                queue->pop_front();

                if (queue->empty()) {
                    map_.erase(iter);
                }
            }
            return result;
        }

        NIXL_ERROR_FUNC << "unreachable";
        std::terminate();
    }

    [[nodiscard]] bool
    recvRndvData(const ucp_worker_h worker,
                 recvRequestH *handle,
                 void *rndv_desc,
                 const std::size_t size) {
        NIXL_ASSERT(handle != nullptr);

        if (handle->local[handle->recvDataCalls].len != size) {
            // For now it was agreed that we only need to handle matching
            // sizes. If we were to accept less incoming data we would need
            // a way to tell the application how much was actually received?
            NIXL_ERROR_FUNC << "UCX AM RECV DATA expected " << handle->local[handle->recvDataCalls].len << " incoming " << size;
            ucp_am_data_release(worker, rndv_desc); // TODO: Correct? Needed elsewhere?
            return false;
        }

        ucp_request_param_t params{0};

        NIXL_DEBUG << "UCX AM RECV AM DATA handle " << handle << " desc " << rndv_desc;

        const auto req = ucp_am_recv_data_nbx(worker,
                                              rndv_desc,
                                              reinterpret_cast<void *>(handle->local[handle->recvDataCalls].addr),
                                              handle->local[handle->recvDataCalls].len,
                                              &params);

        ++handle->recvDataCalls;

        if (req == nullptr) {
            NIXL_DEBUG << "UCX AM RECV DATA immediate";
        } else if (UCS_PTR_IS_ERR(req)) {
            NIXL_ERROR_FUNC << "UCX AM RECV DATA failure " << UCS_PTR_STATUS(req);
            ++handle->recvDataErrors;
            return false;
        } else if (UCS_PTR_IS_PTR(req)) {
            NIXL_DEBUG << "UCX AM RECV DATA success";
            handle->append(req);
        } else {
            NIXL_ERROR_FUNC << "unreachable";
            std::terminate();
        }

        return true;
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
