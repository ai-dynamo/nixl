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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RUNTIME_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RUNTIME_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <vector>

#include "backend_aux.h"
#include "device/device_ops.h"
#include "proxy_protocol.h"
#include "proxy_config.h"
#include "proxy_transport.h"
#include "proxy_control_buffer.h"
#include "proxy_memview_manager.h"
#include "proxy_channel.h"
#include "proxy_worker.h"

namespace nixl {

class proxyRuntime {
public:
    ~proxyRuntime();

    proxyRuntime(proxyRuntime &&) = delete;
    proxyRuntime(const proxyRuntime &) = delete;
    proxyRuntime &
    operator=(proxyRuntime &&) = delete;
    proxyRuntime &
    operator=(const proxyRuntime &) = delete;

    /**
     * Build without starting workers. The runtime owns the transport; the allocator
     * must outlive the runtime. This overload uses getDeviceOps().
     * @retval NIXL_ERR_INVALID_PARAM No transport, or an invalid config.
     * @retval NIXL_ERR_NOT_SUPPORTED No device operations implementation is loaded.
     */
    [[nodiscard]] static nixl_status_t
    create(std::unique_ptr<proxyTransport> transport,
           const proxyConfig &config,
           std::unique_ptr<proxyRuntime> &out);

    [[nodiscard]] static nixl_status_t
    create(std::unique_ptr<proxyTransport> transport,
           const proxyConfig &config,
           std::unique_ptr<proxyRuntime> &out,
           deviceOps &allocator);

    [[nodiscard]] nixl_status_t
    prepMemView(const nixl_meta_dlist_t &dlist, proxy_view_handle_t *proxy_memview);

    /** Resolves the transport's direct pointers first; it may offer none. */
    [[nodiscard]] nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist, proxy_view_handle_t *proxy_memview);

    [[nodiscard]] nixl_status_t
    releaseMemView(proxy_view_handle_t proxy_memview);

    [[nodiscard]] nixl_status_t
    startWorkers();

    nixl_status_t
    shutdown();

    const nixlProxyRingDesc *
    deviceChannelViews() const {
        return device_channel_views_.empty() ? nullptr : device_channel_views_.data();
    }

    nixlProxyDeviceContextData *
    deviceContext() const {
        return static_cast<nixlProxyDeviceContextData *>(device_context_mem_.get());
    }

private:
    /** Ask owning workers to drain and reset, then wait for their acknowledgments. */
    void
    drainChannels() noexcept;

    proxyRuntime(std::unique_ptr<proxyTransport> transport,
                 const proxyConfig &config,
                 deviceOps &allocator) noexcept;

    /** Allocate rings, device context and workers; see create(). */
    nixl_status_t
    build();

    void
    joinWorkerThreads() noexcept;

    /**
     * create() yields a built runtime; startWorkers() and shutdown() are the only
     * transitions after that, and the only writers of the device shutdown word and
     * the workers' stop signal. `initialized` records that the transport's init()
     * succeeded, which is what obliges shutdown() to call its shutdown().
     */
    enum class state { created, initialized, built, running, stopped };

    /** First member, so it is destroyed after the workers, rings and memory views. */
    std::unique_ptr<proxyTransport> transport_;
    deviceOps &allocator_;
    proxyConfig config_;
    mutable std::mutex control_mutex_;
    std::vector<proxyChannel> channels_;
    std::unique_ptr<proxyControlBuffer> control_slots_;
    std::vector<nixlProxyRingDesc> device_channel_views_;
    deviceMem device_channel_views_mem_;
    deviceMem device_context_mem_;
    /** Shared by the workers; destroyed after them. */
    std::unique_ptr<const proxyWorkerContext> worker_context_;
    std::vector<std::unique_ptr<proxyWorker>> workers_;
    /** Created after the device context; destroyed after workers stop. */
    std::unique_ptr<proxyMemViewManager> memview_manager_;
    std::stop_source stop_source_;
    /** Bumped once per drain; each worker acks it when it has applied it. */
    alignas(64) std::atomic<uint64_t> drain_requested_{0};
    uint64_t *shutdown_word_dev_ = nullptr;
    state state_ = state::created;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RUNTIME_H
