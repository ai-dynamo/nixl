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

#ifndef NIXL_SRC_PLUGINS_UCX_UCX_PROXY_ENGINE_H
#define NIXL_SRC_PLUGINS_UCX_UCX_PROXY_ENGINE_H

#include <memory>
#include <mutex>
#include <string>

#include "ucx_backend.h"

namespace nixl {
class deviceOps;
class proxyRuntime;
struct proxyConfig;
} // namespace nixl

namespace nixl::ucx {
/**
 * The UCX engine with the device proxy (device_proxy=true): memory views run through the proxy
 * runtime, and each (channel, peer) ring runs on one of the shared UCX workers, which only its
 * proxy thread progresses.
 */
class proxyEngine final : public nixlUcxEngine {
public:
    proxyEngine(const nixlBackendInitParams &init_params, const nixl::proxyConfig &config);

    ~proxyEngine() override;

    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist,
                nixlMemViewH &mvh,
                const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    prepMemView(const nixl_meta_dlist_t &dlist,
                nixlMemViewH &mvh,
                const nixl_opt_b_args_t *opt_args = nullptr) const override;

    void
    releaseMemView(nixlMemViewH mem_view) const override;

    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;

protected:
    void
    appendNotif(std::string &&remote_name, std::string &&msg) override;

private:
    class transport;

    template<typename DlistT>
    nixl_status_t
    prepProxyMemView(const DlistT &dlist, nixlMemViewH &mvh) const;

    nixl::deviceOps &ops_;
    std::mutex notifMutex_;
    std::unique_ptr<nixl::proxyRuntime> runtime_;
};
} // namespace nixl::ucx

#endif
