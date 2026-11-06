/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_ENGINE_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_ENGINE_H

#include <memory>

#include "libfabric_backend.h"

namespace nixl {
class deviceOps;
class proxyRuntime;
struct proxyConfig;
} // namespace nixl

namespace nixl::libfabric {
/** The libfabric engine with the device proxy; proxy thread r % proxy_thread_count owns rail r. */
class proxyEngine final : public nixlLibfabricEngine {
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

private:
    class transport;

    template<typename DlistT>
    nixl_status_t
    prepProxyMemView(const DlistT &dlist, nixlMemViewH &mvh) const;

    nixl::deviceOps &ops_;
    std::unique_ptr<nixl::proxyRuntime> runtime_;
};
} // namespace nixl::libfabric

#endif
