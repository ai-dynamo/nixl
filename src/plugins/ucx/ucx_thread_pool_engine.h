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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_THREAD_POOL_ENGINE_H
#define NIXL_SRC_PLUGINS_UCX_UCX_THREAD_POOL_ENGINE_H

#include <memory>
#include <string>
#include <vector>

#include "ucx_thread_engine.h"

namespace asio {
class io_context;
}

class nixlUcxThreadPoolEngine : public nixlUcxThreadEngine {
public:
    nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params, size_t num_threads);
    ~nixlUcxThreadPoolEngine();

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

protected:
    nixl_status_t
    sendXferRange(const nixl_xfer_op_t &operation,
                  const nixl_meta_dlist_t &local,
                  const nixl_meta_dlist_t &remote,
                  const std::string &remote_agent,
                  nixlBackendReqH *handle,
                  size_t start_idx,
                  size_t end_idx) const override;

private:
    std::unique_ptr<asio::io_context> io_;
    std::vector<std::unique_ptr<nixlUcxThread>> dedicatedThreads_;
    size_t splitBatchSize_;
};

#endif // NIXL_SRC_PLUGINS_UCX_UCX_THREAD_POOL_ENGINE_H
