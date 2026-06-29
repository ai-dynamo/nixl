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
#ifndef __GDS_MT_ENGINE_H
#define __GDS_MT_ENGINE_H

#include <atomic>
#include <future>
#include <memory>
#include <vector>

#include "gds_backend.h"
#include "taskflow/core/executor.hpp"
#include "taskflow/taskflow.hpp"

[[nodiscard]] size_t
defaultGdsMtThreadCount() noexcept;

class nixlGdsMtReqH : public nixlBackendReqH {
public:
    ~nixlGdsMtReqH() override;

    std::vector<GdsXferReq> request_list;
    tf::Taskflow taskflow;
    std::future<void> running_transfer;
    std::atomic<nixl_status_t> overall_status{NIXL_SUCCESS};
};

// "GDS_MT" backend: one blocking cuFileRead/cuFileWrite per request, run across
// a TaskFlow executor. Only this engine pulls in TaskFlow.
//
// Inherits from nixlGdsEngine (see gds_backend.h): registerMem/deregisterMem,
// queryMem, the cuFile driver lifecycle, and the prepXfer preamble (validation +
// descriptor->GdsXferReq, which then calls finalizePrep below). This class only
// implements the transfer mechanism: finalizePrep + postXfer/checkXfer/
// releaseReqH.
class nixlGdsMtEngine : public nixlGdsEngine {
public:
    explicit nixlGdsMtEngine(const nixlBackendInitParams *init_params);
    ~nixlGdsMtEngine() override = default;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;
    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

protected:
    nixl_status_t
    finalizePrep(std::vector<GdsXferReq> &&reqs, nixlBackendReqH *&handle) const override;

private:
    size_t thread_count_;
    std::unique_ptr<tf::Executor> executor_;
};

#endif // __GDS_MT_ENGINE_H
