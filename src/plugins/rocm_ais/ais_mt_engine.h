/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef NIXL_SRC_PLUGINS_ROCM_AIS_AIS_MT_ENGINE_H
#define NIXL_SRC_PLUGINS_ROCM_AIS_AIS_MT_ENGINE_H

#include <atomic>
#include <cstddef>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include <taskflow/taskflow.hpp>

#include "ais_backend.h"

size_t
defaultAisMtThreadCount() noexcept;

class nixlAisMtReqH : public nixlBackendReqH {
public:
    ~nixlAisMtReqH() override;

    std::vector<aisXferReq> request_list;
    tf::Taskflow taskflow;
    std::future<void> running_transfer;
    std::atomic<nixl_status_t> overall_status{NIXL_SUCCESS};
};

// "AIS_MT" backend: one blocking hipFileRead/hipFileWrite per request, run
// across a TaskFlow executor. Only this engine pulls in TaskFlow.
//
// Inherits from nixlAisEngine (see ais_backend.h): registerMem/deregisterMem,
// queryMem, the hipFile driver lifecycle, and the prepXfer preamble (validation
// + descriptor->aisXferReq, which then calls finalizePrep below). This class
// only implements the transfer mechanism: finalizePrep + postXfer/checkXfer/
// releaseReqH.
class nixlAisMtEngine : public nixlAisEngine {
public:
    explicit nixlAisMtEngine(const nixlBackendInitParams *init_params);
    ~nixlAisMtEngine() override = default;

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
    finalizePrep(std::vector<aisXferReq> &&reqs, nixlBackendReqH *&handle) const override;

private:
    size_t thread_count_;
    std::unique_ptr<tf::Executor> executor_;
};

#endif // NIXL_SRC_PLUGINS_ROCM_AIS_AIS_MT_ENGINE_H
