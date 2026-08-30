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
// The TENT engine half of the Mooncake backend. Kept apart from
// mooncake_backend.cpp so the two engines' request paths stay readable; the
// classic path and the mode dispatch live over there.
#include "mooncake_backend.h"
#include "mooncake_backend_internal.h"

#include "common/nixl_log.h"

#ifdef HAVE_MOONCAKE_TENT
#include <tent/transfer_engine.h>
#endif

#include <chrono>
#include <cstdio>
#include <thread>

#ifdef HAVE_MOONCAKE_TENT
nixl_status_t
nixlMooncakeEngine::postXferTent(const nixl_xfer_op_t &operation,
                                 const nixl_meta_dlist_t &local,
                                 const nixl_meta_dlist_t &remote,
                                 uint64_t segment_id,
                                 nixlMooncakeBackendReqH *priv,
                                 const nixl_opt_b_args_t *opt_args) const {
    // TENT batches have the same fixed-capacity semantics as classic ones, so
    // the free-on-completion / reallocate-on-post recycling is kept.
    if (priv->batch_id == kTentInvalidBatch) {
        uint64_t batch_id = tent_allocate_batch(tent_engine_, kMaxRequestCount);
        if (batch_id == kTentInvalidBatch) {
            return NIXL_ERR_BACKEND;
        }
        priv->batch_id = batch_id;
        priv->request_count = 0;
    }
    size_t request_count = local.descCount();
    // Value-initialization zeroes every field, which the TENT C API requires:
    // transport_hint relies on UNSPEC == 0 (follow engine policy) and priority
    // 0 is the default.
    std::vector<tent_request_t> requests(request_count);
    for (size_t index = 0; index < request_count; ++index) {
        if (local[index].len != remote[index].len) {
            return NIXL_ERR_INVALID_PARAM;
        }
        requests[index].opcode = (operation == NIXL_READ) ? OPCODE_READ : OPCODE_WRITE;
        requests[index].source = (void *)local[index].addr;
        requests[index].target_offset = remote[index].addr;
        requests[index].length = local[index].len;
        requests[index].target_id = segment_id;
    }
    int rc = 0;
    if (opt_args && opt_args->hasNotif) {
        if (opt_args->notifMsg.size() > kMaxNotifMsgLen) {
            return NIXL_ERR_INVALID_PARAM;
        }
        rc = tent_submit_notif(tent_engine_,
                               priv->batch_id,
                               requests.data(),
                               request_count,
                               local_agent_name_.c_str(),
                               opt_args->notifMsg.c_str());
    } else {
        rc = tent_submit(tent_engine_, priv->batch_id, requests.data(), request_count);
    }
    if (rc) {
        return NIXL_ERR_BACKEND;
    }
    priv->request_count += request_count;
    return NIXL_IN_PROG;
}
#else
nixl_status_t
nixlMooncakeEngine::postXferTent(const nixl_xfer_op_t &,
                                 const nixl_meta_dlist_t &,
                                 const nixl_meta_dlist_t &,
                                 uint64_t,
                                 nixlMooncakeBackendReqH *,
                                 const nixl_opt_b_args_t *) const {
    return NIXL_ERR_NOT_SUPPORTED;
}
#endif

#ifdef HAVE_MOONCAKE_TENT
nixl_status_t
nixlMooncakeEngine::checkXferTent(nixlMooncakeBackendReqH *priv) const {
    tent_status_t status;
    // One aggregated poll instead of a per-task loop: COMPLETED only when every
    // task succeeded, the worst terminal state when all tasks are terminal,
    // PENDING otherwise. The call also drives engine-internal progress.
    if (tent_overall_status(tent_engine_, priv->batch_id, &status)) {
        return NIXL_ERR_BACKEND;
    }
    switch (status.status) {
    case STATUS_WAITING:
    case STATUS_PENDING:
        return NIXL_IN_PROG;
    case STATUS_COMPLETED:
        if (!priv->abort_requested) {
            // Recycle the batch so the same handle can be posted again;
            // releaseReqH() owns the batch once an abort was requested.
            tent_free_batch(tent_engine_, priv->batch_id);
            priv->batch_id = kTentInvalidBatch;
            priv->request_count = 0;
        }
        return NIXL_SUCCESS;
    case STATUS_CANCELED:
        return NIXL_ERR_CANCELED;
    default:
        return NIXL_ERR_BACKEND;
    }
}
#else
nixl_status_t
nixlMooncakeEngine::checkXferTent(nixlMooncakeBackendReqH *) const {
    return NIXL_ERR_NOT_SUPPORTED;
}
#endif

#ifdef HAVE_MOONCAKE_TENT
nixl_status_t
nixlMooncakeEngine::releaseReqHTent(nixlMooncakeBackendReqH *priv) const {
    // Idle handle: never posted, or the batch was already reclaimed on
    // completion.
    if (priv->batch_id == kTentInvalidBatch) {
        delete priv;
        return NIXL_SUCCESS;
    }
    tent_status_t status;
    if (tent_overall_status(tent_engine_, priv->batch_id, &status)) {
        // The engine no longer tracks this batch, so nothing can still be in
        // flight. Best-effort free and reclaim the handle.
        tent_free_batch(tent_engine_, priv->batch_id);
        delete priv;
        return NIXL_SUCCESS;
    }
    bool terminal = (status.status != STATUS_WAITING) && (status.status != STATUS_PENDING);
    if (!terminal && !priv->abort_requested) {
        priv->abort_requested = true;
        // Best-effort cancellation, as the BackendGuide requires: release must
        // handle cancelling in-flight requests without blocking. Tasks not yet
        // handed to a transport are cancelled immediately; work already posted
        // to a device may still complete, and transports without cancellation
        // support report an error - both are tolerated, the poll below decides.
        for (size_t task_id = 0; task_id < priv->request_count; ++task_id) {
            (void)tent_cancel_task(tent_engine_, priv->batch_id, task_id);
        }
        if (tent_overall_status(tent_engine_, priv->batch_id, &status) == 0) {
            terminal = (status.status != STATUS_WAITING) && (status.status != STATUS_PENDING);
        }
    }
    if (!terminal && !priv->release_refused) {
        // Transfers are still in flight after best-effort cancellation. Freeing
        // now would let the engine keep writing into memory the caller is about
        // to reuse, so refuse this release without blocking and let the caller
        // come back once the cancellation has landed.
        priv->release_refused = true;
        return NIXL_ERR_REPOST_ACTIVE;
    }
    if (!terminal) {
        // Second release of the same handle. nixlAgent::releaseXferReq() stored
        // the first refusal in the request status, so this call no longer sees
        // NIXL_IN_PROG: it skips the cancellation path and deletes the handle,
        // and ~nixlXferReqH() discards whatever releaseReqH() returns. Refusing
        // again would strand the batch in the engine for good. Wait a bounded
        // time for the cancellation to land, then reclaim it either way -
        // leaking the batch is the worse of the two outcomes.
        const auto deadline = std::chrono::steady_clock::now() + kReleaseDrainTimeout;
        while (!terminal && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (tent_overall_status(tent_engine_, priv->batch_id, &status)) {
                break;
            }
            terminal = (status.status != STATUS_WAITING) && (status.status != STATUS_PENDING);
        }
        if (!terminal) {
            NIXL_ERROR << "Mooncake TENT batch " << priv->batch_id
                       << " did not reach a terminal state within " << kReleaseDrainTimeout.count()
                       << " ms of cancellation; reclaiming it anyway. Transfers "
                          "may still be writing into the released buffers.";
        }
    }
    tent_free_batch(tent_engine_, priv->batch_id);
    priv->batch_id = kTentInvalidBatch;
    delete priv;
    return NIXL_SUCCESS;
}
#else
nixl_status_t
nixlMooncakeEngine::releaseReqHTent(nixlMooncakeBackendReqH *) const {
    return NIXL_ERR_NOT_SUPPORTED;
}
#endif
