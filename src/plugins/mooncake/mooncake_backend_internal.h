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
#ifndef NIXL_MOONCAKE_BACKEND_INTERNAL_H
#define NIXL_MOONCAKE_BACKEND_INTERNAL_H

#include "mooncake_backend.h"

#include <chrono>
#include <cstddef>
#include <cstdint>

// A batch cannot take more than this many requests before it has to be
// recycled; both engines check the capacity at submit time.
constexpr size_t kMaxRequestCount = 1024;

#ifdef HAVE_MOONCAKE_TENT
// TENT reports batch allocation failure as 0; the classic engine uses
// INVALID_BATCH (UINT64_MAX).
constexpr uint64_t kTentInvalidBatch = 0;
// How long a second releaseReqH() waits for a cancelled batch to reach a
// terminal state before reclaiming it regardless. The core does not come back
// a third time, so this is the last chance to free it.
constexpr std::chrono::milliseconds kReleaseDrainTimeout{100};
// The TENT C notification record is {char name[256]; char msg[4096]} and the
// receive path copies with strncpy, silently truncating anything longer. A
// truncated notification is worse than a rejected one.
constexpr size_t kMaxNotifNameLen = 255;
constexpr size_t kMaxNotifMsgLen = 4095;
#endif

// Shared by both translation units of the plugin: the request handle the
// classic and TENT paths both hand around.
struct nixlMooncakeBackendReqH : public nixlBackendReqH {
    explicit nixlMooncakeBackendReqH(uint64_t invalid_batch)
        : nixlBackendReqH(),
          batch_id(invalid_batch) {}

    virtual ~nixlMooncakeBackendReqH() {}

    uint64_t batch_id;
    size_t request_count = 0;
    // Set once releaseReqH() started best-effort cancellation (TENT mode); it
    // keeps a retried release from cancelling twice and keeps checkXfer() from
    // racing the batch reclamation.
    bool abort_requested = false;
    // Set once releaseReqH() has already refused one release. The core stores
    // that refusal in the request status, so the next release does not come
    // back through the retry path: it deletes the handle and discards what we
    // return. The second call therefore has to reclaim the batch.
    bool release_refused = false;
};

#endif // NIXL_MOONCAKE_BACKEND_INTERNAL_H
