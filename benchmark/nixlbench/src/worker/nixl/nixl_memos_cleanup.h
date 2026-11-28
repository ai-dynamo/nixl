/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nixl.h>
#include "utils/memos_cleanup.h"
#include <thread>

namespace nixlbench {

// Shared teardown policy for the worker and ownership tests. Keep handles on
// refusal; callers must retain their owners rather than free active DMA memory.
inline bool
releaseMemosHandles(nixlAgent &agent,
                    nixlXferReqH *&request,
                    nixlDlistH *&local,
                    nixlDlistH *&remote) {
    if (request) {
        if (!releaseMemosRequest(
                [&] {
                    const auto rc = agent.releaseXferReq(request);
                    if (rc == NIXL_SUCCESS) {
                        return MemosReleaseStatus::released;
                    }
                    return rc == NIXL_ERR_REPOST_ACTIVE ? MemosReleaseStatus::pending :
                                                          MemosReleaseStatus::error;
                },
                [&] {
                    agent.getXferStatus(request);
                    std::this_thread::yield();
                })) {
            return false;
        }
        request = nullptr;
    }
    for (auto *list : {&local, &remote}) {
        if (*list) {
            if (agent.releasedDlistH(*list) != NIXL_SUCCESS) {
                return false;
            }
            *list = nullptr;
        }
    }
    return true;
}

} // namespace nixlbench
