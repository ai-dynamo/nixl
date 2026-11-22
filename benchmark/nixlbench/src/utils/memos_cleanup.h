/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace nixlbench {

enum class MemosReleaseStatus { released, pending, error };

// A deferred abort retains the request and its buffer references. Progress
// it and retry release; the caller must not discard the handle on failure.
// There is intentionally no unsafe timeout that frees live DMA buffers.
template<typename Release, typename Progress>
bool
releaseMemosRequest(Release release, Progress progress) {
    for (;;) {
        switch (release()) {
        case MemosReleaseStatus::released:
            return true;
        case MemosReleaseStatus::error:
            return false;
        case MemosReleaseStatus::pending:
            progress();
            break;
        }
    }
}

} // namespace nixlbench
