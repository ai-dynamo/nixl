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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_SPLIT_FAILURE_H
#define NIXL_SRC_PLUGINS_UCX_UCX_SPLIT_FAILURE_H

#include <atomic>
#include <cstdint>

extern "C" {
#include <ucp/api/ucp.h>
}

#include "nixl_types.h"

class nixlUcxSplitFailure {
public:
    enum class decision_t : std::uint8_t { WAIT, RECORD, DROP };

    struct failure_t {
        nixl_status_t status = NIXL_SUCCESS;
        std::int32_t native = UCS_OK;
    };

    void
    reset() noexcept {
        failure_.store({}, std::memory_order_relaxed);
        submission_.store(submission_t::SUBMITTING, std::memory_order_relaxed);
    }

    [[nodiscard]] bool
    publish(nixl_status_t status, ucs_status_t native) noexcept {
        failure_t none;
        return failure_.compare_exchange_strong(none, {status, native});
    }

    [[nodiscard]] nixl_status_t
    status() const noexcept {
        return failure_.load().status;
    }

    [[nodiscard]] ucs_status_t
    native() const noexcept {
        return static_cast<ucs_status_t>(failure_.load(std::memory_order_relaxed).native);
    }

    void
    finishSubmission(nixl_status_t status) noexcept {
        submission_.store((status == NIXL_SUCCESS) ? submission_t::SUBMITTED :
                                                     submission_t::REJECTED,
                          std::memory_order_release);
    }

    [[nodiscard]] decision_t
    decide() const noexcept {
        switch (submission_.load(std::memory_order_acquire)) {
        case submission_t::SUBMITTING:
            return decision_t::WAIT;
        case submission_t::SUBMITTED:
            return decision_t::RECORD;
        case submission_t::REJECTED:
            return decision_t::DROP;
        }
        return decision_t::DROP;
    }

    [[nodiscard]] nixl_status_t
    submissionStatus(nixl_status_t fanout) const noexcept {
        if (fanout == NIXL_SUCCESS) {
            return fanout;
        }
        const nixl_status_t claimed = status();
        return (claimed != NIXL_SUCCESS) ? claimed : fanout;
    }

private:
    enum class submission_t : std::uint8_t { SUBMITTING, SUBMITTED, REJECTED };

    static_assert(std::atomic<failure_t>::is_always_lock_free);

    std::atomic<failure_t> failure_{};
    std::atomic<submission_t> submission_{submission_t::SUBMITTING};
};

#endif
