/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "posix_queue.h"
#include <algorithm>

nixl_status_t
nixlPosixQueue::submit(const nixl_meta_dlist_t &, const nixl_meta_dlist_t &) {
    // If nothing left to submit, we're done
    if (num_ios_submitted_total >= num_ios_to_submit) {
        return NIXL_IN_PROG;
    }

    // Calculate how many more we can submit to reach target outstanding
    int remaining = num_ios_to_submit - num_ios_submitted_total;
    int slots_available = MAX_IO_OUTSTANDING - num_ios_outstanding;
    int to_submit = std::min(remaining, slots_available);

    // Nothing to submit if we're already at target outstanding
    if (to_submit <= 0) {
        return NIXL_IN_PROG;
    }

    // Call queue-specific batch submission
    int actual_submitted = 0;
    nixl_status_t status = submitBatch(num_ios_submitted_total, to_submit, actual_submitted);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    // Update tracking with actual number submitted
    num_ios_submitted_total += actual_submitted;
    num_ios_outstanding += actual_submitted;

    return NIXL_IN_PROG;
}
