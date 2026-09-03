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

#ifndef NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_COMPLETION_STATUS_H
#define NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_COMPLETION_STATUS_H

#include "nixl_types.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace nixl::gpunetio {

/**
 * @brief Convert the shared GPU error latch to a NIXL completion status.
 *
 * @param error_flag Error latch written by the GPU progress kernel.
 * @return NIXL_SUCCESS when the latch is clear, otherwise NIXL_ERR_BACKEND.
 */
inline nixl_status_t
completionStatus(const volatile uint32_t *error_flag) {
    return *error_flag == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
}

/**
 * @brief Reserve the next notification slot in a power-of-two ring.
 *
 * @param send_pi Producer index advanced by this call.
 * @param elems_num Number of ring slots; must be a power of two.
 * @param send_addr Address of the first slot.
 * @param elems_size Size of each slot in bytes.
 * @param notification_slot Receives the reserved slot index.
 * @return Address of the reserved slot.
 */
inline uint8_t *
reserveNotificationSlot(std::atomic<uint32_t> &send_pi,
                        uint32_t elems_num,
                        uint8_t *send_addr,
                        size_t elems_size,
                        uint32_t &notification_slot) {
    notification_slot = send_pi.fetch_add(1) & (elems_num - 1);
    return send_addr + (notification_slot * elems_size);
}

} // namespace nixl::gpunetio

#endif /* NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_COMPLETION_STATUS_H */
