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
#ifndef NIXL_SRC_UTILS_COMMON_UTIL_H
#define NIXL_SRC_UTILS_COMMON_UTIL_H

#include "nixl_types.h"

#define CONCAT(a, b) CONCAT_0(a, b)
#define CONCAT_0(a, b) a ## b
#define UNIQUE_NAME(name) CONCAT(name, __COUNTER__)

namespace nixl {

[[nodiscard]] constexpr bool
isReadWrite(const nixl_xfer_op_t operation) noexcept {
    switch (operation) {
    case NIXL_READ:
    case NIXL_WRITE:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool
localIsSource(const nixl_xfer_op_t operation) noexcept {
    switch (operation) {
    case NIXL_READ:
        return false;
    case NIXL_WRITE:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool
localIsTarget(const nixl_xfer_op_t operation) noexcept {
    switch (operation) {
    case NIXL_READ:
        return true;
    case NIXL_WRITE:
        return false;
    }
    return false;
}

} // namespace nixl

#endif
