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
#ifndef NIXL_SRC_UTILS_COMMON_OPERATORS_H
#define NIXL_SRC_UTILS_COMMON_OPERATORS_H

#include <ostream>

#include "nixl_types.h"

inline std::ostream &
operator<<(std::ostream &os, const nixl_mem_t value) {
    return os << nixlEnumStrings::memTypeStr(value);
}

inline std::ostream &
operator<<(std::ostream &os, const nixl_xfer_op_t value) {
    return os << nixlEnumStrings::xferOpStr(value);
}

inline std::ostream &
operator<<(std::ostream &os, const nixl_status_t value) {
    return os << nixlEnumStrings::statusStr(value);
}

#endif
