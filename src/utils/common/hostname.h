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
#ifndef NIXL_SRC_UTILS_COMMON_HOSTNAME_H
#define NIXL_SRC_UTILS_COMMON_HOSTNAME_H

#include <array>
#include <optional>
#include <string>

#include <limits.h>
#include <unistd.h>

namespace nixl {

// Returns the host name, or std::nullopt if it could not be read; callers pick
// their own fallback (e.g. .value_or("unknown")).
[[nodiscard]] inline std::optional<std::string>
getHostname() {
    std::array<char, HOST_NAME_MAX + 1> buf{};
    if (::gethostname(buf.data(), buf.size() - 1) == 0) {
        return std::string(buf.data());
    }
    return std::nullopt;
}

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_HOSTNAME_H
