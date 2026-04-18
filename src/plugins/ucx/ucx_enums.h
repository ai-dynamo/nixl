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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_ENUMS_H
#define NIXL_SRC_PLUGINS_UCX_UCX_ENUMS_H

#include <ostream>
#include <string_view>
#include <type_traits>

#include <ucs/type/status.h>

#include "nixl_types.h"

inline constexpr std::string_view nixl_ucx_invalid = "INVALID";

enum class nixl_ucx_mt_t { SINGLE, CTX, WORKER };

[[nodiscard]] constexpr std::string_view
toStringView(const nixl_ucx_mt_t t) noexcept {
    switch (t) {
    case nixl_ucx_mt_t::SINGLE:
        return "SINGLE";
    case nixl_ucx_mt_t::CTX:
        return "CTX";
    case nixl_ucx_mt_t::WORKER:
        return "WORKER";
    }
    return nixl_ucx_invalid;
}

enum class nixl_ucx_ep_state_t { NULL_, CONNECTED, FAILED, DISCONNECTED };

[[nodiscard]] constexpr std::string_view
toStringView(const nixl_ucx_ep_state_t t) noexcept {
    switch (t) {
    case nixl_ucx_ep_state_t::NULL_:
        return "NULL";
    case nixl_ucx_ep_state_t::CONNECTED:
        return "CONNECTED";
    case nixl_ucx_ep_state_t::FAILED:
        return "FAILED";
    case nixl_ucx_ep_state_t::DISCONNECTED:
        return "DISCONNECTED";
    }
    return nixl_ucx_invalid;
}

enum class nixl_ucx_cb_op_t { NOTIF_STR };

[[nodiscard]] constexpr std::string_view
toStringView(const nixl_ucx_cb_op_t t) noexcept {
    switch (t) {
    case nixl_ucx_cb_op_t::NOTIF_STR:
        return "NOTIF_STR";
    }
    return nixl_ucx_invalid;
}

template<typename Enum>
[[nodiscard]] constexpr auto
toInteger(const Enum e) noexcept {
    static_assert(std::is_enum_v<Enum>);
    return std::underlying_type_t<Enum>(e);
}

template<typename Enum>
inline void
toStream(std::ostream &os, const Enum t) {
    static_assert(std::is_enum_v<Enum>);

    const auto view = toStringView(t);

    if (view != nixl_ucx_invalid) {
        os << view;
    } else {
        os << toInteger(t);
    }
}

std::ostream &
operator<<(std::ostream &os, const nixl_ucx_mt_t t);

std::ostream &
operator<<(std::ostream &os, const nixl_ucx_ep_state_t t);

std::ostream &
operator<<(std::ostream &os, const nixl_ucx_cb_op_t t);

[[nodiscard]] constexpr nixl_status_t
toNixlStatus(const nixl_ucx_ep_state_t t) noexcept {
    switch (t) {
    case nixl_ucx_ep_state_t::CONNECTED:
        return NIXL_SUCCESS;
    case nixl_ucx_ep_state_t::FAILED:
        return NIXL_ERR_REMOTE_DISCONNECT;
    case nixl_ucx_ep_state_t::NULL_:
    case nixl_ucx_ep_state_t::DISCONNECTED:
        return NIXL_ERR_BACKEND;
    }
    return NIXL_ERR_BACKEND;
}

// Functions for weakly typed enums.

// Prints warning for unexpected values.
[[nodiscard]] nixl_status_t
ucxStatusToNixlStatus(const ucs_status_t t);

#endif
