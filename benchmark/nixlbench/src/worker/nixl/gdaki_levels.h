/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GDAKI_LEVELS_H
#define GDAKI_LEVELS_H

#include <string_view>

namespace gdaki_levels {

inline constexpr std::string_view kThread{"thread"};
inline constexpr std::string_view kWarp{"warp"};
inline constexpr std::string_view kBlock{"block"};

inline bool
isValid(std::string_view level) {
    return level == kThread || level == kWarp || level == kBlock;
}

} // namespace gdaki_levels

#endif // GDAKI_LEVELS_H
