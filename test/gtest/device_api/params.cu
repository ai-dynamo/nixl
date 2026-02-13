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

#include "params.h"

#include <string_view>

namespace {
namespace da = nixl::device_api;

const std::vector<nixl::device_api::send_mode_t> modes = {
    da::send_mode_t::NODELAY_WITH_REQ,
    da::send_mode_t::NODELAY_WITHOUT_REQ,
    da::send_mode_t::DELAY_WITHOUT_REQ,
    da::send_mode_t::MULTI_CHANNEL,
};

const std::vector<nixl_gpu_level_t> levels_wo_block = {
    nixl_gpu_level_t::WARP,
    nixl_gpu_level_t::THREAD,
};

const std::vector<nixl_gpu_level_t> levels = {
    nixl_gpu_level_t::BLOCK,
    nixl_gpu_level_t::WARP,
    nixl_gpu_level_t::THREAD,
};

[[nodiscard]] std::string_view
memTypetoString(nixl_mem_t mem_type) {
    switch (mem_type) {
    case VRAM_SEG:
        return "dst_vram";
    case DRAM_SEG:
        return "dst_dram";
    default:
        return "unknown";
    }
}

[[nodiscard]] std::string_view
levelToString(nixl_gpu_level_t level) {
    switch (level) {
    case nixl_gpu_level_t::WARP:
        return "warp";
    case nixl_gpu_level_t::BLOCK:
        return "block";
    case nixl_gpu_level_t::THREAD:
        return "thread";
    default:
        return "unknown";
    }
}

[[nodiscard]] std::string_view
sendModeToString(da::send_mode_t mode) {
    switch (mode) {
    case da::send_mode_t::NODELAY_WITH_REQ:
        return "nodelay_with_req";
    case da::send_mode_t::NODELAY_WITHOUT_REQ:
        return "nodelay_without_req";
    case da::send_mode_t::DELAY_WITHOUT_REQ:
        return "delay_without_req";
    case da::send_mode_t::MULTI_CHANNEL:
        return "multi_channel";
    default:
        return "unknown";
    }
}

[[nodiscard]] std::vector<da::params>
getParams(const std::vector<nixl_gpu_level_t> &levels) {
    std::vector<da::params> params;
    for (const auto &level : levels) {
        for (const auto &mode : modes) {
            for (const auto &mem_type : {VRAM_SEG, DRAM_SEG}) {
                params.emplace_back(level, mode, mem_type);
            }
        }
    }
    return params;
}
} // namespace

namespace nixl::device_api {
std::string
params::toString() const {
    return std::string(memTypetoString(memType)) + "_" + std::string(levelToString(level)) + "_" +
        std::string(sendModeToString(mode));
}

std::string
paramsLevelToString(const testing::TestParamInfo<nixl_gpu_level_t> &info) {
    return std::string(levelToString(info.param));
}

std::vector<params>
paramsWithoutBlockLevel() {
    return getParams(levels_wo_block);
}

std::vector<params>
paramsWithBlockLevel() {
    return getParams(levels);
}
} // namespace nixl::device_api
