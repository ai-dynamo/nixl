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

#ifndef TEST_GTEST_DEVICE_API_PARAMS_H
#define TEST_GTEST_DEVICE_API_PARAMS_H

#include "send_mode.h"

#include "nixl_device.cuh"
#include "nixl_types.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace nixl::device_api {
struct params {
    params(nixl_gpu_level_t l, send_mode_t m, nixl_mem_t mt) : level(l), mode(m), memType(mt) {}

    [[nodiscard]] std::string
    toString() const;

    nixl_gpu_level_t level;
    send_mode_t mode;
    nixl_mem_t memType;
};

[[nodiscard]] inline std::string
paramsInfoToString(const testing::TestParamInfo<params> &info) {
    return info.param.toString();
}

[[nodiscard]] std::string
paramsLevelToString(const testing::TestParamInfo<nixl_gpu_level_t> &);

[[nodiscard]] std::vector<params>
allParams();

[[nodiscard]] std::vector<params>
paramsWithoutBlockLevel();

[[nodiscard]] std::vector<params>
paramsWithoutMultiChannel();
} // namespace nixl::device_api
#endif // TEST_GTEST_DEVICE_API_PARAMS_H
