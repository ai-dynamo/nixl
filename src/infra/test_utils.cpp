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
#include "test_utils.h"
#include "nixl_types.h"
#include "common/nixl_log.h"
#include <cstdlib>

void
nixl_exit_on_failure(nixl_status_t status, std::string_view message, std::string_view agent) {
    if (status == NIXL_SUCCESS) return;

    NIXL_ERROR << message << (agent.empty() ? "" : " for agent " + std::string{agent}) << ": "
               << nixlEnumStrings::statusStr(status) << " [" << status << "]";
    exit(EXIT_FAILURE);
}

void
nixl_exit_on_failure(bool condition, std::string_view message, std::string_view agent) {
    if (condition) return;

    NIXL_ERROR << message << (agent.empty() ? "" : " for agent " + std::string{agent});
    exit(EXIT_FAILURE);
}
