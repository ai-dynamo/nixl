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
#ifndef NIXL_TEST_UTILS_H
#define NIXL_TEST_UTILS_H

#include <string_view>
#include "nixl_types.h"

/**
 * @brief Exit on failure utility functions for tests and examples
 *
 * These functions provide a convenient way to check conditions and exit
 * with appropriate error messages if they fail. They are designed for
 * use in tests and examples where immediate termination on error is desired.
 */

/**
 * @brief Exit if nixl_status_t indicates failure
 * @param status The nixl status to check
 * @param message Error message to display
 * @param agent Optional agent name for context
 */
void
nixl_exit_on_failure(nixl_status_t status, std::string_view message, std::string_view agent = {});

/**
 * @brief Exit if boolean condition is false
 * @param condition The condition to check (exits if false)
 * @param message Error message to display
 * @param agent Optional agent name for context
 */
void
nixl_exit_on_failure(bool condition, std::string_view message, std::string_view agent = {});

#endif /* NIXL_TEST_UTILS_H */
