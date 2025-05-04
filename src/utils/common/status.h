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

#ifndef STATUS_H
#define STATUS_H

#include <iostream>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include "nixl_log.h"

#define NIXL_LOG_AND_RETURN_IF_ERROR(status, message) \
    do { \
        if ((status) != NIXL_SUCCESS && (status) != NIXL_IN_PROG) { \
            NIXL_ERROR << absl::StrFormat("Error: %d - %s", (status), (message)); \
            return (status); \
        } \
    } while (0)

#define NIXL_RETURN_IF_NOT_IN_PROG(status) \
    do { \
        if ((status) != NIXL_IN_PROG) { \
            NIXL_LOG_AND_RETURN_IF_ERROR(status, " Received handle with pre-existing error"); \
            return (status); \
        } \
    } while (0)

#endif /* STATUS_H */
