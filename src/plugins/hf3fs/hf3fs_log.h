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
#ifndef HF3FS_LOG_H
#define HF3FS_LOG_H

#include <absl/strings/str_format.h>
#include "common/nixl_log.h"

#define HF3FS_LOG_RETURN(error_code, message) \
    do { \
        NIXL_ERROR << absl::StrFormat("Error: %d - %s", (error_code), (message)); \
        return error_code; \
    } while (0)

#endif // HF3FS_LOG_H
