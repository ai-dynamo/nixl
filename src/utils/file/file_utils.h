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
#ifndef __FILE_UTILS_H
#define __FILE_UTILS_H

#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <sys/stat.h>
#include "nixl_types.h"

/**
 * @brief File utilities for NIXL file backends
 */

namespace nixl {

/**
 * @brief Query file information for a single file
 * @param filename The filename to query (can be prefixed)
 * @return nixl_query_resp_t containing file info if accessible, std::nullopt otherwise
 */
nixl_query_resp_t
queryFileInfo(std::string_view filename);

/**
 * @brief Query file information for multiple files
 * @param filenames Vector of filenames to query (can be prefixed)
 * @param resp Output response vector
 * @return NIXL_SUCCESS on success, error code otherwise
 */
nixl_status_t
queryFileInfoList(const std::vector<std::string> &filenames, std::vector<nixl_query_resp_t> &resp);

} // namespace nixl

#endif // __FILE_UTILS_H
