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
#ifndef NIXL_SRC_UTILS_FILE_FILE_UTILS_H
#define NIXL_SRC_UTILS_FILE_FILE_UTILS_H

#include "nixl_types.h"

#include <string>
#include <vector>

/**
 * @brief File utilities for NIXL file backends
 */

namespace nixl {

/**
 * @brief Query file information for a single file
 * @param filename The filename to query (can be prefixed)
 * @return nixl_query_resp_t containing file info if accessible, std::nullopt otherwise
 */
[[nodiscard]] nixl_query_resp_t
queryFileInfo(const std::string &filename);

/**
 * @brief Query file information for multiple files
 * @param descs Descriptors from which to take the filenames
 * @return std::vector<nixl_query_resp_t> with each entry obtained from queryFileInfo()
 */
template<typename DescList>
[[nodiscard]] std::vector<nixl_query_resp_t>
queryFileInfoFromDescList(const DescList &descs) {
    std::vector<nixl_query_resp_t> resp;

    resp.reserve(descs.descCount());

    for (const auto &desc : descs) {
        resp.emplace_back(queryFileInfo(desc.metaInfo));
    }
    return resp;
}

} // namespace nixl

#endif
