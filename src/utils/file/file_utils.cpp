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
#include "file_utils.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <cstring>
#include <iostream>

namespace nixl {

std::optional<nixl_b_params_t>
queryFileInfo(std::string_view filename) {
    // If filename is empty, return nullopt (same as any invalid name)
    if (filename.empty()) {
        return std::nullopt;
    }

    // Check if file exists using stat
    struct stat stat_buf;
    if (stat(std::string(filename).c_str(), &stat_buf) != 0) {
        return std::nullopt;
    }

    nixl_b_params_t info;
    info["size"] = std::to_string(stat_buf.st_size);
    info["mode"] = std::to_string(stat_buf.st_mode);
    info["mtime"] = std::to_string(stat_buf.st_mtime);

    return info;
}

nixl_status_t
queryFileInfoList(const std::vector<std::string> &filenames, std::vector<nixl_query_resp_t> &resp) {
    resp.clear();
    resp.reserve(filenames.size());

    for (const auto &filename : filenames)
        resp.emplace_back(queryFileInfo(filename));

    return NIXL_SUCCESS;
}

} // namespace nixl
