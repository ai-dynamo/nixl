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
#include "file_utils.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace nixl {

nixl_query_resp_t
queryFileInfo(const std::string &filename) {

    if (filename.empty()) {
        return std::nullopt;
    }

    struct ::stat stat_buf;
    if (::stat(filename.c_str(), &stat_buf) != 0) {
        return std::nullopt;
    }

    return nixl_b_params_t{
        {"size", std::to_string(stat_buf.st_size)},
        {"mode", std::to_string(stat_buf.st_mode)},
        {"mtime", std::to_string(stat_buf.st_mtime)}
    };
}

} // namespace nixl
