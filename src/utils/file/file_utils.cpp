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
#include "file_path_mode.h"

#include <fcntl.h>
#include <optional>
#include <unistd.h>
#include <sys/stat.h>

namespace nixl {

nixl_query_resp_t
queryFileInfo(const std::string &filename) {

    if (filename.empty()) {
        return std::nullopt;
    }

    // A path-mode metaInfo ("ro:/path", "rw,direct:/path", ...) names the file after the
    // modes prefix; stat that path, as registerMem would open it. Anything else is a plain path.
    const auto spec = parsePathMeta(filename);
    const std::string &path = spec ? spec->path : filename;

    struct ::stat stat_buf;
    if (::stat(path.c_str(), &stat_buf) != 0) {
        return std::nullopt;
    }

    return nixl_b_params_t{{"size", std::to_string(stat_buf.st_size)},
                           {"mode", std::to_string(stat_buf.st_mode)},
                           {"mtime", std::to_string(stat_buf.st_mtime)}};
}

std::vector<nixl_query_resp_t>
queryFileInfoFromDescList(const nixl_reg_dlist_t &descs) {
    std::vector<nixl_query_resp_t> resp;

    resp.reserve(descs.descCount());

    for (const auto &desc : descs) {
        resp.emplace_back(queryFileInfo(desc.metaInfo));
    }
    return resp;
}

} // namespace nixl
