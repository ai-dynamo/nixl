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

// Helper function to parse prefix and return both filename and prefix type
static std::pair<std::string, std::string> parsePrefixAndType(const std::string& prefix) {
    if (prefix.empty()) {
        return {prefix, ""};
    }

    // Parse prefix type from the beginning
    if (prefix.length() >= 3 && prefix[2] == ':') {
        std::string prefix_type = prefix.substr(0, 3);

        if (prefix_type == "RO:" ||
	    prefix_type == "RW:" ||
	    prefix_type == "WR:") {
            return {prefix.substr(3), prefix_type};
        }
    }

    // If no prefix or unknown prefix, return original string and empty prefix type
    return {prefix, ""};
}

int prefixedOpen(const std::string& prefix, int* fd, bool* was_opened) {
    if (prefix.empty() || !fd) {
        return -1;
    }

    // Handle open operations
    auto [actual_filename, prefix_type] = parsePrefixAndType(prefix);

    // If no prefix type found, this is not a prefixed filename
    if (prefix_type.empty()) {
        if (was_opened) {
            *was_opened = false; // File was not opened by NIXL
        }
        return 0; // Success, but no file was opened
    }

    int actual_flags = O_RDWR; // Default to read-write
    mode_t actual_mode = 0644; // Default mode

    // Determine flags and mode based on prefix type
    if (prefix_type == "RO:") {
        actual_flags = O_RDONLY;
        actual_mode = 0444; // Read-only mode
    } else if (prefix_type == "RW:") {
        actual_flags = O_RDWR;
        actual_mode = 0644; // Read-write mode
    } else if (prefix_type == "WR:") {
        actual_flags = O_WRONLY;
        actual_mode = 0222; // Write-only mode
    }
    // If prefix doesn't match known patterns, use default flags and mode

    if (actual_filename.empty()) {
        std::cerr << "Invalid filename after prefix parsing: "
		  << prefix << std::endl;
        return -1;
    }

    int new_fd = open(actual_filename.c_str(), actual_flags, actual_mode);
    if (new_fd == -1) {
        std::cerr << "Failed to open file: " << actual_filename
		  << " (from prefix: " << prefix << ") - "
		  << strerror(errno) << std::endl;
        return -1;
    }

    // Set the file descriptor through the pointer
    *fd = new_fd;
    if (was_opened) {
        *was_opened = true; // File was opened by NIXL
    }
    return 0; // Return 0 for successful open
}

nixl_status_t queryFileInfo(const std::string& filename, nixl_query_resp_t& resp) {
    // Initialize response
    resp.accessible = false;
    resp.info.clear();

    if (filename.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Extract actual filename from prefix
    std::string actual_filename = parsePrefixAndType(filename).first;

    if (actual_filename.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Check if file exists using stat
    struct stat stat_buf;
    bool file_exists = (stat(actual_filename.c_str(), &stat_buf) == 0);

    resp.accessible = file_exists;

    // Add additional file information if file exists
    if (file_exists) {
        resp.info["size"] = std::to_string(stat_buf.st_size);
        resp.info["mode"] = std::to_string(stat_buf.st_mode);
        resp.info["mtime"] = std::to_string(stat_buf.st_mtime);
    }

    return NIXL_SUCCESS;
}

nixl_status_t queryFileInfoList(const std::vector<std::string>& filenames, std::vector<nixl_query_resp_t>& resp) {
    resp.clear();
    resp.reserve(filenames.size());

    for (const auto& filename : filenames) {
        nixl_query_resp_t query_resp;
        nixl_status_t status = queryFileInfo(filename, query_resp);

        if (status != NIXL_SUCCESS) {
            // If one file fails, mark it as inaccessible but continue
            query_resp.accessible = false;
            query_resp.info.clear();
        }

        resp.push_back(query_resp);
    }

    return NIXL_SUCCESS;
}


