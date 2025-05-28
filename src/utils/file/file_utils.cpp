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

#include <errno.h>
#include <file_utils.h>
#include <filesystem>
#include <algorithm>
#include <fcntl.h>
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>

int nixlFileUtils::getModeFlags(nixlFileMode mode, bool use_direct) {
    // Base flags for each mode
    switch (mode) {
        case nixlFileMode::CREATE:
        case nixlFileMode::CREATE_DIRECT:
            return ((mode == nixlFileMode::CREATE_DIRECT || use_direct) ? O_DIRECT : 0) |
                   O_CREAT | O_WRONLY | O_TRUNC;

        case nixlFileMode::READ_ONLY:
        case nixlFileMode::READ_DIRECT:
            return ((mode == nixlFileMode::READ_DIRECT || use_direct) ? O_DIRECT : 0) |
                   O_RDONLY;

        case nixlFileMode::WRITE_ONLY:
        case nixlFileMode::WRITE_DIRECT:
            return ((mode == nixlFileMode::WRITE_DIRECT || use_direct) ? O_DIRECT : 0) |
                   O_WRONLY;

        case nixlFileMode::READ_WRITE:
        case nixlFileMode::READ_WRITE_DIRECT:
            return ((mode == nixlFileMode::READ_WRITE_DIRECT || use_direct) ? O_DIRECT : 0) |
                   O_RDWR | O_CREAT;

        default:
            return -1;
    }
}

nixl_status_t nixlFileUtils::openFile(const std::string& path, nixlFileMode mode, int& fd, bool use_direct) {
    int flags = getModeFlags(mode, use_direct);
    if (flags == -1) {
        NIXL_ERROR << absl::StrFormat("Invalid mode flags for path: %s", path);
        return NIXL_ERR_INVALID_PARAM;
    }

    // For WRITE_ONLY, check if file exists and add O_CREAT if it doesn't
    if (mode == nixlFileMode::WRITE_ONLY) {
        nixl_status_t status = fileExists(path);
        if (status == NIXL_ERR_NOT_FOUND) {
            NIXL_DEBUG << absl::StrFormat("File doesn't exist, adding O_CREAT flag for: %s", path);
            flags |= O_CREAT;
        } else if (status != NIXL_SUCCESS) {
            NIXL_ERROR << absl::StrFormat("Failed to check file existence: %s", path);
            return status;
        }
    }

    // Set permissions only for modes that might create a file
    mode_t permissions = ((mode == nixlFileMode::CREATE) ||
                         (mode == nixlFileMode::READ_WRITE) ||
                         (flags & O_CREAT)) ? 0644 : 0;

    NIXL_DEBUG << absl::StrFormat("Opening file: %s with flags: 0x%x and permissions: %o%s",
                                 path, flags, permissions,
                                 use_direct ? " (using O_DIRECT)" : "");

    fd = open(path.c_str(), flags, permissions);
    if (fd == -1) {
        if (errno == EINVAL && (flags & O_DIRECT)) {
            // If O_DIRECT fails, try without it but warn
            NIXL_ERROR << absl::StrFormat("O_DIRECT not supported for %s, falling back to buffered I/O", path);
            flags &= ~O_DIRECT;
            fd = open(path.c_str(), flags, permissions);
        }

        if (fd == -1) {
            NIXL_ERROR << absl::StrFormat("Failed to open file: %s - %s", path, strerror(errno));
            switch (errno) {
                case EACCES:
                    return NIXL_ERR_NOT_ALLOWED;
                case EEXIST:
                    return NIXL_ERR_INVALID_PARAM;
                case ENOENT:
                    return NIXL_ERR_NOT_FOUND;
                default:
                    return NIXL_ERR_BACKEND;
            }
        }
    }

    NIXL_DEBUG << absl::StrFormat("Successfully opened file: %s with fd: %d", path, fd);
    return NIXL_SUCCESS;
}

nixl_status_t nixlFileUtils::openPrefixedFile(const std::string& prefixed_path, int& fd) {
    NIXL_DEBUG << absl::StrFormat("Attempting to open prefixed path: %s", prefixed_path);

    auto [success, mode, path] = parsePrefixedPath(prefixed_path);
    if (!success) {
        // Error already logged in parsePrefixedPath
        return NIXL_ERR_INVALID_PARAM;
    }

    // Check if this is a direct I/O prefix (3-letter prefix ending with 'D')
    bool use_direct = prefixed_path.length() >= 4 && prefixed_path[2] == 'D' && prefixed_path[3] == ':';

    NIXL_DEBUG << absl::StrFormat("Opening with mode: %d, path: %s, direct I/O: %s",
                                 static_cast<int>(mode), path, use_direct ? "Yes" : "No");

    return openFile(path, mode, fd, use_direct);
}

std::tuple<bool, nixlFileMode, std::string> nixlFileUtils::parsePrefixedPath(const std::string& prefixed_path) {
    // Minimum valid path is "XX:/path" (5 chars)
    if (prefixed_path.empty()) {
        NIXL_ERROR << "Invalid path: empty string";
        return {false, nixlFileMode::READ_ONLY, ""};
    }

    // Find the colon separator
    size_t colon_pos = prefixed_path.find(':');
    if (colon_pos == std::string::npos) {
        NIXL_ERROR << absl::StrFormat("Invalid path format: missing ':' in '%s'", prefixed_path);
        return {false, nixlFileMode::READ_ONLY, ""};
    }

    // Extract and validate prefix
    std::string prefix = prefixed_path.substr(0, colon_pos);
    if (prefix.empty() || (prefix.length() != 2 && prefix.length() != 3)) {
        NIXL_ERROR << absl::StrFormat("Invalid prefix length: '%s' (must be 2 or 3 characters)", prefix);
        return {false, nixlFileMode::READ_ONLY, ""};
    }

    // Convert prefix to uppercase for comparison
    std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::toupper);

    // Extract path (everything after the colon)
    std::string path = prefixed_path.substr(colon_pos + 1);
    if (path.empty()) {
        NIXL_ERROR << absl::StrFormat("Invalid path: no path after prefix '%s:'", prefix);
        return {false, nixlFileMode::READ_ONLY, ""};
    }

    // Map prefix to file mode
    nixlFileMode mode;
    if (prefix == "RD") {
        mode = nixlFileMode::READ_ONLY;
    } else if (prefix == "WR") {
        mode = nixlFileMode::WRITE_ONLY;
    } else if (prefix == "RW") {
        mode = nixlFileMode::READ_WRITE;
    } else if (prefix == "CR") {
        mode = nixlFileMode::CREATE;
    } else if (prefix == "RWD") {
        mode = nixlFileMode::READ_WRITE_DIRECT;
    } else if (prefix == "RDD") {
        mode = nixlFileMode::READ_DIRECT;
    } else if (prefix == "WRD") {
        mode = nixlFileMode::WRITE_DIRECT;
    } else if (prefix == "CRD") {
        mode = nixlFileMode::CREATE_DIRECT;
    } else {
        NIXL_ERROR << absl::StrFormat("Invalid prefix: '%s' (valid prefixes: RD, WR, RW, CR, RWD, RDD, WRD, CRD)", prefix);
        return {false, nixlFileMode::READ_ONLY, ""};
    }

    return {true, mode, path};
}

nixl_status_t nixlFileUtils::closeFile(int fd) {
    if (close(fd) == -1) {
        switch (errno) {
            case EBADF:
                return NIXL_ERR_INVALID_PARAM;
            default:
                return NIXL_ERR_BACKEND;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlFileUtils::unlinkFile(const std::string& path) {
    if (unlink(path.c_str()) == -1) {
        switch (errno) {
            case EACCES:
            case EPERM:
                return NIXL_ERR_NOT_ALLOWED;
            case ENOENT:
                return NIXL_ERR_NOT_FOUND;
            default:
                return NIXL_ERR_BACKEND;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlFileUtils::fileExists(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == -1) {
        if (errno == ENOENT) {
            return NIXL_ERR_NOT_FOUND;
        }
        switch (errno) {
            case EACCES:
                return NIXL_ERR_NOT_ALLOWED;
            default:
                return NIXL_ERR_BACKEND;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlFileUtils::getFileSize(int fd, size_t& size) {
    struct stat st;
    if (fstat(fd, &st) == -1) {
        switch (errno) {
            case EBADF:
                return NIXL_ERR_INVALID_PARAM;
            default:
                return NIXL_ERR_BACKEND;
        }
    }
    size = st.st_size;
    return NIXL_SUCCESS;
}

nixl_status_t nixlFileUtils::validateFileDescriptor(int fd) {
    if (fcntl(fd, F_GETFD) == -1) {
        return errno == EBADF ? NIXL_ERR_INVALID_PARAM : NIXL_ERR_NOT_ALLOWED;
    }
    return NIXL_SUCCESS;
}
