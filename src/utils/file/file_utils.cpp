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

int nixlFileUtils::getModeFlags(nixlFileMode mode) {
    switch (mode) {
        case nixlFileMode::CREATE:
            return O_CREAT | O_WRONLY | O_TRUNC;
        case nixlFileMode::READ_ONLY:
            return O_RDONLY;
        case nixlFileMode::WRITE_ONLY:
            return O_WRONLY;
        case nixlFileMode::READ_WRITE:
            return O_RDWR;
        default:
            return -1;
    }
}

nixl_status_t nixlFileUtils::openFile(const std::string& path, nixlFileMode mode, int& fd) {
    int flags = getModeFlags(mode);
    if (flags == -1) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // For create mode, set permissions to 0644 (rw-r--r--)
    mode_t permissions = (mode == nixlFileMode::CREATE) ? 0644 : 0;
    
    fd = open(path.c_str(), flags, permissions);
    if (fd == -1) {
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

    return NIXL_SUCCESS;
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