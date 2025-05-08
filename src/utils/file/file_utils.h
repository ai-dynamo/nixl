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
#include <nixl_types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

/**
 * @brief File access modes
 */
enum class nixlFileMode {
    CREATE,     // Create new file (O_CREAT | O_WRONLY | O_TRUNC)
    READ_ONLY,  // Read only (O_RDONLY)
    WRITE_ONLY, // Write only (O_WRONLY)
    READ_WRITE  // Read and write (O_RDWR)
};

/**
 * @brief File operations
 */
enum class nixlFileOperation {
    OPEN,       // Open a file
    CLOSE,      // Close a file
    UNLINK,     // Delete file
    STAT        // Get file status
};

/**
 * @brief File utilities class for basic file operations
 */
class nixlFileUtils {
public:
    /**
     * @brief Opens a file with specified mode
     * @param path File path
     * @param mode File access mode
     * @param[out] fd File descriptor if successful
     * @return NIXL_SUCCESS on success, error code otherwise
     */
    static nixl_status_t openFile(const std::string& path, nixlFileMode mode, int& fd);

    /**
     * @brief Closes a file descriptor
     * @param fd File descriptor to close
     * @return NIXL_SUCCESS on success, error code otherwise
     */
    static nixl_status_t closeFile(int fd);

    /**
     * @brief Unlinks (deletes) a file
     * @param path Path to the file
     * @return NIXL_SUCCESS on success, error code otherwise
     */
    static nixl_status_t unlinkFile(const std::string& path);

    /**
     * @brief Check if file exists
     * @param path Path to the file
     * @return NIXL_SUCCESS if file exists, NIXL_ERR_NOT_FOUND if not exists, error code otherwise
     */
    static nixl_status_t fileExists(const std::string& path);

    /**
     * @brief Gets file size
     * @param fd File descriptor
     * @param[out] size File size in bytes
     * @return NIXL_SUCCESS on success, error code otherwise
     */
    static nixl_status_t getFileSize(int fd, size_t& size);

private:
    // Convert FileMode to system flags
    static int getModeFlags(nixlFileMode mode);
};

#endif // __FILE_UTILS_H 