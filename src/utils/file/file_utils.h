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
#include <optional>
#include <tuple>

/**
 * @brief File access modes
 */
enum class nixlFileMode {
    READ_ONLY,          // Open existing file for reading
    WRITE_ONLY,         // Open existing file for writing
    READ_WRITE,         // Open existing file for reading and writing
    CREATE,             // Create new file or truncate existing
    READ_DIRECT,        // Open existing file for reading with O_DIRECT
    WRITE_DIRECT,       // Open existing file for writing with O_DIRECT
    READ_WRITE_DIRECT,  // Open existing file for reading and writing with O_DIRECT
    CREATE_DIRECT       // Create new file or truncate existing with O_DIRECT
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
     * @param use_direct Whether to use O_DIRECT flag
     * @return NIXL_SUCCESS on success, error code otherwise
     */
    static nixl_status_t openFile(const std::string& path, nixlFileMode mode, int& fd, bool use_direct = false);

    /**
     * @brief Opens a file using a prefixed path string
     * @param prefixed_path Path with prefix
     * @param[out] fd File descriptor if successful
     * @return NIXL_SUCCESS on success, error code otherwise
     *
     * Supported prefixes:
     * - "RD:"  - Open in read-only mode
     * - "WR:"  - Open in write-only mode (creates file if it doesn't exist)
     * - "RW:"  - Open in read-write mode
     * - "RDD:" - Open in read-only mode with O_DIRECT
     * - "WRD:" - Open in write-only mode with O_DIRECT
     * - "RWD:" - Open in read-write mode with O_DIRECT
     * - "CRD:" - Create/truncate with O_DIRECT
     */
    static nixl_status_t openPrefixedFile(const std::string& prefixed_path, int& fd);

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

    /**
     * @brief Validates if a file descriptor is valid and accessible
     * @param fd File descriptor to validate
     * @return NIXL_SUCCESS if valid, error code otherwise
     */
    static nixl_status_t validateFileDescriptor(int fd);

private:
    /**
     * @brief Convert FileMode to system flags
     * @param mode File access mode
     * @param use_direct Whether to use O_DIRECT flag
     * @return Combined flags or -1 if invalid mode
     */
    static int getModeFlags(nixlFileMode mode, bool use_direct = false);

    /**
     * @brief Parse a prefixed path string into mode and actual path
     * @param prefixed_path Path with prefix
     * @return tuple of (success, mode, path) where success indicates if parsing was successful
     */
    static std::tuple<bool, nixlFileMode, std::string> parsePrefixedPath(const std::string& prefixed_path);
};

#endif // __FILE_UTILS_H
