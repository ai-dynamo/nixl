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
#ifndef __TEMP_FILE_H
#define __TEMP_FILE_H

#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <mutex>

class tempFile {
public:
    int fd;
    std::string path;

    // Constructor: opens the file and stores the fd and path
    tempFile(const std::string& filename, int flags, mode_t mode = 0600)
        : path(filename)
    {
        print_protected("Opening file: " + filename + " (flags: " + std::to_string(flags) + ", mode: " + std::to_string(mode) + ")");
        fd = open(filename.c_str(), flags, mode);
        if (fd == -1) {
            print_protected("ERROR: Failed to open file: " + filename + " - " + strerror(errno));
            throw std::runtime_error("Failed to open file: " + filename + " - " + strerror(errno));
        }
        print_protected("Successfully opened file: " + filename + " (fd: " + std::to_string(fd) + ")");
    }

    // Deleted copy constructor and assignment to avoid double-close/unlink
    tempFile(const tempFile&) = delete;
    tempFile& operator=(const tempFile&) = delete;

    // Move constructor and assignment
    tempFile(tempFile&& other) noexcept
        : fd(other.fd), path(std::move(other.path))
    {
        other.fd = -1;
    }
    tempFile& operator=(tempFile&& other) noexcept {
        if (this != &other) {
            close_fd();
            path = std::move(other.path);
            fd = other.fd;
            other.fd = -1;
        }
        return *this;
    }

    // Conversion operator to int (file descriptor)
    operator int() const { return fd; }

    // Destructor: closes the fd and deletes the file
    ~tempFile() {
        close_fd();
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }

private:
    std::mutex cout_mutex;

    void close_fd() {
        if (fd != -1) {
            close(fd);
            fd = -1;
        }
    }

    void print_protected(const std::string& message) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << message << std::endl;
    }
};
#endif // __TEMP_FILE_H
