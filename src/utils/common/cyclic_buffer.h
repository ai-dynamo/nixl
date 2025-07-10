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
#ifndef _NIXL_CYCLIC_BUFFER_H
#define _NIXL_CYCLIC_BUFFER_H
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <string>
#include "common/nixl_log.h"

template<typename T, size_t Size> class SharedRingBuffer {
private:
    struct BufferHeader {
        std::atomic<size_t> write_pos{0};
        std::atomic<size_t> read_pos{0};
        std::atomic<uint32_t> version{0};
        uint32_t expected_version{0};
        static constexpr size_t capacity = Size;
        static constexpr size_t mask = Size - 1; // Size must be power of 2

        BufferHeader() {
            static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
            static_assert(std::is_trivially_copyable<T>::value,
                          "T must be trivially copyable for shared memory");
        }
    };

    BufferHeader *header_;
    T *data_;
    int file_fd_;
    bool is_creator_;
    std::string file_path_;

    static constexpr size_t total_size = sizeof(BufferHeader) + sizeof(T) * Size;

public:
    SharedRingBuffer(const char *name, bool create = true, uint32_t version = 1)
        : file_fd_(-1),
          is_creator_(create),
          file_path_(name) {

        if (create) {
            NIXL_INFO << "Creating file-based shared memory on path: " << std::string(name);
            // Create or truncate file
            file_fd_ = open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
            if (file_fd_ == -1) {
                NIXL_WARN << "Failed to create file for shared memory";
                perror("open failed");
                return;
            }

            // Set file size
            if (ftruncate(file_fd_, total_size) == -1) {
                close(file_fd_);
                unlink(name);
                NIXL_WARN << "Failed to set file size";
                return;
            }
        } else {
            // Open existing file
            file_fd_ = open(name, O_RDWR, 0666);
            if (file_fd_ == -1) {
                NIXL_WARN << "Failed to open file for shared memory";
                return;
            }
        }

        // Map memory
        void *ptr = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, file_fd_, 0);
        if (ptr == MAP_FAILED) {
            close(file_fd_);
            if (create) {
                unlink(name);
            }
            NIXL_WARN << "Failed to map file memory";
            return;
        }

        header_ = static_cast<BufferHeader *>(ptr);
        data_ = reinterpret_cast<T *>(static_cast<char *>(ptr) + sizeof(BufferHeader));

        // Initialize header if creator
        if (create) {
            new (header_) BufferHeader();
            header_->version.store(version, std::memory_order_release);
            header_->expected_version = version;
        } else {
            // Check version compatibility
            uint32_t current_version = header_->version.load(std::memory_order_acquire);
            if (current_version != version) {
                munmap(header_, total_size);
                close(file_fd_);
                NIXL_WARN << "Version mismatch: expected " + std::to_string(version) + ", got " +
                        std::to_string(current_version);
                return;
            }
        }
    }

    ~SharedRingBuffer() {
        if (header_) {
            munmap(header_, total_size);
        }
        if (file_fd_ != -1) {
            close(file_fd_);
        }
    }

    // Non-copyable
    SharedRingBuffer(const SharedRingBuffer &) = delete;
    SharedRingBuffer &
    operator=(const SharedRingBuffer &) = delete;

    bool
    push(const T &item) {
        if (!header_) {
            NIXL_WARN << "SharedRingBuffer not initialized";
            return false;
        }
        size_t write_pos = header_->write_pos.load(std::memory_order_relaxed);
        size_t next_write = (write_pos + 1) & header_->mask;

        // Check if buffer is full
        if (next_write == header_->read_pos.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }

        // Write data
        data_[write_pos] = item;

        // Update write position
        header_->write_pos.store(next_write, std::memory_order_release);
        return true;
    }

    bool
    pop(T &item) {
        if (!header_) {
            NIXL_WARN << "SharedRingBuffer not initialized";
            return false;
        }
        size_t read_pos = header_->read_pos.load(std::memory_order_relaxed);

        // Check if buffer is empty
        if (read_pos == header_->write_pos.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }

        // Read data
        item = data_[read_pos];

        // Update read position
        size_t next_read = (read_pos + 1) & header_->mask;
        header_->read_pos.store(next_read, std::memory_order_release);
        return true;
    }

    size_t
    size() const {
        if (!header_) {
            NIXL_WARN << "SharedRingBuffer not initialized";
            return 0;
        }
        size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
        size_t read_pos = header_->read_pos.load(std::memory_order_acquire);
        return (write_pos - read_pos) & header_->mask;
    }

    bool
    empty() const {
        if (!header_) {
            NIXL_WARN << "SharedRingBuffer not initialized";
            return true;
        }
        return header_->read_pos.load(std::memory_order_acquire) ==
            header_->write_pos.load(std::memory_order_acquire);
    }

    bool
    full() const {
        if (!header_) {
            NIXL_WARN << "SharedRingBuffer not initialized";
            return true;
        }
        size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
        size_t next_write = (write_pos + 1) & header_->mask;
        return next_write == header_->read_pos.load(std::memory_order_acquire);
    }

    uint32_t
    get_version() const {
        if (!header_) {
            NIXL_WARN << "SharedRingBuffer not initialized";
            return 0;
        }
        return header_->version.load(std::memory_order_acquire);
    }

    static void
    cleanup(const char *name) {
        if (!name) {
            NIXL_WARN << "SharedRingBuffer cleanup: name is null";
            return;
        }
        unlink(name);
    }
};

#endif // _NIXL_CYCLIC_BUFFER_H
