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
#include "util.h"

template<typename T> class SharedRingBuffer {
private:
    struct BufferHeader {
        std::atomic<size_t> write_pos{0};
        std::atomic<size_t> read_pos{0};
        std::atomic<uint32_t> version{0};
        uint32_t expected_version{0};
        size_t capacity;
        size_t mask;

        BufferHeader(size_t size) : capacity(size), mask(size - 1) {
            // Ensure size is power of 2
            if ((size & (size - 1)) != 0) {
                size = nextPowerOf2(size);
                NIXL_WARN << "Size " << size << " is not power of 2, rounding up to: " << size;
            }
            static_assert(std::is_trivially_copyable<T>::value,
                          "T must be trivially copyable for shared memory");
        }
    };

    BufferHeader *header_;
    T *data_;
    int file_fd_;
    bool is_creator_;
    std::string file_path_;
    bool initialized_;
    size_t buffer_size_;

    size_t
    getTotalSize() const {
        return sizeof(BufferHeader) + sizeof(T) * buffer_size_;
    }

public:
    SharedRingBuffer()
        : header_(nullptr),
          data_(nullptr),
          file_fd_(-1),
          is_creator_(false),
          file_path_(""),
          initialized_(false),
          buffer_size_(0) {}

    SharedRingBuffer(const char *name, size_t size, bool create, uint32_t version = 1)
        : file_fd_(-1),
          is_creator_(create),
          file_path_(name),
          initialized_(false),
          buffer_size_(size) {
        initialize(name, size, create, version);
    }

    // Constructor for reading existing buffer (size will be read from header)
    SharedRingBuffer(const char *name, uint32_t version = 1)
        : file_fd_(-1),
          is_creator_(false),
          file_path_(name),
          initialized_(false),
          buffer_size_(0) {
        initialize(name, 0, false, version);
    }

    void
    initialize(const char *name, size_t size = 0, bool create = true, uint32_t version = 1) {
        if (file_fd_ != -1) {
            NIXL_WARN << "SharedRingBuffer already initialized";
            return;
        }

        if (create && size == 0) {
            NIXL_WARN << "Cannot create buffer with size 0";
            return;
        }

        if (create) {
            buffer_size_ = size;
            NIXL_INFO << "Creating file-based shared memory on path: " << std::string(name)
                      << " with size: " << size;
            // Create or truncate file
            file_fd_ = open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
            if (file_fd_ == -1) {
                NIXL_WARN << "Failed to create file for shared memory";
                perror("open failed");
                return;
            }

            // Set file size
            if (ftruncate(file_fd_, getTotalSize()) == -1) {
                close(file_fd_);
                unlink(name);
                NIXL_WARN << "Failed to set file size";
                return;
            }

            // Map memory
            void *ptr =
                mmap(nullptr, getTotalSize(), PROT_READ | PROT_WRITE, MAP_SHARED, file_fd_, 0);
            if (ptr == MAP_FAILED) {
                close(file_fd_);
                unlink(name);
                NIXL_WARN << "Failed to map file memory";
                return;
            }

            header_ = static_cast<BufferHeader *>(ptr);
            data_ = reinterpret_cast<T *>(static_cast<char *>(ptr) + sizeof(BufferHeader));

            // Initialize header
            new (header_) BufferHeader(size);
            header_->version.store(version, std::memory_order_release);
            header_->expected_version = version;
            initialized_ = true;
        } else {
            // Reading existing buffer
            if (size == 0) {
                // Auto-detect size from header
                initializeFromHeader(name, version);
            } else {
                // Use provided size
                buffer_size_ = size;
                initializeWithSize(name, version);
            }
        }
    }

private:
    // Helper method to initialize by reading size from header
    void
    initializeFromHeader(const char *name, uint32_t version) {
        // Open existing file
        file_fd_ = open(name, O_RDWR, 0666);
        if (file_fd_ == -1) {
            NIXL_WARN << "Failed to open file for shared memory";
            return;
        }

        // First, map just the header to read the size
        void *header_ptr =
            mmap(nullptr, sizeof(BufferHeader), PROT_READ | PROT_WRITE, MAP_SHARED, file_fd_, 0);
        if (header_ptr == MAP_FAILED) {
            close(file_fd_);
            NIXL_WARN << "Failed to map header memory";
            return;
        }

        BufferHeader *temp_header = static_cast<BufferHeader *>(header_ptr);

        // Check version compatibility
        uint32_t current_version = temp_header->version.load(std::memory_order_acquire);
        if (current_version != version) {
            munmap(temp_header, sizeof(BufferHeader));
            close(file_fd_);
            NIXL_WARN << "Version mismatch: expected " + std::to_string(version) + ", got " +
                    std::to_string(current_version);
            return;
        }

        // Read the buffer size from header
        buffer_size_ = temp_header->capacity;
        NIXL_INFO << "Reading existing buffer with size: " << buffer_size_;

        // Unmap the header and remap the entire buffer
        munmap(temp_header, sizeof(BufferHeader));

        // Map the entire buffer
        void *ptr = mmap(nullptr, getTotalSize(), PROT_READ | PROT_WRITE, MAP_SHARED, file_fd_, 0);
        if (ptr == MAP_FAILED) {
            close(file_fd_);
            NIXL_WARN << "Failed to map file memory";
            return;
        }

        header_ = static_cast<BufferHeader *>(ptr);
        data_ = reinterpret_cast<T *>(static_cast<char *>(ptr) + sizeof(BufferHeader));
        initialized_ = true;
    }

    // Helper method to initialize with known size
    void
    initializeWithSize(const char *name, uint32_t version) {
        // Open existing file
        file_fd_ = open(name, O_RDWR, 0666);
        if (file_fd_ == -1) {
            NIXL_WARN << "Failed to open file for shared memory";
            return;
        }

        // Map memory
        void *ptr = mmap(nullptr, getTotalSize(), PROT_READ | PROT_WRITE, MAP_SHARED, file_fd_, 0);
        if (ptr == MAP_FAILED) {
            close(file_fd_);
            NIXL_WARN << "Failed to map file memory";
            return;
        }

        header_ = static_cast<BufferHeader *>(ptr);
        data_ = reinterpret_cast<T *>(static_cast<char *>(ptr) + sizeof(BufferHeader));

        // Check version compatibility
        uint32_t current_version = header_->version.load(std::memory_order_acquire);
        if (current_version != version) {
            munmap(header_, getTotalSize());
            close(file_fd_);
            NIXL_WARN << "Version mismatch: expected " + std::to_string(version) + ", got " +
                    std::to_string(current_version);
            return;
        }
        initialized_ = true;
    }

public:
    ~SharedRingBuffer() {
        if (header_) {
            munmap(header_, getTotalSize());
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

    size_t
    get_capacity() const {
        if (!header_) {
            return 0;
        }
        return header_->capacity;
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
