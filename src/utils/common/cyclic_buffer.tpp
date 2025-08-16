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

#include "cyclic_buffer.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "nixl_log.h"
#include "util.h"

template<typename T>
sharedRingBuffer<T>::sharedRingBuffer(const std::string &name, bool create, int version, size_t size)
    : header_(nullptr),
      data_(nullptr),
      bufferSize_(size) {

    if (create) {
        createCyclicBuffer(name, version);
    } else {
        openCyclicBuffer(name, version);
    }
}

template<typename T>
sharedRingBuffer<T>::~sharedRingBuffer() {
    if (header_) {
        msync(header_, getTotalSize(), MS_SYNC);
        munmap(header_, getTotalSize());
    }
}

template<typename T>
bool
sharedRingBuffer<T>::push(const T &item) {
    size_t write_pos = header_->write_pos.load(std::memory_order_relaxed);
    size_t next_write = (write_pos + 1) & header_->mask;

    if (next_write == header_->read_pos.load(std::memory_order_acquire))
        return false; // Buffer full

    data_[write_pos] = item;

    header_->write_pos.store(next_write, std::memory_order_release);
    return true;
}

template<typename T>
bool
sharedRingBuffer<T>::pop(T &item) {
    size_t read_pos = header_->read_pos.load(std::memory_order_relaxed);

    if (read_pos == header_->write_pos.load(std::memory_order_acquire)) return false;

    // Read data
    item = data_[read_pos];

    // Update read position
    size_t next_read = (read_pos + 1) & header_->mask;
    header_->read_pos.store(next_read, std::memory_order_release);
    return true;
}

template<typename T>
size_t
sharedRingBuffer<T>::size() const {
    size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
    size_t read_pos = header_->read_pos.load(std::memory_order_acquire);
    return (write_pos - read_pos) & header_->mask;
}

template<typename T>
bool
sharedRingBuffer<T>::empty() const {
    return header_->read_pos.load(std::memory_order_acquire) ==
        header_->write_pos.load(std::memory_order_acquire);
}

template<typename T>
bool
sharedRingBuffer<T>::full() const {
    size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
    size_t next_write = (write_pos + 1) & header_->mask;
    return next_write == header_->read_pos.load(std::memory_order_acquire);
}

template<typename T>
uint32_t
sharedRingBuffer<T>::version() const {
    return header_->version.load(std::memory_order_acquire);
}

template<typename T>
size_t
sharedRingBuffer<T>::capacity() const {
    return header_->capacity;
}

template<typename T>
sharedRingBuffer<T>::bufferHeader::bufferHeader(size_t size) : capacity(size), mask(size - 1) {
    if ((size & (size - 1)) != 0) {
        throw std::invalid_argument("Telemetry buffer size must be a power of 2");
    }

    static_assert(std::is_trivially_copyable<T>::value,
                  "T must be trivially copyable for shared memory");
}

template<typename T>
size_t
sharedRingBuffer<T>::getTotalSize() const {
    return sizeof(bufferHeader) + sizeof(T) * bufferSize_;
}

template<typename T>
void
sharedRingBuffer<T>::createCyclicBuffer(const std::string &name, int version) {
    NIXL_INFO << "Creating file-based shared memory on path: " << name
              << " with size: " << bufferSize_;
    if (bufferSize_ == 0) {
        throw std::invalid_argument("Cannot create buffer with size 0");
    }
    auto file_closer = [](int *fd) {
        close(*fd);
        delete fd;
    };

    int fd = open(name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd == -1) {
        NIXL_ERROR << "Failed to open a file for shared memory: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("Failed to open a file for shared memory");
    }

    std::unique_ptr<int, decltype(file_closer)> file_fd(new int(fd), file_closer);

    if (ftruncate(*file_fd, getTotalSize()) == -1) {
        NIXL_ERROR << "Failed to set file size: " << name << " with error: " << strerror(errno);
        unlink(name.c_str());
        throw std::runtime_error("Failed to set file size");
    }

    void *ptr = mmap(nullptr, getTotalSize(), PROT_READ | PROT_WRITE, MAP_SHARED, *file_fd, 0);
    if (ptr == MAP_FAILED) {
        NIXL_ERROR << "Failed to map file memory: " << name
                   << " with error: " << strerror(errno);
        unlink(name.c_str());
        throw std::runtime_error("Failed to map file memory");
    }

    header_ = static_cast<bufferHeader *>(ptr);
    data_ = reinterpret_cast<T *>(static_cast<char *>(ptr) + sizeof(bufferHeader));

    new (header_) bufferHeader(bufferSize_);
    header_->version.store(version, std::memory_order_release);
    header_->expected_version = version;
}

template<typename T>
void
sharedRingBuffer<T>::openCyclicBuffer(const std::string &name, int version) {
    // Use a lambda with custom deleter to auto-close the file descriptor
    auto file_closer = [](int *fd) {
        close(*fd);
        delete fd;
    };

    int fd = open(name.c_str(), O_RDWR);
    if (fd == -1) {
        NIXL_ERROR << "Failed to open a file for shared memory: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("Failed to open a file for shared memory");
    }

    std::unique_ptr<int, decltype(file_closer)> file_fd(new int(fd), file_closer);

    // Check file size before mapping
    struct stat st;
    if (fstat(*file_fd, &st) == -1) {
        NIXL_ERROR << "Failed to get file stats: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("Failed to get file stats");
    }

    if (static_cast<size_t>(st.st_size) < sizeof(bufferHeader)) {
        NIXL_ERROR << "File too small for buffer header: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("File too small for buffer header");
    }

    // First, map just the header to read the size
    void *header_ptr =
        mmap(nullptr, sizeof(bufferHeader), PROT_READ | PROT_WRITE, MAP_SHARED, *file_fd, 0);
    if (header_ptr == MAP_FAILED) {
        unlink(name.c_str());
        NIXL_ERROR << "Failed to map header memory: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("Failed to map header memory");
    }

    bufferHeader *temp_header = static_cast<bufferHeader *>(header_ptr);

    // Check version compatibility
    int current_version = temp_header->version.load(std::memory_order_acquire);
    if (current_version != version) {
        munmap(temp_header, sizeof(bufferHeader));
        unlink(name.c_str());
        NIXL_ERROR << "Version mismatch: expected " + std::to_string(version) + ", got " +
                std::to_string(current_version);
        throw std::runtime_error("Version mismatch: expected " + std::to_string(version) +
                                 ", got " + std::to_string(current_version));
    }

    // Read the buffer size from header
    bufferSize_ = temp_header->capacity;
    NIXL_INFO << "Reading existing buffer with size: " << bufferSize_;

    // Check if file is large enough for the entire buffer
    if (static_cast<size_t>(st.st_size) < getTotalSize()) {
        munmap(temp_header, sizeof(bufferHeader));
        NIXL_ERROR << "File too small for buffer data: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("File too small for buffer data");
    }

    // Unmap the header and remap the entire buffer
    munmap(temp_header, sizeof(bufferHeader));

    // Map the entire buffer
    void *ptr = mmap(nullptr, getTotalSize(), PROT_READ | PROT_WRITE, MAP_SHARED, *file_fd, 0);
    if (ptr == MAP_FAILED) {
        NIXL_ERROR << "Failed to map file memory: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("Failed to map file memory");
    }

    header_ = static_cast<bufferHeader *>(ptr);
    data_ = reinterpret_cast<T *>(static_cast<char *>(ptr) + sizeof(bufferHeader));
}

template class sharedRingBuffer<uint8_t>;
