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

#include "cyclic_buffer.h"

#include <sys/file.h>
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
    size_t next_write = (write_pos + 1) & cachedMask_;

    // Don't use full(): it re-reads the consumer-owned read_pos on every call.
    // The cached check only touches that line when the buffer appears full.
    if (next_write == cachedReadPos_) {
        cachedReadPos_ = header_->read_pos.load(std::memory_order_acquire);
        if (next_write == cachedReadPos_) {
            return false; // Buffer full
        }
    }

    data_[write_pos] = item;

    header_->write_pos.store(next_write, std::memory_order_release);
    return true;
}

template<typename T>
bool
sharedRingBuffer<T>::pop(T &item) {
    size_t read_pos = header_->read_pos.load(std::memory_order_relaxed);

    // Don't use empty(): it re-reads the producer-owned write_pos on every call.
    // The cached check only touches that line when the buffer appears empty.
    if (read_pos == cachedWritePos_) {
        cachedWritePos_ = header_->write_pos.load(std::memory_order_acquire);
        if (read_pos == cachedWritePos_) {
            return false;
        }
    }

    // Read data
    item = data_[read_pos];

    // Update read position
    size_t next_read = (read_pos + 1) & cachedMask_;
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
    static_assert(sizeof(bufferHeader) == 2 * DESTRUCTIVE_INTERFERENCE_SIZE,
                  "bufferHeader layout changed.");
    static_assert(offsetof(bufferHeader, write_pos) == 0 &&
                      offsetof(bufferHeader, capacity) == 8 &&
                      offsetof(bufferHeader, version) == 16 &&
                      offsetof(bufferHeader, expected_version) == 20 &&
                      offsetof(bufferHeader, mask) == 24 &&
                      offsetof(bufferHeader, read_pos) == 256,
                  "bufferHeader field offsets changed.");
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
    // auto closer for fd. currently the producer does not need to keep its fd open
    auto file_closer = [](int *fd) {
        // Release flock so consumer can attach
        flock(*fd, LOCK_UN);
        close(*fd);
        delete fd;
    };

    // An existing file is handled in this order:
    // - version or capacity differ: replaced on a new inode, an old consumer keeps its mapping
    // - a consumer attached (shared flock held): reused with its positions
    // - no consumer attached: reset in place, unread events discarded
    // - the consumer cannot be probed: replaced on a new inode, an old consumer keeps its mapping
    enum class file_action_t { CREATE, REUSE, RESET, REPLACE };
    file_action_t action = file_action_t::CREATE;
    int fd = open(name.c_str(), O_RDWR | O_CLOEXEC);
    if (fd == -1 && errno != ENOENT) {
        // only a missing file falls through to create
        NIXL_ERROR << "Failed to open a file for shared memory: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("Failed to open a file for shared memory");
    }
    if (fd != -1) {
        // check compatibility
        bool compatible = false;
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            NIXL_WARN << "Cannot stat telemetry buffer " << name << " (" << strerror(errno)
                      << "), replacing it on a new file";
        } else if (static_cast<size_t>(st.st_size) != getTotalSize()) {
            NIXL_INFO << "Telemetry buffer " << name
                      << " exists with a different size, replacing it on a new file";
        } else {
            void *hp = mmap(nullptr, sizeof(bufferHeader), PROT_READ, MAP_SHARED, fd, 0);
            if (hp == MAP_FAILED) {
                NIXL_WARN << "Cannot map telemetry buffer header " << name << " ("
                          << strerror(errno) << "), replacing it on a new file";
            } else {
                const bufferHeader *h = static_cast<const bufferHeader *>(hp);
                // Only compatible if version, capacity, mask and positions are valid
                compatible = h->version.load(std::memory_order_acquire) == version &&
                    h->capacity == bufferSize_ && h->mask == bufferSize_ - 1 &&
                    h->write_pos.load(std::memory_order_acquire) <= h->mask &&
                    h->read_pos.load(std::memory_order_acquire) <= h->mask;
                munmap(hp, sizeof(bufferHeader));
                if (!compatible) {
                    NIXL_INFO << "Telemetry buffer " << name
                              << " exists with a different version or capacity, replacing it on "
                                 "a new file";
                }
            }
        }
        if (!compatible) {
            action = file_action_t::REPLACE;
        } else {
            const int probe = flock(fd, LOCK_EX | LOCK_NB);
            if (probe != 0 && errno == EWOULDBLOCK) {
                // consumer attached: keep its positions
                action = file_action_t::REUSE;
            } else if (probe == 0) {
                // no consumer holds the file: reset it under this lock
                NIXL_INFO << "Telemetry buffer " << name
                          << " has no consumer attached, resetting it in place and discarding "
                             "its unread events";
                action = file_action_t::RESET;
            } else {
                // lock unusable, cannot tell
                NIXL_WARN << "Cannot probe the consumer of telemetry buffer " << name << " ("
                          << strerror(errno) << "), replacing it on a new file";
                action = file_action_t::REPLACE;
            }
        }
        if (action == file_action_t::REPLACE) {
            close(fd);
            fd = -1;
            if (unlink(name.c_str()) != 0) {
                NIXL_ERROR << "Failed to remove telemetry buffer " << name
                           << " with error: " << strerror(errno);
                throw std::runtime_error("Failed to remove telemetry buffer");
            }
        }
    }
    if (action == file_action_t::CREATE || action == file_action_t::REPLACE) {
        fd = open(name.c_str(),
                  O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    }
    if (fd == -1) {
        NIXL_ERROR << "Failed to open a file for shared memory: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("Failed to open a file for shared memory");
    }
    // Taken until the header is complete, so a consumer opening meanwhile waits for it
    if (action != file_action_t::REUSE && flock(fd, LOCK_EX | LOCK_NB) != 0) {
        NIXL_WARN << "Cannot lock telemetry buffer " << name << " (" << strerror(errno)
                  << "); a consumer may attach before the header is complete";
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

    // Keep the positions of a reused file, new header for a reset, replaced or fresh one
    if (action != file_action_t::REUSE) {
        new (header_) bufferHeader(bufferSize_);
        header_->version.store(version, std::memory_order_release);
        header_->expected_version = version;
    }

    // Seed the cached values from the header
    cachedWritePos_ = header_->write_pos.load(std::memory_order_acquire);
    cachedReadPos_ = header_->read_pos.load(std::memory_order_acquire);
    cachedMask_ = header_->mask;

    if (action == file_action_t::REUSE) {
        NIXL_INFO << "Telemetry buffer " << name
                  << " has a consumer attached, reusing it and its positions with " << size()
                  << " unread events";
    }
}

template<typename T>
void
sharedRingBuffer<T>::openCyclicBuffer(const std::string &name, int version) {
    // File descriptor is kept open, it holds the shared lock below
    fd_ = nixl::scopedFd(open(name.c_str(), O_RDWR | O_CLOEXEC));
    if (!fd_.valid()) {
        NIXL_ERROR << "Failed to open a file for shared memory: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("Failed to open a file for shared memory");
    }

    // Shared lock for the object's lifetime: tells a restarting producer a consumer is attached
    while (flock(fd_.get(), LOCK_SH) != 0) {
        if (errno != EINTR) {
            // not fatal: the ring is still read, only this consumer's presence stops being provable
            NIXL_WARN << "Cannot lock telemetry buffer " << name << " (" << strerror(errno)
                      << "); a restarting producer cannot tell this consumer is attached and "
                         "will not keep the buffer for it";
            break;
        }
    }

    // Check file size before mapping
    struct stat st;
    if (fstat(fd_.get(), &st) == -1) {
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
        mmap(nullptr, sizeof(bufferHeader), PROT_READ | PROT_WRITE, MAP_SHARED, fd_.get(), 0);
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
    void *ptr = mmap(nullptr, getTotalSize(), PROT_READ | PROT_WRITE, MAP_SHARED, fd_.get(), 0);
    if (ptr == MAP_FAILED) {
        NIXL_ERROR << "Failed to map file memory: " << name
                   << " with error: " << strerror(errno);
        throw std::runtime_error("Failed to map file memory");
    }

    header_ = static_cast<bufferHeader *>(ptr);
    data_ = reinterpret_cast<T *>(static_cast<char *>(ptr) + sizeof(bufferHeader));

    // Seed the cached values from the header
    cachedWritePos_ = header_->write_pos.load(std::memory_order_acquire);
    cachedReadPos_ = header_->read_pos.load(std::memory_order_acquire);
    cachedMask_ = header_->mask;
}

template class sharedRingBuffer<uint8_t>;
