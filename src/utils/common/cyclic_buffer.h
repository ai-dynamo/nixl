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
#ifndef _NIXL_CYCLIC_BUFFER_HPP
#define _NIXL_CYCLIC_BUFFER_HPP

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <string>
#include <memory>

template<typename T> class sharedRingBuffer {
public:
    sharedRingBuffer(const std::string &name, bool create, int version, size_t size = 0);
    ~sharedRingBuffer();

    // Non-copyable
    sharedRingBuffer(const sharedRingBuffer &) = delete;
    sharedRingBuffer &
    operator=(const sharedRingBuffer &) = delete;

    bool
    push(const T &item);
    bool
    pop(T &item);
    size_t
    size() const;
    bool
    empty() const;
    bool
    full() const;
    uint32_t
    version() const;
    size_t
    capacity() const;

private:
    struct bufferHeader {
        std::atomic<size_t> write_pos{0};
        std::atomic<size_t> read_pos{0};
        std::atomic<int> version{0};
        int expected_version{0};
        const size_t capacity;
        size_t mask;

        bufferHeader(size_t size);
    };

    size_t
    getTotalSize() const;
    void
    createCyclicBuffer(const std::string &name, int version);
    void
    openCyclicBuffer(const std::string &name, int version);

    bufferHeader *header_;
    T *data_;
    size_t bufferSize_;
};

#include "cyclic_buffer.tpp"
#endif // _NIXL_CYCLIC_BUFFER_HPP
