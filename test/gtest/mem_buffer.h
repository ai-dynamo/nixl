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

#ifndef MEM_BUFFER_H
#define MEM_BUFFER_H

#include "nixl_types.h"
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <random>
#include <memory>
#include <type_traits>

namespace gtest {

template<nixl_mem_t MemType> class MemBuffer;

template<nixl_mem_t MemType> using MemBufferVec = std::vector<MemBuffer<MemType>>;

template<nixl_mem_t MemType> using MemBufferVec2D = std::vector<MemBufferVec<MemType>>;

template<nixl_mem_t MemType>
std::vector<uint8_t>
createRandomData(size_t size,
                 std::mt19937_64 &gen,
                 std::uniform_int_distribution<uint64_t> &distrib) {
    size_t aligned_size = (size + 7) & ~7;
    std::vector<uint8_t> data(aligned_size);

    for (size_t i = 0; i < aligned_size; i += 8) {
        uint64_t rand_val = distrib(gen);
        for (size_t j = 0; j < 8; ++j) {
            data[i + j] = static_cast<uint8_t>(rand_val >> (j * 8));
        }
    }

    data.resize(size);
    return data;
}

template<nixl_mem_t LocalMemType, nixl_mem_t RemoteMemType, nixl_xfer_op_t Mode>
void
initBuffersWithRandomData(MemBufferVec<LocalMemType> &local_buffers,
                          MemBufferVec<RemoteMemType> &remote_buffers,
                          size_t count,
                          size_t size,
                          std::mt19937_64 &gen,
                          std::uniform_int_distribution<uint64_t> &distrib) {
    for (size_t i = 0; i < count; i++) {
        auto random_data = createRandomData<LocalMemType>(size, gen, distrib);
        if constexpr (Mode == NIXL_WRITE) {
            local_buffers.emplace_back(std::move(random_data));
            remote_buffers.emplace_back(size);
        } else if constexpr (Mode == NIXL_READ) {
            local_buffers.emplace_back(size);
            remote_buffers.emplace_back(std::move(random_data));
        } else {
            static_assert(Mode == NIXL_WRITE || Mode == NIXL_READ, "Invalid transfer mode");
        }
    }
}

template<nixl_mem_t LocalMemType, nixl_mem_t RemoteMemType, nixl_xfer_op_t Mode>
void
zeroBuffers(MemBufferVec<LocalMemType> &local_buffers,
            MemBufferVec<RemoteMemType> &remote_buffers,
            size_t count) {
    if constexpr (Mode == NIXL_WRITE) {
        for (size_t i = 0; i < count; i++) {
            remote_buffers[i].zero();
        }
    } else if constexpr (Mode == NIXL_READ) {
        for (size_t i = 0; i < count; i++) {
            local_buffers[i].zero();
        }
    } else {
        static_assert(Mode == NIXL_WRITE || Mode == NIXL_READ, "Invalid transfer mode");
    }
}

template<> class MemBuffer<DRAM_SEG> {
public:
    MemBuffer(size_t size) : buffer_(size) {}

    MemBuffer(std::vector<uint8_t> &&data) : buffer_(std::move(data)) {}

    bool
    operator==(const MemBuffer<DRAM_SEG> &other) const {
        return buffer_ == other.buffer_;
    }

    uintptr_t
    data() const {
        return reinterpret_cast<uintptr_t>(buffer_.data());
    }

    size_t
    size() const {
        return buffer_.size();
    }

    void
    zero() {
        std::fill(buffer_.begin(), buffer_.end(), 0);
    }

private:
    std::vector<uint8_t> buffer_;
};

} // namespace gtest

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

namespace gtest {

template<> class MemBuffer<VRAM_SEG> {
public:
    MemBuffer(size_t size) : size_(size) {
        cudaError_t err = cudaMalloc(&buffer_, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory");
        }
    }

    MemBuffer(std::vector<uint8_t> &&data) : MemBuffer(data.size()) {
        // TODO
    }

    ~MemBuffer() {
        if (buffer_) {
            cudaFree(buffer_);
        }
    }

    MemBuffer(const MemBuffer &) = delete;
    MemBuffer &
    operator=(const MemBuffer &) = delete;

    MemBuffer(MemBuffer &&other) noexcept :
            buffer_(std::exchange(other.buffer_, nullptr)),
            size_(std::exchange(other.size_, 0)) {}

    MemBuffer &
    operator=(MemBuffer &&other) noexcept {
        if (this != &other) {
            if (buffer_) {
                cudaFree(buffer_);
            }
            buffer_ = std::exchange(other.buffer_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    bool
    operator==(const MemBuffer<VRAM_SEG> &other) const {
        // TODO
        return true;
    }

    uintptr_t
    data() const {
        return reinterpret_cast<uintptr_t>(buffer_);
    }

    size_t
    size() const {
        return size_;
    }

    void
    zero() {
        // TODO
    }

private:
    void *buffer_ = nullptr;
    size_t size_;
};

} // namespace gtest

#endif // HAVE_CUDA

#endif /* MEM_BUFFER_H */
