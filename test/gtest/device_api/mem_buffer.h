/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_TEST_GTEST_DEVICE_API_MEM_BUFFER_CUH
#define NIXL_TEST_GTEST_DEVICE_API_MEM_BUFFER_CUH

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace nixl::gpu {
class memBuffer {
public:
    explicit memBuffer(size_t size, const void *src = nullptr)
        : size_{size},
          ptr_{malloc(size), free} {
        if (src) {
            if (cudaMemcpy(ptr_.get(), src, size, cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("Failed to memcpy from host to device");
            }
        } else {
            if (cudaMemset(ptr_.get(), 0, size) != cudaSuccess) {
                throw std::runtime_error("Failed to memset memory");
            }
        }
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return size_;
    }

    [[nodiscard]] uintptr_t
    ptr() const {
        return reinterpret_cast<uintptr_t>(ptr_.get());
    }

    [[nodiscard]] bool
    operator==(const memBuffer &rhs) const {
        const auto host_buffer = std::make_unique<std::byte[]>(size_);
        copyToHost(host_buffer.get());
        return rhs == host_buffer.get();
    }

private:
    [[nodiscard]] static void *
    malloc(size_t size) {
        void *ptr;
        if (cudaMalloc(&ptr, size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate memory");
        }
        return ptr;
    }

    static void
    free(void *ptr) {
        cudaFree(ptr);
    }

    void
    copyToHost(void *dst) const {
        if (cudaMemcpy(dst, ptr_.get(), size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("Failed to memcpy from device to host");
        }
    }

    friend void
    PrintTo(const memBuffer &mb, std::ostream *os);

    [[nodiscard]] bool
    operator==(const void *rhs) const {
        const auto host_buffer = std::make_unique<std::byte[]>(size_);
        copyToHost(host_buffer.get());
        return std::memcmp(host_buffer.get(), rhs, size_) == 0;
    }

    size_t size_;
    std::unique_ptr<void, void (*)(void *)> ptr_;
};

inline void
PrintTo(const memBuffer &mb, std::ostream *os) {
    const size_t n = mb.size();
    *os << "memBuffer{" << n << " bytes, ";
    try {
        const size_t num_u64 = (n + sizeof(uint64_t) - 1U) / sizeof(uint64_t);
        std::vector<uint64_t> host(num_u64, 0U);
        if (n > 0) {
            mb.copyToHost(static_cast<void *>(host.data()));
        }
        *os << "u64={";
        *os << std::hex << std::setfill('0');
        for (size_t i = 0; i < host.size(); ++i) {
            if (i > 0) {
                *os << ',';
            }
            *os << "0x" << std::setw(16) << host[i];
        }
        *os << std::dec << '}';
    }
    catch (const std::exception &e) {
        *os << "u64=<copyToHost failed: " << e.what() << ">";
    }
    *os << '}';
}
} // namespace nixl::gpu
#endif // NIXL_TEST_GTEST_DEVICE_API_MEM_BUFFER_CUH
