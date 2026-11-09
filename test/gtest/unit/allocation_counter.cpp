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
#include "allocation_counter.h"

#include <cstdlib>
#include <new>

namespace {

constinit thread_local bool counting = false;
constinit thread_local std::size_t allocations = 0;

[[nodiscard]] void *
allocate(std::size_t size) noexcept {
    if (counting) {
        ++allocations;
    }
    return std::malloc((size == 0) ? 1 : size);
}

[[nodiscard]] void *
allocate(std::size_t size, std::align_val_t alignment) noexcept {
    if (counting) {
        ++allocations;
    }
    void *memory = nullptr;
    const auto align = static_cast<std::size_t>(alignment);
    if (posix_memalign(&memory, (align < sizeof(void *)) ? sizeof(void *) : align, size) != 0) {
        return nullptr;
    }
    return memory;
}

[[nodiscard]] void *
allocateOrThrow(std::size_t size) {
    void *memory = allocate(size);
    if (memory == nullptr) {
        throw std::bad_alloc();
    }
    return memory;
}

[[nodiscard]] void *
allocateOrThrow(std::size_t size, std::align_val_t alignment) {
    void *memory = allocate(size, alignment);
    if (memory == nullptr) {
        throw std::bad_alloc();
    }
    return memory;
}

} // namespace

namespace gtest {

AllocationCounter::AllocationCounter() noexcept : start_(allocations), wasCounting_(counting) {
    counting = true;
}

AllocationCounter::~AllocationCounter() {
    counting = wasCounting_;
}

std::size_t
AllocationCounter::count() const noexcept {
    return allocations - start_;
}

} // namespace gtest

void *
operator new(std::size_t size) {
    return allocateOrThrow(size);
}

void *
operator new[](std::size_t size) {
    return allocateOrThrow(size);
}

void *
operator new(std::size_t size, std::align_val_t alignment) {
    return allocateOrThrow(size, alignment);
}

void *
operator new[](std::size_t size, std::align_val_t alignment) {
    return allocateOrThrow(size, alignment);
}

void *
operator new(std::size_t size, const std::nothrow_t &) noexcept {
    return allocate(size);
}

void *
operator new[](std::size_t size, const std::nothrow_t &) noexcept {
    return allocate(size);
}

void *
operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t &) noexcept {
    return allocate(size, alignment);
}

void *
operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t &) noexcept {
    return allocate(size, alignment);
}

void
operator delete(void *memory) noexcept {
    std::free(memory);
}

void
operator delete[](void *memory) noexcept {
    std::free(memory);
}

void
operator delete(void *memory, std::size_t) noexcept {
    std::free(memory);
}

void
operator delete[](void *memory, std::size_t) noexcept {
    std::free(memory);
}

void
operator delete(void *memory, std::align_val_t) noexcept {
    std::free(memory);
}

void
operator delete[](void *memory, std::align_val_t) noexcept {
    std::free(memory);
}

void
operator delete(void *memory, std::size_t, std::align_val_t) noexcept {
    std::free(memory);
}

void
operator delete[](void *memory, std::size_t, std::align_val_t) noexcept {
    std::free(memory);
}

void
operator delete(void *memory, const std::nothrow_t &) noexcept {
    std::free(memory);
}

void
operator delete[](void *memory, const std::nothrow_t &) noexcept {
    std::free(memory);
}

void
operator delete(void *memory, std::align_val_t, const std::nothrow_t &) noexcept {
    std::free(memory);
}

void
operator delete[](void *memory, std::align_val_t, const std::nothrow_t &) noexcept {
    std::free(memory);
}
