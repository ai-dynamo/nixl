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

#ifndef NIXL_SRC_UTILS_COMMON_BYTE_BUFFER_H
#define NIXL_SRC_UTILS_COMMON_BYTE_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <type_traits>
#include <vector>

#include "nixl_log.h"

namespace nixl {

template<typename R>
concept byte_range_t =
    std::ranges::input_range<R> && std::ranges::sized_range<R> && std::ranges::common_range<R> &&
    std::is_convertible_v<std::ranges::range_value_t<R>, std::uint8_t> &&
    sizeof(std::ranges::range_value_t<R>) == sizeof(std::uint8_t);

/**
 * @class inputByteBuffer
 * @brief A simple input buffer that can be used to store and manipulate binary data.
 *        Arithmetic values are stored in native byte order, and byte ranges are
 *        stored as their length followed by the bytes.
 *
 *        The buffer is not thread-safe.
 */
class dynamicSizedBuffer {
public:
    /**
     * @brief Construct a new empty byte buffer.
     */
    dynamicSizedBuffer() = default;

    /**
     * @brief Move constructor.
     * @param other The other byte buffer to move from.
     */
    dynamicSizedBuffer(dynamicSizedBuffer &&other) noexcept = default;

    /**
     * @brief Move assignment operator.
     * @param other The other byte buffer to move from.
     * @return this.
     */
    dynamicSizedBuffer &
    operator=(dynamicSizedBuffer &&other) noexcept = default;

    // Delete copy constructor and assignment operator.
    dynamicSizedBuffer(const dynamicSizedBuffer &) = delete;
    dynamicSizedBuffer &
    operator=(const dynamicSizedBuffer &) = delete;

    ~dynamicSizedBuffer() = default;

    /**
     * @brief Put an arithmetic value into the buffer.
     * @param value The value to put into the buffer.
     * @return this.
     */
    template<typename T>
        requires std::is_arithmetic_v<T>
    void
    put(T value) {
        const auto *begin = reinterpret_cast<const std::uint8_t *>(&value);
        data_.insert(data_.end(), begin, begin + sizeof(T));
    }

    /**
     * @brief Put a byte sequence into the buffer.
     * @param value The byte sequence to put into the buffer. The given sequence
     *        must NOT come from this buffer.
     * @return this.
     */
    void
    put(byte_range_t auto &&value) {
        put(static_cast<std::uint64_t>(std::ranges::size(value)));
        data_.insert(data_.end(), std::ranges::cbegin(value), std::ranges::cend(value));
    }

    /**
     * @brief Put multiple values into the buffer.
     * @param value The first value to put into the buffer.
     * @param args The rest of the values to put into the buffer.
     * @return this.
     */
    template<typename... Args>
    void
    putValues(Args &&...args) {
        data_.reserve(data_.size() + (... + getSize(args)));
        (put(args), ...);
    }

    /**
     * @brief Get a const pointer to the internal data.
     * @return A const pointer to the internal data.
     */
    [[nodiscard]] const std::uint8_t *
    data() const noexcept {
        return data_.data();
    }

    /**
     * @brief Get the size of the buffer.
     * @return The size of the buffer.
     */
    [[nodiscard]] std::size_t
    size() const noexcept {
        return data_.size();
    }

private:
    template<typename T>
        requires std::is_arithmetic_v<T>
    [[nodiscard]] std::size_t
    getSize(T value) const noexcept {
        return sizeof(T);
    }

    [[nodiscard]] std::size_t
    getSize(byte_range_t auto &&value) const noexcept {
        return sizeof(std::uint64_t) + std::ranges::size(value);
    }

    std::vector<std::uint8_t> data_;
};

class fixedCapacityBuffer {
public:
    /**
     * @brief Construct a new output byte buffer with the given capacity.
     * @param capacity The capacity of the buffer.
     */
    explicit fixedCapacityBuffer(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ > 0) {
            data_ = std::make_unique_for_overwrite<std::uint8_t[]>(capacity_);
        }
    }

    /**
     * @brief Move constructor.
     * @param other The other byte buffer to move from.
     */
    fixedCapacityBuffer(fixedCapacityBuffer &&other) noexcept = default;

    /**
     * @brief Move assignment operator.
     * @param other The other byte buffer to move from.
     * @return this.
     */
    fixedCapacityBuffer &
    operator=(fixedCapacityBuffer &&other) noexcept = default;

    // Delete copy constructor and assignment operator.
    fixedCapacityBuffer(const fixedCapacityBuffer &) = delete;
    fixedCapacityBuffer &
    operator=(const fixedCapacityBuffer &) = delete;

    ~fixedCapacityBuffer() = default;

    /**
     * @brief Get a mutable pointer to the internal data.
     * @return A mutable pointer to the internal data.
     */
    [[nodiscard]] std::uint8_t *
    data() noexcept {
        return data_.get();
    }

    /**
     * @brief Get the capacity of the buffer.
     * @return The capacity of the buffer.
     */
    [[nodiscard]] std::size_t
    capacity() const noexcept {
        return capacity_;
    }

    /**
     * @brief Set the size of the buffer.
     * @param size The size of the buffer.
     */
    void
    setSize(std::size_t size) noexcept {
        NIXL_ASSERT(size <= capacity_);
        size_ = size;
    }

    /**
     * @brief Get the size of the buffer.
     * @return The size of the buffer.
     */
    [[nodiscard]] std::size_t
    size() const noexcept {
        return size_;
    }

private:
    std::unique_ptr<std::uint8_t[]> data_;
    std::size_t capacity_;
    std::size_t size_ = 0;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_BYTE_BUFFER_H
