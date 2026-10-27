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
#include <ranges>
#include <type_traits>
#include <vector>

namespace nixl {

template<typename R>
concept byte_range_t = std::ranges::input_range<R> && std::ranges::sized_range<R> &&
    std::is_convertible_v<std::ranges::range_value_t<R>, std::uint8_t> &&
    sizeof(std::ranges::range_value_t<R>) == sizeof(std::uint8_t);

/**
 * @class byteBuffer
 * @brief A simple byte buffer that can be used to store and manipulate binary data.
 *        Arithmetic values are stored in native byte order, and byte sequences are
 *        stored as their length followed by the bytes.
 *
 *        The buffer is not thread-safe.
 */
class byteBuffer {
public:
    /**
     * @brief Construct a new empty byte buffer.
     */
    byteBuffer() = default;

    /**
     * @brief Move constructor.
     * @param other The other byte buffer to move from.
     */
    byteBuffer(byteBuffer &&other) noexcept = default;

    /**
     * @brief Move assignment operator.
     * @param other The other byte buffer to move from.
     * @return this.
     */
    byteBuffer &
    operator=(byteBuffer &&other) noexcept = default;

    // Delete copy constructor and assignment operator.
    byteBuffer(const byteBuffer &) = delete;
    byteBuffer &
    operator=(const byteBuffer &) = delete;

    ~byteBuffer() = default;

    /**
     * @brief Reserve space for the given number of bytes. This is used to
     *        set the size of the receive buffer.
     * @param size The number of bytes to reserve.
     */
    void
    resize(std::size_t size) {
        data_.resize(size);
    }

    /**
     * @brief Put an arithmetic value into the buffer.
     * @param value The value to put into the buffer.
     * @return this.
     */
    template<typename T>
        requires std::is_arithmetic_v<T>
    byteBuffer &
    put(T value) {
        const auto *begin = reinterpret_cast<const std::uint8_t *>(&value);
        data_.insert(data_.end(), begin, begin + sizeof(T));
        return *this;
    }

    /**
     * @brief Put a byte sequence into the buffer.
     * @param value The byte sequence to put into the buffer. The given sequence
     *        must NOT come from this buffer.
     * @return this.
     */
    byteBuffer &
    put(byte_range_t auto &&value) {
        put(static_cast<std::uint64_t>(std::ranges::size(value)));
        data_.insert(data_.end(), value.begin(), value.end());
        return *this;
    }

    /**
     * @brief Put multiple values into the buffer.
     * @param value The first value to put into the buffer.
     * @param args The rest of the values to put into the buffer.
     * @return this.
     */
    template<typename... Args>
    byteBuffer &
    putValues(Args &&...args) {
        data_.reserve(data_.size() + (... + getSize(args)));
        (put(args), ...);
        return *this;
    }

    /**
     * @brief Get a const reference to the internal data.
     * @return A const reference to the internal data.
     */
    std::vector<std::uint8_t> const &
    getRef() const noexcept {
        return data_;
    }

    /**
     * @brief Get a mutable reference to the internal data.
     * @return A reference to the internal data.
     */
    std::vector<std::uint8_t> &
    getRef() noexcept {
        return data_;
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

/**
 * @brief Make a receive buffer of the given size.
 * @param size The size of the receive buffer.
 * @return A receive buffer of the given size.
 */
[[nodiscard]] inline static nixl::byteBuffer
makeReceiveBuffer(std::size_t size) {
    nixl::byteBuffer buf;
    buf.resize(size);
    return buf;
}

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_BYTE_BUFFER_H
