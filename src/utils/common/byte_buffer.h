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
#include <iterator>
#include <type_traits>
#include <vector>

namespace nixl {

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

    // Move constructor and assignment operator.
    byteBuffer(byteBuffer &&) noexcept = default;
    byteBuffer &
    operator=(byteBuffer &&) noexcept = default;

    // Copy constructor and assignment operator.
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
        static_assert(std::is_same_v<std::uint8_t, unsigned char>);
        const auto *begin = reinterpret_cast<const std::uint8_t *>(&value);
        data_.insert(data_.end(), begin, begin + sizeof(T));
        return *this;
    }

    /**
     * @brief Put a byte sequence into the buffer.
     * @param value The byte sequence to put into the buffer.
     * @return this.
     */
    template<typename T>
        requires(sizeof(typename std::decay_t<T>::value_type) == sizeof(std::uint8_t) &&
                 std::input_iterator<typename std::decay_t<T>::iterator>)
    byteBuffer &
    put(T &&value) {
        put(static_cast<std::uint64_t>(value.size()));
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
        requires(!std::is_same_v<std::decay_t<Args>, byteBuffer> && ...)
    byteBuffer &
    putValues(Args &&...args) {
        data_.reserve(getSize(args...));
        putValuesInternal(args...);
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

    template<typename T>
        requires(sizeof(typename std::decay_t<T>::value_type) == sizeof(std::uint8_t) &&
                 std::input_iterator<typename std::decay_t<T>::iterator>)
    [[nodiscard]] std::size_t
    getSize(T &&value) const noexcept {
        return sizeof(std::uint64_t) + value.size();
    }

    template<typename T, typename... Args>
    [[nodiscard]] std::size_t
    getSize(T &&value, Args &&...args) const noexcept {
        return getSize(value) + getSize(std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    byteBuffer &
    putValuesInternal(T &&value, Args &&...args) {
        put(value);
        if constexpr (sizeof...(args) > 0) {
            putValuesInternal(args...);
        }
        return *this;
    }

    std::vector<std::uint8_t> data_;
};

[[nodiscard]] inline static nixl::byteBuffer
make_receive_buffer(std::size_t size) {
    nixl::byteBuffer buf;
    buf.resize(size);
    return buf;
}

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_BYTE_BUFFER_H
