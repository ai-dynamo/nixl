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
#ifndef NIXL_SRC_UTILS_SERDES_SERDES_H
#define NIXL_SRC_UTILS_SERDES_SERDES_H

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "nixl_types.h"

namespace nixl {
namespace internal {
    template<typename...>
    inline constexpr bool dependentFalse = false; // TODO: Move somewhere else?

} // namespace internal

class serializer {
public:
    explicit serializer(const size_t preAlloc = 4000);

    void
    addString(const std::string_view &tag, const std::string_view &str) {
        addRawImpl(tag);
        addRawImpl(str);
    }

    void
    addBinary(const std::string_view &tag, const void *buf, size_t len) {
        addRawImpl(tag);
        addRawImpl(buf, len);
    }

    template<typename Int>
    void
    addIntegral(const std::string_view &tag, const Int value) {
        static_assert(std::is_integral_v<Int> || std::is_enum_v<Int>);
        addRawImpl(tag);
        addIntegralImpl(value);
    }

    [[nodiscard]] std::string
    getBuffer() && noexcept {
        return std::move(buffer_);
    }

    [[nodiscard]] const std::string &
    getBuffer() const & noexcept {
        return buffer_;
    }

protected:
    std::string buffer_;

    void
    addLongLength(size_t len);
    void
    addRawImpl(const std::string_view &str);
    void
    addRawImpl(const void *buf, size_t len);

    template<typename Int>
    void
    addIntegralImpl(const Int value) {
        static_assert(std::is_integral_v<Int> || std::is_enum_v<Int>);
        if (value < 16) {
            if constexpr (sizeof(Int) == 1) {
                append(char(0x80) | char(value));
            } else if constexpr (sizeof(Int) == 2) {
                append(char(0x90) | char(value));
            } else if constexpr (sizeof(Int) == 4) {
                append(char(0xa0) | char(value));
            } else if constexpr (sizeof(Int) == 8) {
                append(char(0xb0) | char(value));
            } else {
                static_assert(internal::dependentFalse<Int>);
            }
        } else {
            append(char(sizeof(value)));
            append(&value, sizeof(value));
        }
    }

    void
    append(char);
    void
    append(const void *buf, size_t len);

    // template<size_t N>
    // void append(const char(&literal)[N])
    // {
    //     append(literal, N - 1);
    // }
};

class deserializer {
public:
    deserializer(const void *data,
                 size_t size); // Does NOT copy the data; Throws on introString mismatch.

    [[nodiscard]] bool
    empty() const noexcept {
        return buffer_.empty();
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return buffer_.size();
    }

    [[nodiscard]] const std::string_view &
    rawBuffer() const noexcept {
        return buffer_;
    }

    // MUST (only) be used for values serialized with addString (also addBinary?)
    [[nodiscard]] nixl_status_t
    getString(const std::string_view &tag, std::string &result);

    // MUST (only) be used for values serialized with addBinary (also addString?)
    // Before the call size must be the size of the buffer pointed to by data.
    // After the call size is the size of the found data for NIXL_ERR_MISMATCH and
    // NIXL_SUCCESS, in all other cases size is unchanged (mismatch signals that
    // the supplied size was not sufficient to copy out the found data).
    [[nodiscard]] nixl_status_t
    getBinary(const std::string_view &tag, void *data, size_t &size);

    // MUST (only) be used for values serialized with addIntegral with the same type.
    template<typename Int>
    [[nodiscard]] nixl_status_t
    getIntegral(const std::string_view &tag, Int &result) {
        static_assert(std::is_integral_v<Int> || std::is_enum_v<Int>);

        if (findTag(tag)) {
            return getIntegralImpl(&result, sizeof(result));
        }
        return NIXL_ERR_NOT_FOUND;
    }

protected:
    deserializer() noexcept = default;

    std::string_view buffer_;

    void
    require(size_t) const;
    void
    consume(size_t) noexcept;
    [[nodiscard]] size_t
    getTagLength();
    [[nodiscard]] size_t
    getLongLength(size_t);
    [[nodiscard]] int8_t
    getFirstByte();
    [[nodiscard]] std::string_view
    getStringView();
    [[nodiscard]] bool
    findTag(const std::string_view &tag);
    void
    skipValue();
    [[nodiscard]] nixl_status_t
    getIntegralImpl(void *result, size_t length);
    [[nodiscard]] nixl_status_t
    getIntegralImpl1(const int8_t first, void *result, size_t length);
    [[nodiscard]] nixl_status_t
    getIntegralImpl2(const int8_t first, void *result, size_t length);
};

// BEGIN COMPATIBILITY CLASS

namespace internal {
    class compatibility : public nixl::deserializer {
    public:
        using nixl::deserializer::deserializer;
        using nixl::deserializer::getFirstByte;
        using nixl::deserializer::getLongLength;

        explicit compatibility(const std::string_view &pbuf) noexcept {
            buffer_ = pbuf;
        }

        [[nodiscard]] static ssize_t
        getBufLenCompat(compatibility &);
        [[nodiscard]] ssize_t
        getBufLenCompat(const std::string_view &tag);
        [[nodiscard]] nixl_status_t
        getBufCompat(void *buf, ssize_t size);
        [[nodiscard]] nixl_status_t
        getBufCompat(const std::string_view &tag, void *buf, const ssize_t size);
        [[nodiscard]] std::string
        getStrCompat();
        [[nodiscard]] std::string
        getStrCompat(const std::string_view &tag);
    };

    class string_holder {
    public:
        explicit string_holder(std::string &&data) noexcept;

    protected:
        std::string string_;
    };

    class deserializer : private string_holder, public compatibility {
    public:
        explicit deserializer(std::string &&data);

        deserializer(deserializer &&) noexcept;
        deserializer(const deserializer &);

        deserializer &
        operator=(deserializer &&) noexcept;
        deserializer &
        operator=(const deserializer &);

    private:
        deserializer(std::string &&data, size_t offset) noexcept;

        void
        assign(std::string &&data, size_t offset) noexcept;
    };

} // namespace internal

} // namespace nixl

class nixlSerDes {
private:
    std::variant<nixl::serializer, nixl::internal::deserializer> impl_;

    [[nodiscard]] nixl::serializer &
    serializer();
    [[nodiscard]] const nixl::serializer &
    serializer() const;

    [[nodiscard]] nixl::internal::deserializer &
    deserializer();
    [[nodiscard]] const nixl::internal::deserializer &
    deserializer() const;
    [[nodiscard]] nixl::internal::deserializer &
    mutable_deserializer() const;

    [[nodiscard]] nixl_status_t
    importImpl(std::string &&sdbuf); // Converts to deserializer.

public:
    // Default construct as serializer (as before).
    // Throws bad_alloc on out-of-memory (as before).
    nixlSerDes();

    // Add c'tor that takes a string and does an implicit importStr?

    // Serialization of strings as before.
    nixl_status_t
    addStr(const std::string &tag,
           const std::string
               &str); // Always returns NIXL_SUCCESS, throws on out-of-memory (as before).
    std::string
    getStr(const std::string &tag); // Returns empty string if tag not found.

    // Serialization of arbitrary data (as before).
    // Also (mostly?) used for bools, enums and integers.
    nixl_status_t
    addBuf(const std::string &tag,
           const void *buf,
           ssize_t len); // Always returns NIXL_SUCCESS, throws on out-of-memory (as before).
    ssize_t
    getBufLen(const std::string &tag) const; // Returns -1 if tag not found.
    nixl_status_t
    getBuf(const std::string &tag,
           void *buf,
           ssize_t len); // Returns NIXL_ERR_MISMATCH if tag not found.

    // Buffer management functions (as before).
    std::string
    exportStr() const; // Can we return by const-reference?
    nixl_status_t
    importStr(const std::string &sdbuf); // Converts to deserializer.

    // Static functions that copy data between strings and buffers (as before).
    static std::string
    _bytesToString(const void *buf, ssize_t size);
    static void
    _stringToBytes(void *fill_buf, const std::string &s, ssize_t size);
};

#endif
