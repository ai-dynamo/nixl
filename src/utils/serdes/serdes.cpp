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
#include <cstring>
#include <stdexcept>

#include "serdes.h"

namespace nixl {
namespace {
    const std::string_view introString = "N1XL";

} // namespace

serializer::serializer(const size_t preAlloc) {
    buffer_.reserve(preAlloc);
    buffer_.append(introString);
}

void
serializer::addLongLength(size_t len) {
    uint8_t tmp[9];
    uint8_t used = 0;

    while (len > 0) {
        tmp[++used] = uint8_t(len);
        len >>= 8;
    };
    tmp[0] = uint8_t(-int8_t(used));
    append(tmp, used + 1);
}

void
serializer::addRawImpl(const std::string_view &str) {
    addRawImpl(str.data(), str.size());
}

void
serializer::addRawImpl(const void *buf, const size_t len) {
    if (len < 128) /* [[likely]] -- C++20 */ {
        append(char(len));
    } else {
        addLongLength(len);
    }
    append(buf, len);
}

void
serializer::append(const char c) {
    buffer_ += c;
}

void
serializer::append(const void *buf, const size_t len) {
    buffer_.append(reinterpret_cast<const char *>(buf), len);
}

deserializer::deserializer(const void *data, const size_t size)
    : buffer_(static_cast<const char *>(data), size) {
    if (buffer_.size() < introString.size()) /* [[unlikely]] -- C++20 */ {
        throw std::runtime_error("insufficient deserialization data for intro");
    }
    if (std::memcmp(buffer_.data(), introString.data(), introString.size()) !=
        0) /* [[unlikely]] -- C++20 */ {
        throw std::runtime_error("invalid deserialization intro data");
    }
    consume(introString.size());
}

nixl_status_t
deserializer::getString(const std::string_view &tag, std::string &result) {
    if (findTag(tag)) {
        result = getStringView();
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_NOT_FOUND;
}

nixl_status_t
deserializer::getBinary(const std::string_view &tag, void *data, size_t &size) {
    if (findTag(tag)) {
        const auto view = getStringView();
        if (view.size() <= size) {
            std::memcpy(data, view.data(), view.size());
            size = view.size();
            return NIXL_SUCCESS;
        }
        size = view.size();
        return NIXL_ERR_MISMATCH;
    }
    return NIXL_ERR_NOT_FOUND;
}

void
deserializer::require(const size_t amount) const {
    if (buffer_.size() < amount) /* [[unlikely]] -- C++20 */ {
        throw std::runtime_error("deserialization data insufficient");
    }
}

void
deserializer::consume(const size_t amount) noexcept {
    // assert(buffer_.size() >= amount);
    buffer_ = buffer_.substr(amount);
}

size_t
deserializer::getTagLength() {
    const int8_t first = getFirstByte();

    if (first >= 0) /* [[likely]] -- C++20 */ {
        require(first);
        return first;
    }
    if (first <= -9) /* [[unlikely]] -- C++20 */ {
        throw std::runtime_error("deserialization data mangled");
    }
    return getLongLength(-first);
}

size_t
deserializer::getLongLength(const size_t bytes) {
    require(bytes);
    size_t result = 0;
    std::memcpy(&result, buffer_.data(), bytes);
    consume(bytes);
    require(result);
    return result;
}

std::string_view
deserializer::getStringView() {
    const size_t length = getTagLength();
    const char *buffer = buffer_.data();
    consume(length);
    return std::string_view(buffer, length);
}

int8_t
deserializer::getFirstByte() {
    require(1);
    const auto result = *reinterpret_cast<const int8_t *>(buffer_.data());
    consume(1);
    return result;
}

bool
deserializer::findTag(const std::string_view &tag) {
    deserializer des(*this);

    while (!des.empty()) {
        if (des.getStringView() == tag) {
            buffer_ = des.buffer_;
            return true;
        }
        des.skipValue();
    }
    return false;
}

void
deserializer::skipValue() {
    const int8_t first = getFirstByte();

    if (first >= 0) {
        require(first);
        consume(first);
    } else if (first <= -9) {
        // Nothing to do.
    } else {
        const size_t length = getLongLength(-first);
        consume(length);
    }
}

nixl_status_t
deserializer::getIntegralImpl1(const int8_t first, void *result, const size_t size) {
    require(first);
    if (size_t(first) == size) {
        std::memcpy(result, buffer_.data(), size);
        consume(size);
        return NIXL_SUCCESS;
    }
    consume(size);
    return NIXL_ERR_MISMATCH;
}

nixl_status_t
deserializer::getIntegralImpl2(const int8_t first, void *result, const size_t size) {
    switch (first & 0xb0) {
    case 0x80:
        if (size != 1) {
            return NIXL_ERR_MISMATCH;
        }
        *reinterpret_cast<char *>(result) = char(first & 0x0f);
        return NIXL_SUCCESS;
    case 0x90:
    case 0xa0:
    case 0xb0:
        if (size != size_t(1 << ((first & 0x30) >> 4))) {
            return NIXL_ERR_MISMATCH;
        }
        std::memset(result, 0, size);
        *reinterpret_cast<char *>(result) = char(first & 0x0f);
        return NIXL_SUCCESS;
    /* [[unlikely]] -- C++20 */ default:
        return NIXL_ERR_UNKNOWN;
    }
}

nixl_status_t
deserializer::getIntegralImpl(void *result, const size_t size) {
    const int8_t first = getFirstByte();

    if (first >= 0) {
        return getIntegralImpl1(first, result, size);
    }
    if (first <= -9) {
        return getIntegralImpl2(first, result, size);
    }
    const size_t length = getLongLength(-first);
    consume(length);
    return NIXL_ERR_MISMATCH;
}

// BEGIN COMPATIBILITY CLASS

namespace internal {
    namespace {
        [[nodiscard]] constexpr size_t
        compressedSize(const int8_t first, const size_t fallback) noexcept {
            switch (first & 0xb0) {
            case 0x80:
                return 1;
            case 0x90:
                return 2;
            case 0xa0:
                return 4;
            case 0xb0:
                return 8;
            default:
                return fallback;
            }
        }

    } // namespace

    ssize_t
    compatibility::getBufLenCompat(compatibility &des) {
        const int8_t first = des.getFirstByte();

        if (first >= 0) {
            return first;
        }
        if (first <= -9) {
            return compressedSize(first, -1);
        }
        return des.getLongLength(-first);
    }

    ssize_t
    compatibility::getBufLenCompat(const std::string_view &tag) {
        compatibility des(*this);

        while (!des.empty()) {
            const compatibility tmp(des.buffer_);
            if (des.getStringView() == tag) {
                const auto result = getBufLenCompat(des);
                buffer_ = tmp.rawBuffer();
                return result;
            }
        }
        return -1;
    }

    // Some client code does not call serdes::getBufLen() before
    // serdes::getBuf() wherefore getBufCompat() can not assume
    // a preceding call to getBufLenCompat() with the same tag.

    nixl_status_t
    compatibility::getBufCompat(void *buf, const ssize_t size) {
        const int8_t first = getFirstByte();

        if (first >= 0) {
            return getIntegralImpl1(first, buf, size);
        }
        if (first <= -9) {
            return getIntegralImpl2(first, buf, size);
        }
        const size_t length = getLongLength(-first);
        if (length == size_t(size)) {
            std::memcpy(buf, buffer_.data(), size);
            consume(length);
            return NIXL_SUCCESS;
        }
        consume(length);
        return NIXL_ERR_MISMATCH;
    }

    nixl_status_t
    compatibility::getBufCompat(const std::string_view &tag, void *buf, const ssize_t size) {
        compatibility des(*this);

        while (!des.empty()) {
            if (des.getStringView() == tag) {
                buffer_ = des.buffer_;
                return getBufCompat(buf, size);
            }
        }
        return NIXL_ERR_NOT_FOUND;
    }

    std::string
    compatibility::getStrCompat() {
        const int8_t first = getFirstByte();

        if (first >= 0) {
            require(first);
            const std::string result(buffer_.data(), size_t(first));
            consume(first);
            return result;
        }
        if (first <= -9) {
            char tmp[8];
            std::memset(tmp, 0, sizeof(tmp));
            tmp[0] = char(first & 0x0f);
            const size_t size = compressedSize(first, 0);
            return std::string(tmp, size);
        }
        const size_t size = getLongLength(-first);
        const std::string result(buffer_.data(), size);
        consume(size);
        return result;
    }

    std::string
    compatibility::getStrCompat(const std::string_view &tag) {
        compatibility des(*this);

        while (!des.empty()) {
            if (des.getStringView() == tag) {
                buffer_ = des.buffer_;
                return getStrCompat();
            }
        }
        return "";
    }

    string_holder::string_holder(std::string &&data) noexcept : string_(std::move(data)) {}

    deserializer::deserializer(std::string &&data)
        : string_holder(std::move(data)),
          compatibility(string_.data(), string_.size()) {}

    deserializer::deserializer(deserializer &&other) noexcept
        : deserializer(std::move(other.string_), other.buffer_.data() - other.string_.data()) {}

    deserializer::deserializer(const deserializer &other)
        : deserializer(std::string(other.string_), other.buffer_.data() - other.string_.data()) {}

    deserializer &
    deserializer::operator=(deserializer &&other) noexcept {
        if (this != &other) {
            assign(std::move(other.string_), other.buffer_.data() - other.string_.data());
        }
        return *this;
    }

    deserializer &
    deserializer::operator=(const deserializer &other) {
        if (this != &other) {
            assign(std::string(other.string_), other.buffer_.data() - other.string_.data());
        }
        return *this;
    }

    deserializer::deserializer(std::string &&data, const size_t offset) noexcept
        : string_holder(std::move(data)) {
        buffer_ = std::string_view(string_.data() + offset, string_.size() - offset);
    }

    void
    deserializer::assign(std::string &&data, const size_t offset) noexcept {
        string_ = std::move(data);
        buffer_ = std::string_view(string_.data() + offset, string_.size() - offset);
    }

} // namespace internal

namespace {
    template<typename Int>
    [[nodiscard]] Int
    copyFromRaw(const void *buf) {
        Int result;
        std::memcpy(&result, buf, sizeof(result));
        return result;
    }

} // namespace

} // namespace nixl

nixl::serializer &
nixlSerDes::serializer() {
    return std::get<0>(impl_);
}

const nixl::serializer &
nixlSerDes::serializer() const {
    return std::get<0>(impl_);
}

nixl::internal::deserializer &
nixlSerDes::deserializer() {
    return std::get<1>(impl_);
}

const nixl::internal::deserializer &
nixlSerDes::deserializer() const {
    return std::get<1>(impl_);
}

nixl::internal::deserializer &
nixlSerDes::mutable_deserializer() const {
    return const_cast<nixl::internal::deserializer &>(std::get<1>(impl_));
}

nixlSerDes::nixlSerDes() : impl_(std::in_place_type_t<nixl::serializer>()) {}

nixl_status_t
nixlSerDes::addStr(const std::string &tag, const std::string &str) {
    serializer().addString(tag, str);
    return NIXL_SUCCESS;
}

std::string
nixlSerDes::getStr(const std::string &tag) {
    // Can't use nixl::deserializer::getString() because it assumes
    // that the value is not a compressed integer.
    return deserializer().getStrCompat(tag);
}

nixl_status_t
nixlSerDes::addBuf(const std::string &tag, const void *buf, ssize_t len) {
    // This function is (also) used for integers, bools and enums.
    switch (len) {
    case 1:
        serializer().addIntegral(tag, *reinterpret_cast<const uint8_t *>(buf));
        break;
    case 2:
        serializer().addIntegral(tag, nixl::copyFromRaw<uint16_t>(buf));
        break;
    case 4:
        serializer().addIntegral(tag, nixl::copyFromRaw<uint32_t>(buf));
        break;
    case 8:
        serializer().addIntegral(tag, nixl::copyFromRaw<uint64_t>(buf));
        break;
    default:
        serializer().addBinary(tag, buf, len);
        break;
    }
    return NIXL_SUCCESS;
}

ssize_t
nixlSerDes::getBufLen(const std::string &tag) const {
    return mutable_deserializer().getBufLenCompat(tag);
}

nixl_status_t
nixlSerDes::getBuf(const std::string &tag, void *buf, const ssize_t len) {
    return deserializer().getBufCompat(tag, buf, len);
}

std::string
nixlSerDes::exportStr() const {
    return serializer().getBuffer();
}

nixl_status_t
nixlSerDes::importImpl(std::string &&data) {
    try {
        impl_.emplace<nixl::internal::deserializer>(std::move(data));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &) {
        // TODO: Print warning?
        return NIXL_ERR_MISMATCH;
    }
}

nixl_status_t
nixlSerDes::importStr(const std::string &data) {
    return importImpl(std::string(data));
}

std::string
nixlSerDes::_bytesToString(const void *buf, ssize_t size) {
    return std::string(reinterpret_cast<const char *>(buf), size);
}

void
nixlSerDes::_stringToBytes(void *fill_buf, const std::string &s, ssize_t size) {
    s.copy(reinterpret_cast<char *>(fill_buf), size);
}
