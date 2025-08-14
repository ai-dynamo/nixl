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
    inline constexpr size_t varLenBias = 127;

    const std::string_view header1 = "N1XL";

    template<typename...> inline constexpr bool dependentFalse = false;

    template<typename Int>
    [[nodiscard]] Int
    reconstruct(const void *buf) noexcept {
        Int result;
        std::memcpy(&result, buf, sizeof(result));
        return result;
    }

    void
    addRaw(std::string &buffer, const void *data, const size_t size) {
        buffer.append(static_cast<const char *>(data), size);
    }

    template<typename Int>
    void
    addRawInt(std::string &buffer, const Int value) {
        buffer += char(sizeof(value));
        addRaw(buffer, &value, sizeof(value));
    }

    // Can we use __builtin_clzl() to optimize this?

    void
    addVarLen(std::string &buffer, size_t size) {
        // assert(size > 127);

        uint8_t tmp[9];
        uint8_t used = 0;

        size -= varLenBias;

        while (size > 0) {
            tmp[++used] = uint8_t(size);
            size >>= 8;
        }
        tmp[0] = (used | char(0xC0));
        addRaw(buffer, tmp, 1 + used);
    }

    void
    addLen(std::string &buffer, const size_t size) {
        if (size < 128) {
            buffer += char(size);
        } else {
            addVarLen(buffer, size);
        }
    }

    void
    addStr(std::string &buffer, const std::string_view &string) {
        addLen(buffer, string.size());
        addRaw(buffer, string.data(), string.size());
    }

    template<typename Int>
    void
    addImmInt(std::string &buffer, const Int value) {
        static_assert(std::is_unsigned_v<Int>);
        // assert(value < 16);

        if constexpr (sizeof(Int) == 1) {
            buffer += (char(0x80) | char(value));
        } else if constexpr (sizeof(Int) == 2) {
            buffer += (char(0x90) | char(value));
        } else if constexpr (sizeof(Int) == 4) {
            buffer += (char(0xA0) | char(value));
        } else if constexpr (sizeof(Int) == 8) {
            buffer += (char(0xB0) | char(value));
        } else {
            static_assert(dependentFalse<Int>, "Unsupported integer/enum size!");
        }
    }

    template<typename Int>
    void
    addVarInt(std::string &buffer, const Int value) {
        static_assert(sizeof(Int) > 1);
        static_assert(std::is_unsigned_v<Int>);
        // assert(value > 0);

        uint8_t tmp[sizeof(value) + 1];
        uint8_t used = 0;

        for (Int v = value; v > 0; v >>= 8) {
            tmp[++used] = uint8_t(v);
        }
        if (used == sizeof(Int)) {
            addRawInt(buffer, value);
        } else if constexpr (sizeof(Int) == 2) {
            tmp[0] = 0xD0; // used == 1 -> D0
            addRaw(buffer, tmp, 1 + used);
        } else if constexpr (sizeof(Int) == 4) {
            tmp[0] = 0xD0 + used; // used == 1..3 -> D1..D3
            addRaw(buffer, tmp, 1 + used);
        } else if constexpr (sizeof(Int) == 8) {
            tmp[0] = 0xD0 + used + 3; // used == 1..7 -> D4..DA
            addRaw(buffer, tmp, 1 + used);
        } else {
            static_assert(dependentFalse<Int>, "Unsupported integer/enum size!");
        }
    }

    template<typename Int>
    void
    addInt(std::string &buffer, const Int value) {
        static_assert(std::is_unsigned_v<Int>);

        if (value < 16) {
            addImmInt(buffer, value);
        } else if constexpr (sizeof(Int) == 1) {
            buffer += char(1);
            buffer += char(value);
        } else {
            addVarInt(buffer, value);
        }
    }

    [[nodiscard]] std::pair<uint8_t, uint8_t>
    varIntSizes(const uint8_t first) {
        switch (first) {
        case 0xD0:
            return {1, 2};
        case 0xD1:
            return {1, 4};
        case 0xD2:
            return {2, 4};
        case 0xD3:
            return {3, 4};
        case 0xD4:
            return {1, 8};
        case 0xD5:
            return {2, 8};
        case 0xD6:
            return {3, 8};
        case 0xD7:
            return {4, 8};
        case 0xD8:
            return {5, 8};
        case 0xD9:
            return {6, 8};
        case 0xDA:
            return {7, 8};
        default:
            throw std::runtime_error("invalid deserialization type");
        }
    }

} // namespace

} // namespace nixl

void
nixlSerDes::consume(const size_t size) noexcept {
    // assert(offset_ + size <= buffer_.size());
    offset_ += size;
}

void
nixlSerDes::require(const size_t size) {
    if (offset_ + size > buffer_.size()) {
        throw std::runtime_error("deserialization data incomplete");
    }
}

nixlSerDes::nixlSerDes() : buffer_(nixl::header1), offset_(buffer_.size()) {}

nixlSerDes::nixlSerDes(const size_t reserve) {
    buffer_.reserve(reserve);
    buffer_.append(nixl::header1);
    offset_ = buffer_.size();
}

uint8_t
nixlSerDes::getFirst() {
    require(1);
    const uint8_t first = buffer_[offset_];
    consume(1);
    return first;
}

size_t
nixlSerDes::getVarLen(const uint8_t first) {
    const size_t bytes = first & 0x0F;
    if (bytes > 6) {
        throw std::runtime_error("serialized size exceeds 2^48");
    }
    require(bytes);
    size_t result = 0;
    std::memcpy(&result, buffer_.data() + offset_, bytes);
    consume(bytes);
    require(result);
    return result + nixl::varLenBias;
}

size_t
nixlSerDes::getTagLen() {
    const uint8_t first = getFirst();

    if (first < 128) {
        require(first);
        return first;
    }
    if ((first & 0xF0) == 0xC0) {
        return getVarLen(first);
    }
    throw std::runtime_error("mangled serialized data");
}

std::string_view
nixlSerDes::getTag() {
    const size_t size = getTagLen();
    const char *data = buffer_.data() + offset_;
    consume(size);
    return std::string_view(data, size);
}

void
nixlSerDes::skipValue() {
    const uint8_t first = getFirst();

    if (first < 128) {
        require(first);
        consume(first);
        return;
    }
    switch (first & 0xF0) {
    case 0x80:
    case 0x90:
    case 0xA0:
    case 0xB0:
        return;
    case 0xC0:
        consume(getVarLen(first));
        return;
    case 0xD0:
        consume(nixl::varIntSizes(first).first);
        return;
    default:
        throw std::runtime_error("corrupt serialized data");
    }
}

size_t
nixlSerDes::findTag(const std::string_view &tag) {
    const size_t rewind = offset_;

    try {
        while (offset_ < buffer_.size()) {
            const size_t result = offset_;
            if (getTag() == tag) {
                return result;
            }
            skipValue();
        }
    }
    catch (const std::exception &) {
        // TODO -- What?
    }
    offset_ = rewind;
    return 0;
}

bool
nixlSerDes::syntaxCheck() noexcept {
    try {
        offset_ = nixl::header1.size();
        while (offset_ < buffer_.size()) {
            (void)getTag();
            skipValue();
        }
        offset_ = nixl::header1.size();
        return true;
    }
    catch (const std::exception &) {
    }
    return false;
}

nixl_status_t
nixlSerDes::addStr(const std::string &tag, const std::string &str) {
    nixl::addStr(buffer_, tag);
    nixl::addStr(buffer_, str);
    return NIXL_SUCCESS;
}

std::string
nixlSerDes::getStr(const std::string &tag) {
    if (const size_t rewind = findTag(tag)) {
        try {
            return std::string(getTag());
        }
        catch (const std::exception &) {
            offset_ = rewind;
        }
    }
    return "";
}

// This function is (also) used for integers, bools and enums.
nixl_status_t
nixlSerDes::addBuf(const std::string &tag, const void *data, const ssize_t size) {
    nixl::addStr(buffer_, tag);
    switch (size) {
    case 1:
        nixl::addInt(buffer_, *reinterpret_cast<const uint8_t *>(data));
        break;
    case 2:
        nixl::addInt(buffer_, nixl::reconstruct<uint16_t>(data));
        break;
    case 4:
        nixl::addInt(buffer_, nixl::reconstruct<uint32_t>(data));
        break;
    case 8:
        nixl::addInt(buffer_, nixl::reconstruct<uint64_t>(data));
        break;
    default:
        nixl::addLen(buffer_, size);
        nixl::addRaw(buffer_, data, size);
        break;
    }
    return NIXL_SUCCESS;
}

ssize_t
nixlSerDes::getBufLenImpl() {
    const uint8_t first = getFirst();

    if (first < 128) {
        return ssize_t(first);
    }
    switch (first & 0xF0) {
    case 0x80:
        return 1;
    case 0x90:
        return 2;
    case 0xA0:
        return 4;
    case 0xB0:
        return 8;
    case 0xC0:
        return getVarLen(first);
    case 0xD0:
        return nixl::varIntSizes(first).second;
    default:
        throw std::runtime_error("corrupt serialized data");
    }
}

ssize_t
nixlSerDes::getBufLen(const std::string &tag) {
    if (const size_t rewind = findTag(tag)) {
        try {
            const ssize_t result = getBufLenImpl();
            offset_ = rewind;
            return result;
        }
        catch (const std::exception &) {
            offset_ = rewind;
        }
    }
    return -1;
}

nixl_status_t
nixlSerDes::getBufImpl1(const size_t size, void *buf, const ssize_t len) {
    if (size_t(len) != size) {
        return NIXL_ERR_MISMATCH;
    }
    std::memcpy(buf, buffer_.data() + offset_, len);
    consume(len);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlSerDes::getBufImpl2(const uint8_t first, const uint8_t size, void *buf, const ssize_t len) {
    if (len != size) {
        return NIXL_ERR_MISMATCH;
    }
    if (len > 1) {
        std::memset(static_cast<char *>(buf) + 1, 0, len - 1);
    }
    *reinterpret_cast<uint8_t *>(buf) = first & 0x0F;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlSerDes::getBufImpl3(const uint8_t first, void *buf, const ssize_t len) {
    const auto pair = nixl::varIntSizes(first);
    if (pair.second != len) {
        return NIXL_ERR_MISMATCH;
    }
    std::memcpy(buf, buffer_.data() + offset_, pair.first);
    std::memset(static_cast<char *>(buf) + pair.first, 0, pair.second - pair.first);
    consume(pair.first);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlSerDes::getBufImpl(void *buf, const ssize_t len) {
    const uint8_t first = getFirst();

    if (first < 128) {
        require(first);
        return getBufImpl1(size_t(first), buf, len);
    }
    switch (first & 0xF0) {
    case 0x80:
        return getBufImpl2(first, 1, buf, len);
    case 0x90:
        return getBufImpl2(first, 2, buf, len);
    case 0xA0:
        return getBufImpl2(first, 4, buf, len);
    case 0xB0:
        return getBufImpl2(first, 8, buf, len);
    case 0xC0:
        return getBufImpl1(getVarLen(first), buf, len);
    case 0xD0:
        return getBufImpl3(first, buf, len);
    default:
        return NIXL_ERR_UNKNOWN;
    }
}

// Some client code does not call getBufLen() before getBuf()
// wherefore getBuf() has to search for the tag (again).

nixl_status_t
nixlSerDes::getBuf(const std::string &tag, void *buf, const ssize_t len) {
    if (const size_t rewind = findTag(tag)) {
        try {
            return getBufImpl(buf, len);
        }
        catch (const std::exception &) {
            offset_ = rewind;
        }
        return NIXL_ERR_UNKNOWN;
    }
    return NIXL_ERR_NOT_FOUND;
}

nixl_status_t
nixlSerDes::importStr(const std::string &buffer) {
    return importStr(std::string(buffer));
}

nixl_status_t
nixlSerDes::importStr(std::string &&buffer) noexcept {
    if (buffer.size() < nixl::header1.size()) {
        return NIXL_ERR_MISMATCH;
    }
    if (std::memcmp(buffer.data(), nixl::header1.data(), nixl::header1.size()) != 0) {
        return NIXL_ERR_MISMATCH;
    }
    buffer_ = std::move(buffer);
    offset_ = nixl::header1.size();
    return NIXL_SUCCESS;
}

std::string
nixlSerDes::_bytesToString(const void *buf, ssize_t size) {
    return std::string(reinterpret_cast<const char *>(buf), size);
}

void
nixlSerDes::_stringToBytes(void *fill_buf, const std::string &s, ssize_t size) {
    s.copy(reinterpret_cast<char *>(fill_buf), size);
}
