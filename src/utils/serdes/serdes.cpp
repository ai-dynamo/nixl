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
    const std::string_view header1 = "N1XL";

    inline constexpr size_t varLenBias = 127;

    inline constexpr uint8_t lowerNibble = 0x0F;
    inline constexpr uint8_t upperNibble = 0xF0;
    inline constexpr uint8_t varLenNibble = 0xC0;

    void
    addRaw(std::string &buffer, const void *data, const size_t size) {
        buffer.append(static_cast<const char *>(data), size);
    }

    void
    addVarLen(std::string &buffer, size_t size) {
        // assert(size > varLenBias);

        uint8_t tmp[9];
        uint8_t used = 0;

        size -= varLenBias;

        // Can we use __builtin_clzl() to optimize this?
        while (size > 0) {
            tmp[++used] = uint8_t(size);
            size >>= 8;
        }
        tmp[0] = (used | varLenNibble);
        addRaw(buffer, tmp, 1 + used);
    }

    void
    addLen(std::string &buffer, const size_t size) {
        if (size <= varLenBias) {
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

} // namespace

} // namespace nixl

nixlSerDes::nixlSerDes() : buffer_(nixl::header1), offset_(buffer_.size()) {}

void
nixlSerDes::require(const size_t size) {
    if (offset_ + size > buffer_.size()) {
        throw std::runtime_error("serialized data is incomplete");
    }
}

void
nixlSerDes::consume(const size_t size) noexcept {
    // assert(offset_ + size <= buffer_.size());
    offset_ += size;
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
    // assert((first & upperNibble) == varLenNibble);
    const size_t bytes = (first & nixl::lowerNibble);
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
nixlSerDes::getViewLen() {
    const uint8_t first = getFirst();
    if (first <= nixl::varLenBias) {
        require(first);
        return first;
    }
    if ((first & nixl::upperNibble) == nixl::varLenNibble) {
        return getVarLen(first);
    }
    throw std::runtime_error("serialized data is mangled");
}

std::string_view
nixlSerDes::getView() {
    const size_t size = getViewLen();
    const char *data = buffer_.data() + offset_;
    consume(size);
    return std::string_view(data, size);
}

nixl_status_t
nixlSerDes::addStr(const std::string &tag, const std::string &str) {
    nixl::addStr(buffer_, tag);
    nixl::addStr(buffer_, str);
    return NIXL_SUCCESS;
}

std::string
nixlSerDes::getStr(const std::string &tag) {
    const size_t offset = offset_;
    try {
        if (getView() == tag) {
            return std::string(getView());
        }
        offset_ = offset;
        return {};
    }
    catch (...) {
        offset_ = offset;
        return {};
    }
}

nixl_status_t
nixlSerDes::addBuf(const std::string &tag, const void *buf, ssize_t len) {
    nixl::addStr(buffer_, tag);
    nixl::addLen(buffer_, len);
    nixl::addRaw(buffer_, buf, len);
    return NIXL_SUCCESS;
}

ssize_t
nixlSerDes::getBufLen(const std::string &tag) {
    const size_t offset = offset_;
    try {
        if (getView() == tag) {
            const size_t result = getViewLen();
            offset_ = offset;
            return result;
        }
        offset_ = offset;
        return -1;
    }
    catch (...) {
        offset_ = offset;
        return -1;
    }
}

nixl_status_t
nixlSerDes::getBuf(const std::string &tag, void *buf, ssize_t len) {
    const size_t offset = offset_;
    try {
        if (getView() == tag) {
            const std::string_view data = getView();
            if (data.size() == size_t(len)) {
                std::memcpy(buf, data.data(), data.size());
                return NIXL_SUCCESS;
            }
            offset_ = offset;
            return NIXL_ERR_MISMATCH;
        }
        offset_ = offset;
        return NIXL_ERR_NOT_FOUND;
    }
    catch (...) {
        offset_ = offset;
        return NIXL_ERR_UNKNOWN;
    }
}

std::string
nixlSerDes::exportStr() const {
    return buffer_;
}

nixl_status_t
nixlSerDes::importStr(const std::string &sdbuf) {
    if (sdbuf.size() < nixl::header1.size()) {
        return NIXL_ERR_MISMATCH;
    }
    if (std::memcmp(sdbuf.data(), nixl::header1.data(), nixl::header1.size()) != 0) {
        return NIXL_ERR_MISMATCH;
    }
    buffer_ = sdbuf;
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
