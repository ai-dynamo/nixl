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

#include "nixl_types.h"

class nixlSerDes {
private:
    std::string buffer_; // For both.
    std::size_t offset_; // For deserialization.

    void
    consume(size_t) noexcept;
    void
    require(size_t);

    [[nodiscard]] uint8_t
    getFirst();
    [[nodiscard]] size_t
    getVarLen(uint8_t);
    [[nodiscard]] size_t
    getTagLen();
    [[nodiscard]] std::string_view
    getTag();
    void
    skipValue();
    [[nodiscard]] size_t
    findTag(const std::string_view &);
    [[nodiscard]] ssize_t
    getBufLenImpl();
    [[nodiscard]] nixl_status_t
    getBufImpl1(size_t, void *, ssize_t);
    [[nodiscard]] nixl_status_t
    getBufImpl2(uint8_t, uint8_t, void *, ssize_t);
    [[nodiscard]] nixl_status_t
    getBufImpl3(uint8_t, void *, ssize_t);
    [[nodiscard]] nixl_status_t
    getBufImpl(void *, ssize_t);

public:
    nixlSerDes(); // Throws on out-of-memory.

    // Compatibility warning: getStr() will no longer deserialize all values from addBuf().

    nixl_status_t
    addStr(const std::string &tag,
           const std::string &str); // Always returns NIXL_SUCCESS, throws on out-of-memory.
    std::string
    getStr(const std::string &tag); // Returns empty string if tag not found.

    // Compatibility warning: getBufLen() can no longer be declared const.

    nixl_status_t
    addBuf(const std::string &tag,
           const void *buf,
           ssize_t len); // Always returns NIXL_SUCCESS, throws on out-of-memory.
    ssize_t
    getBufLen(const std::string &tag) /* const */; // Returns -1 if tag not found; was const!
    nixl_status_t
    getBuf(const std::string &tag,
           void *buf,
           ssize_t len); // Returns NIXL_ERR_MISMATCH if tag not found.

    // Compatibility warning: All getFoo() functions now search forward for the tag; if the tag
    // is not found the position in the buffer is not changed; if the tag is found all tag-value
    // pairs between the current position and the found tag are skipped.

    std::string
    exportStr() const {
        return buffer_;
    }

    nixl_status_t
    importStr(const std::string &buffer);

    static std::string
    _bytesToString(const void *buf, ssize_t size);
    static void
    _stringToBytes(void *fill_buf, const std::string &s, ssize_t size);

    // Additional functions.

    explicit nixlSerDes(const size_t reserve);  // Reserves space before serializing.

    [[nodiscard]] bool
    syntaxCheck() noexcept;  // Can be called after importStr().

    nixl_status_t
    importStr(std::string &&) noexcept;

    [[nodiscard]] size_t
    totalSize() const noexcept {
        return buffer_.size();
    }

    [[nodiscard]] size_t
    remainingSize() const noexcept {
        return buffer_.size() - offset_;
    }

    // Changes that are not implemented yet.

    // Change to const std::string_view& arguments from const std::string& where possible.

    // These two could replace the original exportStr() function...
    // [[nodiscard]] std::string exportStr() && noexcept
    // {
    //     return std::move( buffer_ );
    // }

    // [[nodiscard]] const std::string& exportStr() const & noexcept
    // {
    //     return buffer_;
    // }

    // Could be implemented to replace addBuf() and getBuf() for integers/enums/bool.
    // template<typename Int>
    // [[nodiscard]] nixl_status_t addInt(const std::string_view& tag, const Int value);
    // template<typename Int>
    // [[nodiscard]] nixl_status_t getInt(const std::string_view& tag, Int& result);
    // Alternatively the second version could return a std::optional<Int> or throw.

    // Alternative form of getStr() that allows distinguishing "not found" from "empty value".
    // [[nodiscard]] nixl_status_t getStr(const std::string_view& tag, std::string& value);
};

#endif
