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

#include <cstring>
#include <string>
#include <string_view>
#include <cstdint>
#include <type_traits>

#include "nixl_types.h"

class nixlSerializer
{
public:
    nixlSerializer();

    explicit nixlSerializer(const std::size_t preAlloc);

    void addStr(const std::string_view& tag, const std::string_view& str);
    void addBuf(const std::string_view& tag, const void* buf, const std::size_t len);

    template<typename Int>
    void addInt(const std::string_view& tag, const Int in)
    {
        static_assert(std::is_integral_v<Int> || std::is_enum_v<Int>);
        addBuf(tag, &in, sizeof(in));
    }

    [[nodiscard]] std::string exportStr() && noexcept;
    [[nodiscard]] const std::string& exportStr() const & noexcept;

private:
    std::string buffer_;

    void addRaw(const std::string_view&);
    void addRaw(const void* buf, const std::size_t len);

    template<std::size_t N>
    void addRaw(const char(&literal)[N])
    {
        addRaw(literal, N - 1);
    }
};

class nixlDeserializer
{
public:
    explicit nixlDeserializer(std::string&&);
    explicit nixlDeserializer(const std::string&);

    // TODO: Change these functions to throw exceptions on failure:

    [[nodiscard]] std::string getStr(const std::string_view& tag);  // Returns empty string on failure.

    template<typename Int>
    [[nodiscard]] nixl_status_t getInt(const std::string_view& tag, Int& out)
    {
        static_assert(std::is_integral_v<Int> || std::is_enum_v<Int>);
        return getIntImpl(tag, &out, sizeof(out));
    }

    // TODO: Remove all functions below after switching to exceptions:

    nixlDeserializer() noexcept = default;

    [[nodiscard]] nixl_status_t importStr(std::string);

private:
    std::string buffer_;
    std::size_t offset_ = 0;

    [[nodiscard]] std::size_t peekLenUnsafe(const std::size_t offset = 0) const;
    [[nodiscard]] nixl_status_t getIntImpl(const std::string_view& tag, void* data, const std::size_t size);
};

#endif
