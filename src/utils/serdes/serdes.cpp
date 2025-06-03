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
#include <stdexcept>
#include <utility>

#include "serdes.h"

namespace
{
   const std::string_view introString = "nixlSerDes|";

}  // namespace

nixlSerializer::nixlSerializer()
    : buffer_(introString)
{}

nixlSerializer::nixlSerializer(const std::size_t pre)
{
    buffer_.reserve(pre);
    buffer_.append(introString);
}

void nixlSerializer::addRaw(const std::string_view& str)
{
    addRaw(str.data(), str.size());
}

void nixlSerializer::addRaw(const void* buf, const std::size_t len)
{
    buffer_.append(reinterpret_cast<const char*>(buf), len);
}

void nixlSerializer::addStr(const std::string_view& tag, const std::string_view& str)
{
    addBuf(tag, str.data(), str.size());
}

void nixlSerializer::addBuf(const std::string_view& tag, const void* buf, std::size_t len)
{
    buffer_.reserve(buffer_.size() + tag.size() + sizeof(len) + len + 1 );

    addRaw(tag);
    addRaw(&len, sizeof(len));
    addRaw(buf, len);
    addRaw("|");
}

std::string nixlSerializer::exportStr() && noexcept
{
    return std::move(buffer_);
}

const std::string& nixlSerializer::exportStr() const & noexcept
{
    return buffer_;
}

nixlDeserializer::nixlDeserializer(std::string&& str)
    : buffer_(std::move(str)),
      offset_(introString.size())
{
    if(buffer_.size() < introString.size() ) {
        throw std::runtime_error("insufficient deserialization data for intro");
    }
    if(std::memcmp(buffer_.data(), introString.data(), introString.size() != 0 )) {
        throw std::runtime_error("invalid deserializtion intro data");
    }
}

nixlDeserializer::nixlDeserializer(const std::string& str)
    : nixlDeserializer(std::string(str))
{}

std::size_t nixlDeserializer::peekLenUnsafe(const std::size_t offset) const
{
    std::size_t result;
    std::memcpy(&result, buffer_.data() + offset_ + offset, sizeof(result));
    return result;
}

std::string nixlDeserializer::getStr(const std::string_view& tag)
{
    if(offset_ + tag.size() + sizeof(std::size_t) > buffer_.size()) {
        // throw std::runtime_error("deserialization data insufficient");
        return "";
    }
    if(buffer_.compare(offset_, tag.size(), tag) != 0) {
        // throw std::runtime_error("deserialization tag mismatch ...");
        return "";
    }
    offset_ += tag.size();

    const std::size_t len = peekLenUnsafe();
    offset_ += sizeof(len);

    if(offset_ + len + 1 > buffer_.size()) {
        // throw std::runtime_error("deserialization data insufficient");
        return "";
    }
    const std::string ret = buffer_.substr(offset_, len);
    offset_ += len + 1;  // Also skip trailing '|'.
    return ret;
}

std::size_t nixlDeserializer::peekBufLen(const std::string_view& tag) const
{
    if(offset_ + tag.size() + sizeof(std::size_t) > buffer_.size()) {
        // throw std::runtime_error("deserialization data insufficient");
        return -1;
    }
    if(buffer_.compare(offset_, tag.size(), tag) != 0) {
        // throw std::runtime_error("deserialization tag mismatch ...");
        return -1;
    }
    return peekLenUnsafe(tag.size());
}

nixl_status_t nixlDeserializer::getBuf(const std::string_view& tag, void *buf, const std::size_t len)
{
    if(offset_ + tag.size() + sizeof(std::size_t) > buffer_.size()) {
        // throw std::runtime_error("deserialization data insufficient");
        return NIXL_ERR_MISMATCH;
    }
    if(buffer_.compare(offset_, tag.size(), tag) != 0) {
        // throw std::runtime_error("deserialization tag mismatch ...");
        return NIXL_ERR_MISMATCH;
    }
    if(peekBufLen(tag) != len) {  // NIXL_ASSERT(peekBufLen(tag) == len);
        // throw std::runtime_error("deserialization size mismatch");
        return NIXL_ERR_MISMATCH;
    }
    offset_ += tag.size() + sizeof(std::size_t);

    if(offset_ + len + 1 > buffer_.size()) {
        // throw std::runtime_error("deserialization data insufficient");
        return NIXL_ERR_MISMATCH;
    }
    std::memcpy(buf, buffer_.data() + offset_, len);
    offset_ += len + 1;  // Also skip trailing '|'.
    return NIXL_SUCCESS;
}

nixl_status_t nixlDeserializer::importStr(std::string raw)
{
    if(raw.size() < introString.size()) {
        return NIXL_ERR_MISMATCH;
    }
    if(raw.compare(0, introString.size(), introString) != 0) {
        return NIXL_ERR_MISMATCH;
    }
    buffer_ = std::move(raw);
    offset_ = introString.size();
    return NIXL_SUCCESS;
}
