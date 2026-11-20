/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <limits>
#include <utility>
#include <variant>

#include "serdes.h"
#include "common/nixl_log.h"

namespace {

enum class FieldError { IncompleteHeader, TagMismatch, IncompleteData };

std::variant<size_t, FieldError>
parseFieldLength(std::string_view wire, size_t offset, std::string_view tag) noexcept {
    if (offset > wire.size()) {
        return FieldError::IncompleteHeader;
    }

    const size_t remaining = wire.size() - offset;
    if (tag.size() > remaining || remaining - tag.size() < sizeof(size_t)) {
        return FieldError::IncompleteHeader;
    }
    if (!tag.empty() && std::memcmp(wire.data() + offset, tag.data(), tag.size()) != 0) {
        return FieldError::TagMismatch;
    }

    size_t len;
    std::memcpy(&len, wire.data() + offset + tag.size(), sizeof(len));
    const size_t header_end = offset + tag.size() + sizeof(len);
    // Reserve the trailing byte without adding the untrusted length to the cursor.
    if (wire.size() - header_end <= len) {
        return FieldError::IncompleteData;
    }
    return len;
}

std::optional<size_t>
fieldLengthOrLog(std::string_view wire, size_t offset, const std::string &tag) {
    const auto field = parseFieldLength(wire, offset, tag);
    if (const auto *len = std::get_if<size_t>(&field)) {
        return *len;
    }

    switch (std::get<FieldError>(field)) {
    case FieldError::IncompleteHeader:
        NIXL_ERROR << "Deserialization of tag " << tag
                   << " failed for incomplete or missing header";
        break;
    case FieldError::TagMismatch:
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for tag mismatch";
        break;
    case FieldError::IncompleteData:
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for incomplete data";
        break;
    }
    return std::nullopt;
}

} // namespace

nixlSerDes::nixlSerDes()
    : workingStr("nixlSerDes|"),
      des_offset(workingStr.size()),
      mode(SERIALIZE) {}

std::string nixlSerDes::_bytesToString(const void *buf, ssize_t size) {
    return std::string(reinterpret_cast<const char *>(buf), size);
}

void nixlSerDes::_stringToBytes(void* fill_buf, const std::string &s, ssize_t size){
    s.copy(reinterpret_cast<char*>(fill_buf), size);
}

// Strings serialization
nixl_status_t nixlSerDes::addStr(const std::string &tag, const std::string &str){

    const size_t len = str.size();

    workingStr.reserve(workingStr.size() + tag.size() + sizeof(len) + len + 1);
    workingStr.append(tag);
    workingStr.append(_bytesToString(&len, sizeof(size_t)));
    workingStr.append(str);
    workingStr.append("|");

    return NIXL_SUCCESS;
}

std::string
nixlSerDes::getStr(const std::string &tag) {
    const auto len = fieldLengthOrLog(workingStr, des_offset, tag);
    if (!len) {
        return "";
    }

    const size_t value_offset = des_offset + tag.size() + sizeof(size_t);
    const std::string ret = workingStr.substr(value_offset, *len);

    // Skip string plus trailing '|'.
    des_offset = value_offset + *len + 1;

    if (ret.empty() && tag != "msg") {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for empty data";
    }
    return ret;
}

std::optional<size_t>
nixlSerDes::peekBufLen(std::string_view tag) const noexcept {
    const auto field = parseFieldLength(workingStr, des_offset, tag);
    if (const auto *len = std::get_if<size_t>(&field)) {
        return *len;
    }
    return std::nullopt;
}

// Byte buffers serialization
nixl_status_t nixlSerDes::addBuf(const std::string &tag, const void* buf, ssize_t len){

    workingStr.append(tag);
    workingStr.append(_bytesToString(&len, sizeof(ssize_t)));
    workingStr.append(_bytesToString(buf, len));
    workingStr.append("|");

    return NIXL_SUCCESS;
}

ssize_t
nixlSerDes::getBufLen(const std::string &tag) const {
    const auto len = fieldLengthOrLog(workingStr, des_offset, tag);
    if (!len) {
        return -1;
    }
    if (*len > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for data length mismatch";
        return -1;
    }

    if (*len == 0) {
        NIXL_WARN << "Deserialization of tag " << tag << " has data length zero";
    }
    return static_cast<ssize_t>(*len);
}

nixl_status_t
nixlSerDes::getBuf(const std::string &tag, void *buf, ssize_t len) {
    // In existing code the value of len is often assumed instead
    // of the return value from a preceding call to getBufLen().
    const auto field_len = fieldLengthOrLog(workingStr, des_offset, tag);
    if (!field_len) {
        return NIXL_ERR_MISMATCH;
    }
    if (len < 0 || static_cast<size_t>(len) != *field_len) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for data length mismatch";
        return NIXL_ERR_MISMATCH;
    }

    const size_t value_offset = des_offset + tag.size() + sizeof(size_t);
    if (*field_len != 0) {
        std::memcpy(buf, workingStr.data() + value_offset, *field_len);
    }

    // Skip data and trailing '|'.
    des_offset = value_offset + *field_len + 1;

    return NIXL_SUCCESS;
}

// Buffer management serialization
std::string nixlSerDes::exportStr() const {
    return workingStr;
}

std::string
nixlSerDes::exportMoveStr() && {
    return std::move(workingStr);
}

nixl_status_t nixlSerDes::importStr(const std::string &sdbuf) {

    if(sdbuf.compare(0, 11, "nixlSerDes|") != 0){
        NIXL_ERROR << "Deserialization failed, missing nixlSerDes tag";
        return NIXL_ERR_MISMATCH;
    }

    workingStr = sdbuf;
    mode = DESERIALIZE;
    des_offset = 11;

    return NIXL_SUCCESS;
}
