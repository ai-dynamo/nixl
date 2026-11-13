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

#include "serdes.h"
#include "common/nixl_log.h"

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
    const auto len = peekBufLen(tag);
    if (!len) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for invalid or incomplete field";
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
    if (des_offset > workingStr.size()) {
        return std::nullopt;
    }

    const size_t remaining = workingStr.size() - des_offset;
    if (tag.size() > remaining || remaining - tag.size() < sizeof(size_t) ||
        (!tag.empty() &&
         std::memcmp(workingStr.data() + des_offset, tag.data(), tag.size()) != 0)) {
        return std::nullopt;
    }

    size_t len;
    std::memcpy(&len, workingStr.data() + des_offset + tag.size(), sizeof(len));
    const size_t header_end = des_offset + tag.size() + sizeof(size_t);
    // Reserve the trailing byte without adding the untrusted length to the cursor.
    if (workingStr.size() - header_end <= len) {
        return std::nullopt;
    }
    return len;
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
    const auto len = peekBufLen(tag);
    if (!len || *len > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for invalid or incomplete field";
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
    const auto field_len = peekBufLen(tag);
    if (!field_len || len < 0 || static_cast<size_t>(len) != *field_len) {
        NIXL_ERROR << "Deserialization of tag " << tag
                   << " failed for invalid field or data length mismatch";
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
