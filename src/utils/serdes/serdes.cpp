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

    workingStr.append(tag);
    workingStr.append(_bytesToString(&len, sizeof(size_t)));
    workingStr.append(str);
    workingStr.append("|");

    return NIXL_SUCCESS;
}

std::string nixlSerDes::getStr(const std::string &tag){

    if (workingStr.size() < des_offset + tag.size() + sizeof(size_t)) {
        NIXL_ERROR << "Deserialization of tag " << tag
                   << " failed for incomplete or missing header";
        return "";
    }

    if (std::memcmp(workingStr.data() + des_offset, tag.data(), tag.size()) != 0) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for tag mismatch";
        return "";
    }

    size_t len;
    std::memcpy(&len, workingStr.data() + des_offset + tag.size(), sizeof(len));
    des_offset += tag.size() + sizeof(len);

    if (workingStr.size() < des_offset + len + 1) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for incomplete data";
        return "";
    }

    const std::string ret = workingStr.substr(des_offset, len);

    // Skip string plus trailing '|'.
    des_offset += len + 1;

    if (ret.empty()) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for empty data";
    }
    return ret;
}

// Byte buffers serialization
nixl_status_t nixlSerDes::addBuf(const std::string &tag, const void* buf, ssize_t len){

    workingStr.append(tag);
    workingStr.append(_bytesToString(&len, sizeof(ssize_t)));
    workingStr.append(_bytesToString(buf, len));
    workingStr.append("|");

    return NIXL_SUCCESS;
}

ssize_t nixlSerDes::getBufLen(const std::string &tag) const{
    if (workingStr.size() < des_offset + tag.size() + sizeof(size_t)) {
        NIXL_ERROR << "Deserialization of tag " << tag
                   << " failed for incomplete or missing header";
        return -1;
    }

    if (std::memcmp(workingStr.data() + des_offset, tag.data(), tag.size()) != 0) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for tag mismatch";
        return -1;
    }

    size_t len;
    std::memcpy(&len, workingStr.data() + des_offset + tag.size(), sizeof(len));

    if (len == 0) {
        NIXL_WARN << "Deserialization of tag " << tag << " has data length zero";
    }
    return len;
}

nixl_status_t nixlSerDes::getBuf(const std::string &tag, void *buf, ssize_t len){
    if (workingStr.size() < des_offset + tag.size() + sizeof(size_t)) {
        NIXL_ERROR << "Deserialization of tag " << tag
                   << " failed for incomplete or missing header";
        return NIXL_ERR_MISMATCH;
    }

    if (std::memcmp(workingStr.data() + des_offset, tag.data(), tag.size()) != 0) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for tag mismatch";
        return NIXL_ERR_MISMATCH;
    }

    size_t tmp;
    std::memcpy(&tmp, workingStr.data() + des_offset + tag.size(), sizeof(tmp));
    des_offset += tag.size() + sizeof(tmp);


    // In existing code the value of len is often assumed instead
    // of the return value from a preceding call to getBufLen().

    if (size_t(len) != tmp) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for data length mismatch";
        return NIXL_ERR_MISMATCH;
    }

    if (workingStr.size() < size_t(des_offset + len + 1)) {
        NIXL_ERROR << "Deserialization of tag " << tag << " failed for incomplete data";
        return NIXL_ERR_MISMATCH;
    }

    std::memcpy(buf, workingStr.data() + des_offset, len);

    // Skip data and trailing '|'.
    des_offset += len + 1;

    return NIXL_SUCCESS;
}

// Buffer management serialization
std::string nixlSerDes::exportStr() const {
    return workingStr;
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
