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
#include "serdes.h"
#include "common/nixl_log.h"

nixlSerDes::nixlSerDes() {
    workingStr = "nixlSerDes|";
    des_offset = 11;

    mode = SERIALIZE;
}

std::string nixlSerDes::_bytesToString(const void *buf, ssize_t size) {
    std::string ret_str = std::string(reinterpret_cast<const char*>(buf), size);
    return ret_str;
}

void nixlSerDes::_stringToBytes(void* fill_buf, const std::string &s, ssize_t size){
    s.copy(reinterpret_cast<char*>(fill_buf), size);
}

// Strings serialization
nixl_status_t nixlSerDes::addStr(const std::string &tag, const std::string &str){

    size_t len = str.size();

    workingStr.append(tag);
    workingStr.append(_bytesToString(&len, sizeof(size_t)));
    workingStr.append(str);
    workingStr.append("|");

    return NIXL_SUCCESS;
}

std::string nixlSerDes::getStr(const std::string &tag){

    if(workingStr.compare(des_offset, tag.size(), tag) != 0){
        NIXL_ERROR << "Deserialization of tag " << tag << " failed";
        return "";
    }
    ssize_t len;

    //skip tag
    des_offset += tag.size();

    //get len
    //_stringToBytes(&len, workingStr.data() + des_offset, sizeof(ssize_t));
    _stringToBytes(&len, workingStr.substr(des_offset, sizeof(ssize_t)), sizeof(ssize_t));
    des_offset += sizeof(ssize_t);

    //get string
    std::string ret = workingStr.substr(des_offset, len);

    //move past string plus | delimiter
    des_offset += len + 1;

    if (ret.empty()) NIXL_ERROR << "Deserialization of tag " << tag << " failed";

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
    if(workingStr.compare(des_offset, tag.size(), tag) != 0){
        NIXL_ERROR << "Deserialization of tag " << tag << " failed";
        return -1;
    }

    ssize_t len;

    //get len
    //_stringToBytes(&len, workingStr.data() + des_offset + tag.size(), sizeof(ssize_t));
    _stringToBytes(&len, workingStr.substr(des_offset + tag.size(), sizeof(ssize_t)), sizeof(ssize_t));

    if (len == 0) NIXL_WARN << "In deserialization of tag " << tag << " the buffer length ios 0";

    return len;
}

nixl_status_t nixlSerDes::getBuf(const std::string &tag, void *buf, ssize_t len){
    if(workingStr.compare(des_offset, tag.size(), tag) != 0){
        NIXL_ERROR << "Deserialization of tag " << tag << " failed";
        return NIXL_ERR_MISMATCH;
    }

    //skip over tag and size, which we assume has been read previously
    des_offset += tag.size() + sizeof(ssize_t);

    //_stringToBytes(buf, workingStr.data() + des_offset, len);
    _stringToBytes(buf, workingStr.substr(des_offset, len), len);

    //bytes in string form are twice as long, skip those plus | delimiter
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
