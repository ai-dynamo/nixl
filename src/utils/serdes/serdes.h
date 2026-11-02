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
#ifndef NIXL_SRC_UTILS_SERDES_SERDES_H
#define NIXL_SRC_UTILS_SERDES_SERDES_H

#include <string>
#include <cstdint>
#include <optional>
#include <string_view>

#include "nixl_types.h"

class nixlSerDes {
private:
    typedef enum { SERIALIZE, DESERIALIZE } ser_mode_t;

    std::string workingStr;
    ssize_t des_offset;
    ser_mode_t mode;

public:
    nixlSerDes();

    /* Ser/Des for Strings */
    nixl_status_t addStr(const std::string &tag, const std::string &str);
    std::string getStr(const std::string &tag);

    /* Ser/Des for Byte buffers */
    nixl_status_t addBuf(const std::string &tag, const void* buf, ssize_t len);
    ssize_t getBufLen(const std::string &tag) const;

    /**
     * @brief Length of the next field's value when that field carries @p tag and is complete,
     *        without consuming it or logging.
     * @return std::nullopt when the remaining buffer holds no complete field tagged @p tag.
     */
    [[nodiscard]] std::optional<size_t>
    peekBufLen(std::string_view tag) const noexcept;

    nixl_status_t getBuf(const std::string &tag, void *buf, ssize_t len);

    /* Ser/Des buffer management */
    std::string
    exportStr() const;

    /* TODO: remove once we can add r-value overload */
    std::string
    exportMoveStr() &&;

    nixl_status_t importStr(const std::string &sdbuf);

    static std::string _bytesToString(const void *buf, ssize_t size);
    static void _stringToBytes(void* fill_buf, const std::string &s, ssize_t size);
};

#endif
