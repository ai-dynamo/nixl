/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_WIRE_H
#define NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_WIRE_H

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <system_error>

namespace nixl::gpunetio {

template<typename unsignedType>
inline bool
parseUnsigned(std::string_view field, unsignedType &value) {
    if (field.empty()) {
        return false;
    }
    const auto result = std::from_chars(field.data(), field.data() + field.size(), value);
    return result.ec == std::errc{} && result.ptr == field.data() + field.size();
}

/** Parse the existing decimal rkey-address-size metadata without unchecked indexing. */
inline bool
parseRemoteMetadata(std::string_view input, uint32_t &rkey, uintptr_t &addr, size_t &size) {
    const size_t first = input.find('-');
    if (first == std::string_view::npos) {
        return false;
    }
    const size_t second = input.find('-', first + 1);
    if (second == std::string_view::npos) {
        return false;
    }
    uint32_t parsed_rkey;
    uintptr_t parsed_addr;
    size_t parsed_size;
    if (!parseUnsigned(input.substr(0, first), parsed_rkey) ||
        !parseUnsigned(input.substr(first + 1, second - first - 1), parsed_addr) ||
        !parseUnsigned(input.substr(second + 1), parsed_size)) {
        return false;
    }
    rkey = parsed_rkey;
    addr = parsed_addr;
    size = parsed_size;
    return true;
}

/** Decode a notification within its slot; neither header nor payload needs a trailing NUL. */
inline bool
parseNotification(std::string_view slot,
                  std::string_view start_tag,
                  std::string_view end_tag,
                  std::string_view &payload) {
    if (!slot.starts_with(start_tag)) {
        return false;
    }
    const size_t end = slot.find(end_tag, start_tag.size());
    if (end == std::string_view::npos) {
        return false;
    }
    size_t payload_size;
    if (!parseUnsigned(slot.substr(start_tag.size(), end - start_tag.size()), payload_size)) {
        return false;
    }
    const size_t offset = end + end_tag.size();
    if (payload_size > slot.size() - offset) {
        return false;
    }
    payload = slot.substr(offset, payload_size);
    return true;
}

} // namespace nixl::gpunetio

#endif // NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_WIRE_H
