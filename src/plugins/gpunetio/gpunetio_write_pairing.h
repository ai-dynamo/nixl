/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_GPUNETIO_WRITE_PAIRING_H
#define NIXL_GPUNETIO_WRITE_PAIRING_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace nixl::gpunetio {

inline bool
canPairWrite(uintptr_t address,
             size_t length,
             uint32_t key,
             uintptr_t next_address,
             size_t next_length,
             uint32_t next_key,
             size_t remaining,
             size_t max_length) {
    // The final descriptor may carry an arrival marker; never fold it into a pair.
    return remaining > 2 && length != 0 && next_length != 0 && key == next_key &&
        next_address >= address && next_address - address == length &&
        next_length <= std::numeric_limits<uintptr_t>::max() - next_address &&
        next_length <= max_length && length <= max_length - next_length;
}

} // namespace nixl::gpunetio

#endif
