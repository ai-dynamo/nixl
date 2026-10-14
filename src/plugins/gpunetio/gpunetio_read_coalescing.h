/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_READ_COALESCING_H
#define NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_READ_COALESCING_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace nixl::gpunetio {

struct readRange {
    uintptr_t localAddr;
    uintptr_t remoteAddr;
    size_t size;
    uint32_t localKey;
    uint32_t remoteKey;
};

/** Merge only adjacent same-key ranges whose combined READ fits the SDK length limit. */
inline bool
canCoalesceRead(const readRange &previous, const readRange &next, size_t max_read_size) {
    return previous.size <= max_read_size && next.size <= max_read_size - previous.size &&
        previous.localAddr <= std::numeric_limits<uintptr_t>::max() - previous.size &&
        previous.remoteAddr <= std::numeric_limits<uintptr_t>::max() - previous.size &&
        previous.localAddr + previous.size == next.localAddr &&
        previous.remoteAddr + previous.size == next.remoteAddr &&
        previous.localKey == next.localKey && previous.remoteKey == next.remoteKey;
}

} // namespace nixl::gpunetio

#endif // NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_READ_COALESCING_H
