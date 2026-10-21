/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>

// First eight bytes encode an injective identity within the supported workload.
#ifdef __CUDACC__
__host__ __device__
#endif
    inline uint8_t
    payloadByte(uint64_t epoch,
                unsigned peer,
                unsigned layer,
                unsigned row,
                unsigned slot,
                unsigned byte) {
    const uint64_t id = (epoch << 24) | (uint64_t(peer) << 20) | (uint64_t(layer) << 12) |
        (uint64_t(row) << 4) | slot;
    if (byte < 8) {
        return static_cast<uint8_t>(id >> (byte * 8));
    }
    uint64_t mix = id ^ (uint64_t(byte) * 0x9e3779b97f4a7c15ULL);
    mix = (mix ^ (mix >> 30)) * 0xbf58476d1ce4e5b9ULL;
    mix = (mix ^ (mix >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<uint8_t>(mix ^ (mix >> 31));
}
