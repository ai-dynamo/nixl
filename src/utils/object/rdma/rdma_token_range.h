/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_UTILS_OBJECT_RDMA_RDMA_TOKEN_RANGE_H
#define NIXL_SRC_UTILS_OBJECT_RDMA_RDMA_TOKEN_RANGE_H

// Map a transfer address onto the cuObject registration that contains it.
//
// cuMemObjGetRDMAToken requires the address passed to cuMemObjGetDescriptor.
// A sub-range is named with buffer_offset, not by passing an interior pointer
// and offset 0. This helper is dependency-free so the address math can be
// tested without libcuobjclient.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace nixl_obj_rdma {

struct RdmaRegistration {
    uintptr_t base = 0;
    size_t length = 0;
};

// Arguments for cuMemObjGetRDMAToken: the registration base, and the byte
// offset of the transfer within that registration.
struct RdmaTokenArgs {
    uintptr_t base = 0;
    size_t offset = 0;
};

// True when the half-open range [start, start + size) lies inside
// [base, base + length). Empty ranges may end at base + length.
[[nodiscard]] inline bool
rdmaRangeInside(uintptr_t base, size_t length, uintptr_t start, size_t size) noexcept {
    if (start < base) {
        return false;
    }
    const uintptr_t rel = start - base;
    if (rel > length) {
        return false;
    }
    return size <= length - static_cast<size_t>(rel);
}

// Resolve (ptr, size, offset) against the live registrations.
//
// The transfer covers [ptr + offset, ptr + offset + size). When ptr itself is
// a registration base and that registration covers the range, it wins, so a
// whole-buffer transfer keeps offset unchanged. Otherwise the smallest
// containing registration is used (highest base breaks a length tie).
// Returns nullopt when the range overflows or is not fully inside one
// registration.
[[nodiscard]] inline std::optional<RdmaTokenArgs>
resolveRdmaTokenRange(std::span<const RdmaRegistration> regs,
                      uintptr_t ptr,
                      size_t size,
                      size_t offset) noexcept {
    if (offset > std::numeric_limits<uintptr_t>::max() - ptr) {
        return std::nullopt;
    }
    const uintptr_t start = ptr + offset;
    if (size > std::numeric_limits<uintptr_t>::max() - start) {
        return std::nullopt;
    }

    const RdmaRegistration *best = nullptr;
    for (const RdmaRegistration &reg : regs) {
        if (!rdmaRangeInside(reg.base, reg.length, start, size)) {
            continue;
        }
        if (reg.base == ptr) {
            if (best == nullptr || best->base != ptr || reg.length < best->length) {
                best = &reg;
            }
            continue;
        }
        if (best != nullptr && best->base == ptr) {
            continue;
        }
        if (best == nullptr || reg.length < best->length ||
            (reg.length == best->length && reg.base > best->base)) {
            best = &reg;
        }
    }
    if (best == nullptr) {
        return std::nullopt;
    }
    return RdmaTokenArgs{best->base, static_cast<size_t>(start - best->base)};
}

} // namespace nixl_obj_rdma

#endif // NIXL_SRC_UTILS_OBJECT_RDMA_RDMA_TOKEN_RANGE_H
