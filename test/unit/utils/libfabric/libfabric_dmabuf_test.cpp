/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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

/*
 * Arithmetic of a CUDA dmabuf export, covered without a GPU.
 *
 * cuMemGetHandleForAddressRange takes a page-aligned range while the MR describes the
 * caller's exact bytes, so the export widens the range to page boundaries and carries the
 * difference in fi_mr_dmabuf::offset: the registered region is base_addr + offset for len
 * bytes. These cases pin the widening and the offset.
 */

#include "libfabric/libfabric_cuda_dmabuf.h"
#include "common/nixl_log.h"

#include <cstdint>

namespace {

constexpr size_t PAGE = 4096;

struct AlignCase {
    const char *name;
    uintptr_t addr;
    size_t length;
    uintptr_t expected_base;
    size_t expected_aligned_length;
    uint64_t expected_offset;
};

int
checkCase(const AlignCase &test_case) {
    uintptr_t base = 0;
    size_t aligned_length = 0;
    uint64_t offset = 0;
    LibfabricUtils::cudaDmabufAlignRange(reinterpret_cast<const void *>(test_case.addr),
                                        test_case.length,
                                        PAGE,
                                        base,
                                        aligned_length,
                                        offset);

    int rc = 0;
    if (base != test_case.expected_base) {
        NIXL_ERROR << test_case.name << ": aligned base expected 0x" << std::hex
                   << test_case.expected_base << ", got 0x" << base << std::dec;
        rc = 1;
    }
    if (aligned_length != test_case.expected_aligned_length) {
        NIXL_ERROR << test_case.name << ": aligned length expected "
                   << test_case.expected_aligned_length << ", got " << aligned_length;
        rc = 1;
    }
    if (offset != test_case.expected_offset) {
        NIXL_ERROR << test_case.name << ": offset expected " << test_case.expected_offset
                   << ", got " << offset;
        rc = 1;
    }

    // Invariants the MR relies on, checked independently of the expectations above.
    if (base + offset != test_case.addr) {
        NIXL_ERROR << test_case.name << ": base + offset must reproduce the buffer address";
        rc = 1;
    }
    if (offset + ((test_case.length == 0) ? 1 : test_case.length) > aligned_length) {
        NIXL_ERROR << test_case.name << ": exported range does not cover the region";
        rc = 1;
    }
    if ((base % PAGE) != 0 || (aligned_length % PAGE) != 0) {
        NIXL_ERROR << test_case.name << ": base and length must both be page aligned";
        rc = 1;
    }
    return rc;
}

} // namespace

int
main() {
    NIXL_INFO << "=== Testing CUDA dmabuf export alignment arithmetic ===";

    const AlignCase cases[] = {
        {"page-aligned, exactly one page", 0x10000, PAGE, 0x10000, PAGE, 0},
        {"page-aligned, exactly two pages", 0x10000, 2 * PAGE, 0x10000, 2 * PAGE, 0},
        // A buffer one byte past a page start spans two pages.
        {"offset by one byte, one page long", 0x10001, PAGE, 0x10000, 2 * PAGE, 1},
        // A region ending on a page boundary spans one page.
        {"unaligned start, ends on a page boundary", 0x10800, PAGE / 2, 0x10000, PAGE, 0x800},
        {"one byte, unaligned", 0x10abc, 1, 0x10000, PAGE, 0xabc},
        {"one byte at the last byte of a page", 0x10fff, 1, 0x10000, PAGE, 0xfff},
        // A region starting mid-page and two pages long spans three pages.
        {"unaligned start spanning three pages", 0x10800, 2 * PAGE, 0x10000, 3 * PAGE, 0x800},
        // A zero length rounds up to one page.
        {"zero length", 0x10800, 0, 0x10000, PAGE, 0x800},
        {"large region, page aligned", 0x200000, 64u * 1024 * 1024, 0x200000, 64u * 1024 * 1024, 0},
    };

    int rc = 0;
    for (const auto &test_case : cases) {
        if (checkCase(test_case) != 0) {
            rc = 1;
        }
    }

    if (rc == 0) {
        NIXL_INFO << "   SUCCESS: dmabuf export ranges are page aligned, cover the region, and "
                     "reproduce the buffer address from base + offset";
    }
    return rc;
}
