/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
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
#include "libfabric/libfabric_common.h"
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

struct VaCase {
    const char *name;
    bool auto_mode;
    bool copies_through_host;
    bool provider_bounds_gdrcopy_reads;
    bool expected;
};

struct VersionCase {
    uint32_t version;
    bool expected;
};

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

    NIXL_INFO << "=== Testing when Auto registers CUDA memory by virtual address ===";

    const VaCase va_cases[] = {
        // No GPU P2P (e.g. g6) and a provider that bounds GDRCopy reads: keep GDRCopy
        {"auto, host bounce buffers, bounded GDRCopy reads", true, true, true, true},
        // Without the bound, GDRCopy would serve bulk copies out of GPU memory: keep the dmabuf
        {"auto, host bounce buffers, unbounded GDRCopy reads", true, true, false, false},
        // GPUDirect rails (p4d, p5, p6, GB200) keep the dmabuf the topology selects
        {"auto, GPUDirect, bounded GDRCopy reads", true, false, true, false},
        {"auto, GPUDirect, unbounded GDRCopy reads", true, false, false, false},
        // A forced mapping type is honored on every rail
        {"forced, host bounce buffers, bounded GDRCopy reads", false, true, true, false},
        {"forced, GPUDirect, bounded GDRCopy reads", false, false, true, false},
    };

    for (const auto &test_case : va_cases) {
        const bool by_va = LibfabricUtils::cudaAutoRegistersByVirtualAddress(
            test_case.auto_mode,
            test_case.copies_through_host,
            test_case.provider_bounds_gdrcopy_reads);
        if (by_va != test_case.expected) {
            NIXL_ERROR << test_case.name << ": expected registration "
                       << (test_case.expected ? "by virtual address" : "with the dmabuf")
                       << ", got " << (by_va ? "virtual address" : "dmabuf");
            rc = 1;
        }
    }

    if (rc == 0) {
        NIXL_INFO << "   SUCCESS: Auto registers by virtual address only on rails that copy "
                     "through host bounce buffers with a provider that bounds GDRCopy reads";
    }

    NIXL_INFO << "=== Testing which libfabric versions bound GDRCopy reads ===";

    const VersionCase version_cases[] = {
        // The oldest API versions the plugin requests, and the releases before the bound
        {FI_VERSION(1, 18), false},
        {FI_VERSION(1, 22), false},
        {FI_VERSION(2, 7), false},
        // The bound is merged for 2.8 (ofiwg/libfabric#13074) and stays in later releases
        {FI_VERSION(2, 8), true},
        {FI_VERSION(2, 9), true},
        {FI_VERSION(3, 0), true},
    };

    bool versions_ok = true;
    for (const auto &test_case : version_cases) {
        const bool bounds = LibfabricUtils::efaBoundsGdrcopyFromHmem(test_case.version);
        if (bounds != test_case.expected) {
            NIXL_ERROR << "libfabric " << FI_MAJOR(test_case.version) << "."
                       << FI_MINOR(test_case.version) << ": expected the bound "
                       << (test_case.expected ? "present" : "absent") << ", got "
                       << (bounds ? "present" : "absent");
            versions_ok = false;
            rc = 1;
        }
    }

    if (versions_ok) {
        NIXL_INFO << "   SUCCESS: libfabric 2.8 and later bound GDRCopy reads, earlier "
                     "versions do not";
    }
    return rc;
}
