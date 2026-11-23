/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/memos_buffer.h"
#include <iostream>
#include <stdexcept>

namespace {
void
check(bool condition) {
    if (!condition) {
        throw std::runtime_error("incorrect MEMOS local buffer mapping");
    }
}

template<typename Function>
void
reject(Function function) {
    try {
        function();
    }
    catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid MEMOS local buffer geometry accepted");
}
} // namespace

int
main() {
    try {
        constexpr size_t hugepage = 2 * 1024 * 1024;
        for (size_t kib : {128, 256, 508, 512, 1020, 1024, 2048, 4096, 8192}) {
            for (size_t batch : {size_t{1}, size_t{16}, size_t{32}}) {
                for (size_t depth : {size_t{1}, size_t{2}, size_t{4}, size_t{32}}) {
                    const size_t block = kib * 1024;
                    const size_t logical = batch * depth * block;
                    const size_t padded = ((logical + hugepage - 1) / hugepage) * hugepage;
                    // Exact, hugepage-padded and oversized regions must all use
                    // the same logical ring, even in a multi-size sweep.
                    for (size_t allocation : {logical, padded, padded + 2 * hugepage}) {
                        for (size_t key = 0; key < 8192; ++key) {
                            const auto offset = nixlbench::memosLocalBufferOffset(
                                key, block, batch, depth, allocation);
                            check(offset == (key % (batch * depth)) * block);
                            check(offset + block <= logical && offset + block <= allocation);
                            // Different lanes have disjoint ranges. Advancing a
                            // lane by depth banks retains exactly its buffers.
                            const size_t next = key + batch * depth;
                            check(offset ==
                                  nixlbench::memosLocalBufferOffset(
                                      next, block, batch, depth, allocation));
                        }
                    }
                    reject([&] {
                        nixlbench::memosLocalBufferOffset(0, block, batch, depth, logical - 1);
                    });
                }
            }
        }
        const size_t limit = std::numeric_limits<size_t>::max();
        reject([&] { nixlbench::memosLocalBufferOffset(0, 0, 32, 2, 64); });
        reject([&] { nixlbench::memosLocalBufferOffset(0, 1, 0, 2, 64); });
        reject([&] { nixlbench::memosLocalBufferOffset(0, 1, 32, 0, 64); });
        reject([&] { nixlbench::memosLocalBufferOffset(0, 1, limit, 2, limit); });
        reject([&] { nixlbench::memosLocalBufferOffset(0, limit, 32, 2, limit); });
        check(nixlbench::memosLocalBufferOffset(limit, 1, 16, 2, 32) == limit % 32);
        std::cout << "MEMOS local buffer tests passed\n";
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
