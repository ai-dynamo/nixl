/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/memos_key_pool.h"

#include <iostream>
#include <stdexcept>

namespace {

void
check(bool condition) {
    if (!condition) {
        throw std::runtime_error("key-pool configuration assertion failed");
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
    throw std::runtime_error("invalid key-pool configuration accepted");
}

} // namespace

int
main() {
    try {
        check(nixlbench::parseMemosKeyPoolSize(nullptr) == 0);
        check(nixlbench::parseMemosKeyPoolSize("2048") == 2048);
        check(nixlbench::parseMemosKeyPoolSize("1000000") == 1000000);
        for (const char *text : {"",
                                 "0",
                                 "-1",
                                 "+32",
                                 " 32",
                                 "32 ",
                                 "32junk",
                                 "1.5",
                                 "1000001",
                                 "18446744073709551616"}) {
            reject([&] { nixlbench::parseMemosKeyPoolSize(text); });
        }

        check(!nixlbench::parseMemosReuseLocal(nullptr));
        check(!nixlbench::parseMemosReuseLocal("0"));
        check(nixlbench::parseMemosReuseLocal("1"));
        for (const char *text : {"", "2", "-1", "true", "1junk", " 1"}) {
            reject([&] { nixlbench::parseMemosReuseLocal(text); });
        }
        // 8,192 remote keys, but only 64 local values per worker.
        nixlbench::validateMemosKeyPoolSize(8192, 32, 512 * 1024, 256 * 512 * 1024, 4, 2, true);
        reject([&] {
            nixlbench::validateMemosKeyPoolSize(
                8192, 32, 512 * 1024, 256 * 512 * 1024 - 1, 4, 2, true);
        });
        reject([&] { nixlbench::validateMemosKeyPoolSize(96, 32, 1, 64, 1, 2, true); });
        reject([&] {
            nixlbench::validateMemosKeyPoolSize(
                8192, 32, 512 * 1024, 256 * 512 * 1024, 4, 2, false);
        });
        constexpr size_t gib = size_t{1} << 30;
        // Eight threads, 2,048 keys per thread, 512 KiB values: exactly 8 GiB.
        nixlbench::validateMemosKeyPoolSize(2048, 32, 512 * 1024, 8 * gib, 8);
        nixlbench::validateMemosKeyPoolSize(2048, 32, 512 * 1024, 8 * gib, 8, 2);
        nixlbench::validateMemosKeyPoolSize(2048, 32, 512 * 1024, 8 * gib, 8, 64);
        for (int depth : {-1, 0, 65}) {
            reject([&] {
                nixlbench::validateMemosKeyPoolSize(2048, 32, 512 * 1024, 8 * gib, 8, depth);
            });
        }
        // Whole-bank rounding supports non-power-of-two value sizes as well.
        nixlbench::validateMemosKeyPoolSize(2048, 32, 508 * 1024, 8 * gib, 8);
        nixlbench::validateMemosKeyPoolSize(32, 32, 512 * 1024, 8 * gib, 8);
        reject([&] { nixlbench::validateMemosKeyPoolSize(2048, 32, 512 * 1024, 8 * gib - 1, 8); });
        reject([&] { nixlbench::validateMemosKeyPoolSize(2048, 32, 1024 * 1024, 8 * gib, 8); });
        reject([&] { nixlbench::validateMemosKeyPoolSize(2047, 32, 512 * 1024, 8 * gib, 8); });
        reject([&] { nixlbench::validateMemosKeyPoolSize(16, 32, 512 * 1024, 8 * gib, 8); });
        reject([&] { nixlbench::validateMemosKeyPoolSize(0, 32, 512 * 1024, 8 * gib, 8); });
        reject([&] { nixlbench::validateMemosKeyPoolSize(2048, 0, 512 * 1024, 8 * gib, 8); });
        reject([&] { nixlbench::validateMemosKeyPoolSize(2048, 32, 0, 8 * gib, 8); });
        reject([&] { nixlbench::validateMemosKeyPoolSize(2048, 32, 512 * 1024, 8 * gib, 0); });
        reject([&] {
            nixlbench::validateMemosKeyPoolSize(
                1000000, 32, 1, std::numeric_limits<size_t>::max(), 4096);
        });
        std::cout << "MEMOS key-pool validation passed\n";
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
