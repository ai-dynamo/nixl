// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#define NIXL_TEST_CHECK(condition)                                                    \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::cerr << __FILE__ << ":" << __LINE__ << ": " #condition << std::endl; \
            std::exit(1);                                                             \
        }                                                                             \
    } while (false)

template<typename F>
void
checkThrows(F &&function) {
    try {
        function();
    }
    catch (const std::exception &) {
        return;
    }
    NIXL_TEST_CHECK(false);
}
