// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "test_support.h"
#include "ucx/registration.h"

int
main() {
    using nixl::ucx::registrationTransaction;
    int releases = 0;
    int finishes = 0;
    auto cleanup = [&]() noexcept { ++releases; };
    NIXL_TEST_CHECK(!registrationTransaction([] { return false; },
                                             [&] {
                                                 ++finishes;
                                                 return true;
                                             },
                                             cleanup));
    NIXL_TEST_CHECK(releases == 0 && finishes == 0);
    NIXL_TEST_CHECK(!registrationTransaction([] { return true; }, [] { return false; }, cleanup));
    NIXL_TEST_CHECK(releases == 1);
    checkThrows([&] {
        (void)registrationTransaction([] { return true; },
                                      []() -> bool { throw std::runtime_error("pack failed"); },
                                      cleanup);
    });
    NIXL_TEST_CHECK(releases == 2);
    NIXL_TEST_CHECK(registrationTransaction([] { return true; }, [] { return true; }, cleanup));
    NIXL_TEST_CHECK(releases == 2);

    // The real registration path nests a map/query transaction inside a map/pack transaction.
    int unmaps = 0;
    int packs = 0;
    for (bool query_ok : {false, true}) {
        auto map_and_query = [&] {
            return registrationTransaction(
                [] { return true; }, [&] { return query_ok; }, [&]() noexcept { ++unmaps; });
        };
        NIXL_TEST_CHECK(!registrationTransaction(
            map_and_query,
            [&] {
                ++packs;
                return false;
            },
            [&]() noexcept { ++unmaps; }));
    }
    NIXL_TEST_CHECK(unmaps == 2 && packs == 1);
    std::cout << "registration: map/query/pack rollback and exceptions passed\n";
}
