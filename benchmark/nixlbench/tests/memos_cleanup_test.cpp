/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils/memos_cleanup.h"
#include <iostream>
#include <stdexcept>

void
check(bool v) {
    if (!v) {
        throw std::runtime_error("MEMOS release safety assertion failed");
    }
}

int
main() {
    try {
        for (int pending : {0, 1, 7, 100}) {
            bool handle_owned = true, buffers_owned = true;
            int active = pending, progress_calls = 0, release_calls = 0;
            check(nixlbench::releaseMemosRequest(
                [&] {
                    check(handle_owned && buffers_owned);
                    ++release_calls;
                    if (active != 0) {
                        return nixlbench::MemosReleaseStatus::pending;
                    }
                    handle_owned = false;
                    return nixlbench::MemosReleaseStatus::released;
                },
                [&] {
                    check(handle_owned && buffers_owned && active > 0);
                    --active;
                    ++progress_calls;
                }));
            check(!handle_owned && buffers_owned && active == 0);
            check(progress_calls == pending && release_calls == pending + 1);
        }
        bool owner_retained = true;
        int progress_calls = 0;
        check(!nixlbench::releaseMemosRequest([&] { return nixlbench::MemosReleaseStatus::error; },
                                              [&] {
                                                  ++progress_calls;
                                                  owner_retained = false;
                                              }));
        check(owner_retained && progress_calls == 0);
        std::cout << "MEMOS deferred release tests passed\n";
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
