// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "gpunetio_backend_aux.h"
#include <iostream>
#include <string>
#include <vector>

int
main() {
    using namespace nixl::gpunetio;
    const auto check = [](bool ok) {
        if (!ok) {
            throw std::runtime_error("native pool option mismatch");
        }
    };
    check(parseNativeQps("") == 1);
    for (uint32_t n = 1; n <= 32; ++n) {
        check(parseNativeQps(std::to_string(n)) == n);
        for (uint32_t channel = 0; channel < n; ++channel) {
            uint32_t q = 99;
            check(parseNativeChannel("native_channel=" + std::to_string(channel), n, q) &&
                  q == channel);
        }
        uint32_t q = 99;
        check(parseNativeChannel("", n, q) && q == 0);
        check(!parseNativeChannel("native_channel=" + std::to_string(n), n, q));
    }
    for (const auto value : {"0", "33", "-1", "+1", "1x", " 1", "1 "}) {
        bool rejected = false;
        try {
            parseNativeQps(value);
        }
        catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected);
    }
    for (const std::string &value :
         std::vector<std::string>{"native_channel=",
                                  "native_channel=-1",
                                  "native_channel=32",
                                  "native_channel=1x",
                                  "channel=1",
                                  std::string("native_channel=1\0x", 18)}) {
        uint32_t q = 0;
        check(!parseNativeChannel(value, 32, q));
    }
    std::cout << "PASS pool sizes, all channels, invalid/binary options\n";
}
