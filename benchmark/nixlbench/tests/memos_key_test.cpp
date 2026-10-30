/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/memos_key.h"

#include <array>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {

void
check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Validate the binary key contract, including entropy outside byte zero,
// and the independent object sets needed by a four-client benchmark.
template<typename Generator>
void
checkClient(Generator &generator, std::set<std::string> &all_keys) {
    std::array<std::set<unsigned int>, 16> byte_values;
    for (int thread = 0; thread < 8; ++thread) {
        for (int batch = 0; batch < 32; ++batch) {
            const std::string key = nixlbench::makeMemosKey(generator);
            check(key.size() == 32, "Expected a 16-byte key encoded as 32 hex digits");
            check(key.find_first_not_of("0123456789abcdef") == std::string::npos,
                  "Key must use the backend's hex parser");
            check(all_keys.insert(key).second, "Object key reused within or across clients");
            for (size_t byte = 0; byte < byte_values.size(); ++byte) {
                byte_values[byte].insert(std::stoul(key.substr(byte * 2, 2), nullptr, 16));
            }
        }
    }
    for (const auto &values : byte_values) {
        check(values.size() > 1, "Every key byte must vary across the workload");
    }
}

} // namespace

int
main() {
    try {
        std::set<std::string> deterministic_keys;
        for (unsigned int client = 0; client < 4; ++client) {
            std::mt19937 generator(client);
            checkClient(generator, deterministic_keys);
        }

        // Exercise the production entropy source separately from deterministic tests.
        std::set<std::string> random_keys;
        for (int client = 0; client < 4; ++client) {
            std::random_device generator;
            checkClient(generator, random_keys);
        }
        std::cout << "PASS: 1024 deterministic and 1024 OS-random keys; "
                     "16-byte decoding, per-byte variation, no duplicate keys\n";
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
