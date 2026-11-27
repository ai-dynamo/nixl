/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils/memos_manifest.h"
#include <filesystem>
#include <iostream>
#include <iterator>
#include <thread>
#include <vector>

namespace {
void
check(bool value) {
    if (!value) {
        throw std::runtime_error("MEMOS manifest assertion failed");
    }
}

std::string
read(const std::filesystem::path &path) {
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
} // namespace

int
main() {
    namespace fs = std::filesystem;
    std::string root = (fs::temp_directory_path() / "nixlbench-manifest-test-XXXXXX").string();
    if (::mkdtemp(root.data()) == nullptr) {
        return 1;
    }
    try {
        const fs::path previous = fs::path(root) / "memos-keys.txt";
        {
            std::ofstream old(previous);
            old << "previous-run\n";
        }
        std::string complete;
        {
            nixlbench::MemosKeyManifest manifest(root);
            complete = manifest.path();
            manifest.append("first-key");
            check(!fs::exists(complete));
            manifest.publish();
            check(read(complete) == "first-key\n");
        }
        std::string failed;
        try {
            nixlbench::MemosKeyManifest manifest(root);
            failed = manifest.path();
            manifest.append("incomplete-key");
            throw std::runtime_error("simulated registration failure");
        }
        catch (const std::runtime_error &) {
        }
        check(!fs::exists(fs::path(failed).parent_path()));
        // Atomic publication failure also cleans the private partial output.
        {
            std::string unpublished;
            try {
                nixlbench::MemosKeyManifest manifest(root);
                unpublished = manifest.path();
                manifest.append("unpublished-key");
                fs::remove(unpublished + ".tmp");
                manifest.publish();
                check(false);
            }
            catch (const std::runtime_error &) {
            }
            check(!fs::exists(fs::path(unpublished).parent_path()));
        }
        std::vector<std::string> paths(8);
        std::vector<std::thread> workers;
        for (size_t i = 0; i < paths.size(); ++i) {
            workers.emplace_back([&, i] {
                nixlbench::MemosKeyManifest manifest(root);
                paths[i] = manifest.path();
                manifest.append(std::to_string(i));
                manifest.publish();
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }
        for (size_t i = 0; i < paths.size(); ++i) {
            check(read(paths[i]) == std::to_string(i) + "\n");
            for (size_t j = 0; j < i; ++j) {
                check(paths[i] != paths[j]);
            }
        }
        check(read(previous) == "previous-run\n" && read(complete) == "first-key\n");
        fs::remove_all(root);
        std::cout
            << "MEMOS manifest tests passed: prior evidence, failure cleanup, concurrent runs\n";
    }
    catch (const std::exception &error) {
        fs::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
