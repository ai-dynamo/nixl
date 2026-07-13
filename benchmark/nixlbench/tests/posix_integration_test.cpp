/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace nixlbench {
namespace {

    class TemporaryDirectory {
    public:
        TemporaryDirectory() {
            const auto base = std::filesystem::temp_directory_path() / "nixlbench-pr1-XXXXXX";
            std::string pattern = base.string();
            pattern.push_back('\0');
            char *created = mkdtemp(pattern.data());
            if (created != nullptr) {
                path_ = created;
            }
        }

        ~TemporaryDirectory() {
            std::error_code error;
            std::filesystem::remove_all(path_, error);
        }

        const std::filesystem::path &
        path() const {
            return path_;
        }

    private:
        std::filesystem::path path_;
    };

    std::string
    shellQuote(const std::string &value) {
        std::string quoted = "'";
        for (const char ch : value) {
            if (ch == '\'') {
                quoted += "'\\''";
            } else {
                quoted += ch;
            }
        }
        return quoted + "'";
    }

    int
    runCommand(const std::string &arguments, const std::filesystem::path &log) {
        const char *binary = std::getenv("NIXLBENCH_BINARY");
        if (binary == nullptr) {
            return -1;
        }
        const std::string command =
            shellQuote(binary) + " " + arguments + " >" + shellQuote(log.string()) + " 2>&1";
        const int status = std::system(command.c_str());
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }

    std::string
    smallRawCommand(const std::filesystem::path &path, const std::string &operation) {
        return "raw posix --path " + shellQuote(path.string()) + " --operation " + operation +
            " --total-buffer-size 64KiB --start-block-size 4KiB --max-block-size 4KiB "
            "--iterations 16 --warmup-iterations 0 --check-consistency";
    }

    std::string
    smallLegacyCommand(const std::filesystem::path &path) {
        return "--backend=POSIX --filepath=" + shellQuote(path.string()) +
            " --op_type=WRITE --total_buffer_size=65536 --start_block_size=4096 "
            "--max_block_size=4096 --num_iter=16 --warmup_iter=0 --check_consistency=true";
    }

    size_t
    regularFileCount(const std::filesystem::path &path) {
        return static_cast<size_t>(
            std::count_if(std::filesystem::directory_iterator(path),
                          std::filesystem::directory_iterator(),
                          [](const auto &entry) { return entry.is_regular_file(); }));
    }

    TEST(PosixIntegrationTest, DryRunDoesNotCreateFiles) {
        TemporaryDirectory directory;
        ASSERT_FALSE(directory.path().empty());
        const auto log = directory.path() / "dry-run.log";
        EXPECT_EQ(runCommand(smallRawCommand(directory.path(), "write") + " --dry-run", log), 0);
        EXPECT_EQ(regularFileCount(directory.path()), 1U); // log only

        std::ifstream stream(log);
        const std::string contents((std::istreambuf_iterator<char>(stream)), {});
        EXPECT_NE(contents.find("Dry run: no worker was created"), std::string::npos);
    }

    TEST(PosixIntegrationTest, RawWriteAndFreshReadPassConsistencyChecks) {
        TemporaryDirectory write_directory;
        TemporaryDirectory read_directory;
        ASSERT_EQ(runCommand(smallRawCommand(write_directory.path(), "write"),
                             write_directory.path() / "write.log"),
                  0);
        ASSERT_EQ(runCommand(smallRawCommand(read_directory.path(), "read"),
                             read_directory.path() / "read.log"),
                  0);
        EXPECT_GE(regularFileCount(write_directory.path()), 2U);
        EXPECT_GE(regularFileCount(read_directory.path()), 2U);
    }

    TEST(PosixIntegrationTest, LegacyAndRawEquivalentWriteConfigurationsBothPass) {
        TemporaryDirectory legacy_directory;
        TemporaryDirectory raw_directory;
        EXPECT_EQ(runCommand(smallLegacyCommand(legacy_directory.path()),
                             legacy_directory.path() / "legacy.log"),
                  0);
        EXPECT_EQ(runCommand(smallRawCommand(raw_directory.path(), "write"),
                             raw_directory.path() / "raw.log"),
                  0);
    }

    TEST(PosixIntegrationTest, InvalidOptionAndOpenFailureLeaveNoBenchmarkFiles) {
        TemporaryDirectory invalid_directory;
        const auto invalid_log = invalid_directory.path() / "invalid.log";
        EXPECT_NE(
            runCommand(smallRawCommand(invalid_directory.path(), "write") + " --gds-batch-limit 4",
                       invalid_log),
            0);
        EXPECT_EQ(regularFileCount(invalid_directory.path()), 1U);

        TemporaryDirectory failure_directory;
        const auto failure_log = failure_directory.path() / "failure.log";
        const auto missing_file = failure_directory.path() / "missing" / "file";
        const std::string command = "raw posix --filenames " + shellQuote(missing_file.string()) +
            " --total-buffer-size 64KiB --start-block-size 4KiB --max-block-size 4KiB "
            "--iterations 16 --warmup-iterations 0";
        EXPECT_NE(runCommand(command, failure_log), 0);
        EXPECT_FALSE(std::filesystem::exists(missing_file));
        EXPECT_EQ(regularFileCount(failure_directory.path()), 1U);
    }

} // namespace
} // namespace nixlbench
