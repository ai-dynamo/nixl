/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "utils/utils.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace nixlbench {
namespace {

    class scenarioTemporaryDirectory {
    public:
        scenarioTemporaryDirectory() {
            const auto base = std::filesystem::temp_directory_path() / "nixlbench-pr2-XXXXXX";
            std::string pattern = base.string();
            pattern.push_back('\0');
            char *created = mkdtemp(pattern.data());
            if (created != nullptr) {
                path_ = created;
            }
        }

        ~scenarioTemporaryDirectory() {
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
    scenarioShellQuote(const std::string &value) {
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
    runScenarioCommand(const std::string &arguments,
                       const std::filesystem::path &log,
                       const std::filesystem::path &plugin_directory = {}) {
        const char *binary = std::getenv("NIXLBENCH_BINARY");
        if (binary == nullptr) {
            return -1;
        }
        const std::string plugin_environment = plugin_directory.empty() ?
            "" :
            "NIXL_PLUGIN_DIR=" + scenarioShellQuote(plugin_directory.string()) + " ";
        const std::string command = plugin_environment + scenarioShellQuote(binary) + " " +
            arguments + " >" + scenarioShellQuote(log.string()) + " 2>&1";
        const int status = std::system(command.c_str());
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }

    std::string
    readScenarioLog(const std::filesystem::path &path) {
        std::ifstream stream(path);
        return {(std::istreambuf_iterator<char>(stream)), {}};
    }

    std::string
    smallAllocateOnceCommand(const std::filesystem::path &path, const std::string &operation) {
        return "scenario allocate-once posix --path " + scenarioShellQuote(path.string()) +
            " --file-size 64KB --block-size 4KB --batch-size 2 --iterations 4 "
            "--warmup-iterations 1 --operation " +
            operation + " --initiator-memory dram --check-consistency";
    }

    TEST(AllocateOnceIntegrationTest, DiscoveryReportsUnreadablePluginMetadata) {
        scenarioTemporaryDirectory directory;
        ASSERT_FALSE(directory.path().empty());
        const auto log = directory.path() / "help.log";
        const char *plugin_directory = std::getenv("NIXL_PLUGIN_DIR");
        ASSERT_NE(plugin_directory, nullptr);
        const auto posix_plugin = std::filesystem::path(plugin_directory) / "libplugin_POSIX.so";
        ASSERT_TRUE(std::filesystem::exists(posix_plugin));
        std::filesystem::create_symlink(posix_plugin, directory.path() / posix_plugin.filename());
        std::ofstream(directory.path() / "libplugin_BROKEN.so");

        EXPECT_NE(runScenarioCommand("scenario allocate-once --help", log, directory.path()), 0);
        const auto contents = readScenarioLog(log);
        EXPECT_NE(contents.find("failed to query BROKEN plugin metadata"), std::string::npos);
    }

    TEST(AllocateOnceIntegrationTest, DryRunDoesNotCreateOrOpenManagedFiles) {
        scenarioTemporaryDirectory directory;
        ASSERT_FALSE(directory.path().empty());
        const auto log = directory.path() / "dry-run.log";
        const auto dataset = directory.path() / "nixlbench_allocate_once_0.dat";

        EXPECT_EQ(runScenarioCommand(
                      smallAllocateOnceCommand(directory.path(), "write") + " --dry-run", log),
                  0);
        EXPECT_FALSE(std::filesystem::exists(dataset));
        const auto contents = readScenarioLog(log);
        EXPECT_NE(contents.find("Dry run: no backing file was opened"), std::string::npos);
    }

    TEST(AllocateOnceIntegrationTest, ManagedDatasetIsCreatedOnceAndReusedForRead) {
        scenarioTemporaryDirectory directory;
        ASSERT_FALSE(directory.path().empty());
        const auto write_log = directory.path() / "write.log";
        const auto read_log = directory.path() / "read.log";
        const auto first_dataset = directory.path() / "nixlbench_allocate_once_0.dat";
        const auto second_dataset = directory.path() / "nixlbench_allocate_once_1.dat";
        const std::string write_command =
            smallAllocateOnceCommand(directory.path(), "write") + " --threads 2 --num-files 2";
        const std::string read_command =
            smallAllocateOnceCommand(directory.path(), "read") + " --threads 2 --num-files 2";

        ASSERT_EQ(runScenarioCommand(write_command, write_log), 0) << readScenarioLog(write_log);
        EXPECT_NE(readScenarioLog(write_log).find(
                      "POSIX backend with plugin parameters from scenario CLI"),
                  std::string::npos);
        ASSERT_TRUE(std::filesystem::exists(first_dataset));
        ASSERT_TRUE(std::filesystem::exists(second_dataset));
        EXPECT_EQ(std::filesystem::file_size(first_dataset), 64U * 1024U);
        EXPECT_EQ(std::filesystem::file_size(second_dataset), 64U * 1024U);

        ASSERT_EQ(runScenarioCommand(read_command, read_log), 0) << readScenarioLog(read_log);
        EXPECT_EQ(std::filesystem::file_size(first_dataset), 64U * 1024U);
        EXPECT_EQ(std::filesystem::file_size(second_dataset), 64U * 1024U);
    }

    TEST(AllocateOnceIntegrationTest, ManagedWrongSizeFileIsReinitializedToTheRequestedSize) {
        scenarioTemporaryDirectory directory;
        ASSERT_FALSE(directory.path().empty());
        const auto dataset = directory.path() / "nixlbench_allocate_once_0.dat";
        {
            std::ofstream stream(dataset);
            stream << "short";
        }
        const auto log = directory.path() / "resize.log";

        ASSERT_EQ(runScenarioCommand(smallAllocateOnceCommand(directory.path(), "write"), log), 0)
            << readScenarioLog(log);
        EXPECT_EQ(std::filesystem::file_size(dataset), 64U * 1024U);
    }

    TEST(AllocateOnceIntegrationTest, ExplicitFileOwnershipAndIdentityArePreserved) {
        scenarioTemporaryDirectory directory;
        ASSERT_FALSE(directory.path().empty());
        const auto missing = directory.path() / "missing.dat";
        const auto missing_log = directory.path() / "missing.log";
        const std::string prefix =
            "scenario allocate-once --file-size 64KB --block-size 4KB posix --filenames ";

        EXPECT_NE(runScenarioCommand(prefix + scenarioShellQuote(missing.string()), missing_log),
                  0);
        EXPECT_FALSE(std::filesystem::exists(missing));

        const auto short_file = directory.path() / "short.dat";
        {
            std::ofstream stream(short_file);
            stream << "short";
        }
        const auto short_log = directory.path() / "short.log";
        EXPECT_NE(runScenarioCommand(prefix + scenarioShellQuote(short_file.string()), short_log),
                  0);
        EXPECT_EQ(std::filesystem::file_size(short_file), 5U);

        const auto existing_file = directory.path() / "existing.dat";
        {
            std::ofstream stream(existing_file);
            stream.seekp(64 * 1024 - 1);
            stream.put('\0');
        }
        const auto existing_log = directory.path() / "existing.log";
        const std::string read_command = prefix + scenarioShellQuote(existing_file.string()) +
            " --operation read --iterations 2 --warmup-iterations 0 --initiator-memory dram";
        EXPECT_EQ(runScenarioCommand(read_command, existing_log), 0)
            << readScenarioLog(existing_log);
        EXPECT_EQ(std::filesystem::file_size(existing_file), 64U * 1024U);

        const auto write_log = directory.path() / "write.log";
        const std::string write_command = prefix + scenarioShellQuote(existing_file.string()) +
            " --operation write --offset-mode sequential --batch-size 2 --iterations 2 "
            "--warmup-iterations 0 --initiator-memory dram";
        ASSERT_EQ(runScenarioCommand(write_command, write_log), 0) << readScenarioLog(write_log);
        std::ifstream existing_stream(existing_file, std::ios::binary);
        const std::vector<unsigned char> contents(std::istreambuf_iterator<char>(existing_stream),
                                                  {});
        ASSERT_EQ(contents.size(), 64U * 1024U);
        EXPECT_TRUE(
            std::all_of(contents.begin(), contents.begin() + 16 * 1024, [](unsigned char byte) {
                return byte == XFERBENCH_INITIATOR_BUFFER_ELEMENT;
            }));
        EXPECT_TRUE(std::all_of(contents.begin() + 16 * 1024,
                                contents.end(),
                                [](unsigned char byte) { return byte == 0; }));

        const auto alias_file = directory.path() / "same-file.dat";
        std::filesystem::create_hard_link(existing_file, alias_file);
        const auto duplicate_log = directory.path() / "duplicate.log";
        const std::string duplicate_files = existing_file.string() + "," + alias_file.string();
        const std::string duplicate_command =
            "scenario allocate-once --file-size 64KB --block-size 4KB --threads 2 "
            "posix --num-files 2 --filenames " +
            scenarioShellQuote(duplicate_files) +
            " --operation read --iterations 2 --warmup-iterations 0 --initiator-memory dram";
        EXPECT_NE(runScenarioCommand(duplicate_command, duplicate_log), 0);
        const auto duplicate_output = readScenarioLog(duplicate_log);
        EXPECT_NE(duplicate_output.find("must refer to distinct files"), std::string::npos)
            << duplicate_output;
    }

    TEST(AllocateOnceIntegrationTest, ManagedFilesDoNotFollowExistingSymbolicLinks) {
        scenarioTemporaryDirectory directory;
        ASSERT_FALSE(directory.path().empty());
        const auto victim = directory.path() / "victim.dat";
        {
            std::ofstream stream(victim);
            stream << "do not modify";
        }
        const auto original_size = std::filesystem::file_size(victim);
        const auto managed_name = directory.path() / "nixlbench_allocate_once_0.dat";
        std::filesystem::create_symlink(victim, managed_name);
        const auto log = directory.path() / "symlink.log";

        EXPECT_NE(runScenarioCommand(smallAllocateOnceCommand(directory.path(), "write"), log), 0);
        EXPECT_EQ(std::filesystem::file_size(victim), original_size);
    }

} // namespace
} // namespace nixlbench
