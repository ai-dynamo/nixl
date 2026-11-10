// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include "utils/scope_guard.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#include <sys/wait.h>
#include <unistd.h>

namespace {
std::string
shellQuote(const std::string &value) {
    std::string result = "'";
    for (const char c : value) {
        result += c == '\'' ? "'\\''" : std::string(1, c);
    }
    return result + "'";
}

class wqsIntegrationTest : public ::testing::TestWithParam<std::tuple<std::string, bool, bool>> {};

TEST_P(wqsIntegrationTest, SweepsWholeObjectsAndChecksData) {
    const char *enabled = std::getenv("NIXLBENCH_WQS_MOCK_TEST");
    if (enabled == nullptr || std::string(enabled) != "1") {
        GTEST_SKIP() << "Requires an installed WQS mock SDK and 2 MiB of hugepages";
    }
    const char *binary = std::getenv("NIXLBENCH_BINARY");
    ASSERT_NE(binary, nullptr);
    char path[] = "/tmp/nixlbench-wqs-XXXXXX";
    const int fd = mkstemp(path);
    ASSERT_GE(fd, 0);
    close(fd);
    const auto cleanup = make_scope_guard([&] { unlink(path); });
    const auto &[operation, progress, reregister] = GetParam();
    const std::string command = shellQuote(binary) + " --backend=WQS --op_type=" + operation +
        " --wqs_unix_path=/unused/mock-socket --wqs_staging_pool_bytes=2097152"
        " --wqs_max_batch_entries=2 --wqs_max_inflight=4 --wqs_key_prefix=nixlbench-test/"
        " --total_buffer_size=65536 --start_block_size=17 --max_block_size=68"
        " --start_batch_size=1 --max_batch_size=4 --num_iter=8 --warmup_iter=0"
        " --num_threads=2 --pipeline_depth=2 --check_consistency"
        " --randomize_location_mode=blockaligned --enable_pt=" +
        (progress ? "true" : "false") + (reregister ? " --reregister_mem" : " --prepared_xfer") +
        " >" + shellQuote(path) + " 2>&1";
    const int status = std::system(command.c_str());
    std::ifstream log(path);
    const std::string contents{std::istreambuf_iterator<char>(log), {}};
    ASSERT_TRUE(WIFEXITED(status)) << contents;
    EXPECT_EQ(WEXITSTATUS(status), 0) << contents;
}

INSTANTIATE_TEST_SUITE_P(OperationsAndThreadModes,
                         wqsIntegrationTest,
                         ::testing::Combine(::testing::Values("READ", "WRITE"),
                                            ::testing::Bool(),
                                            ::testing::Bool()));
} // namespace
