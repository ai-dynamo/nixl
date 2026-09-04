/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <atomic>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/globals.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"

#include "common.h"
#include "common/nixl_log.h"

namespace {

using testing::HasSubstr;

/* Counts what Abseil hands to a sink other than the file sink, which is how the
 * tests below check that adding a file does not displace existing output. */
class countingSink : public absl::LogSink {
public:
    /** @brief Registers with Abseil, so it starts observing immediately. */
    countingSink() {
        absl::AddLogSink(this);
    }

    /** @brief Unregisters, so the sink cannot outlive its registration. */
    ~countingSink() override {
        absl::RemoveLogSink(this);
    }

    /**
     * @brief Records that Abseil delivered @p entry here as well.
     *
     * Stores the bare message rather than the prefixed form, so assertions do
     * not depend on timestamps.
     */
    void
    Send(const absl::LogEntry &entry) override {
        text_.append(std::string(entry.text_message())).append("\n");
        ++count_;
    }

    /** @brief Number of records this sink has received. */
    size_t
    count() const {
        return count_;
    }

    /** @brief Concatenated messages received so far, one per line. */
    const std::string &
    text() const {
        return text_;
    }

private:
    std::atomic<size_t> count_{0};
    std::string text_;
};

class nixlLogFileTest : public testing::Test {
protected:
    /**
     * @brief Gives each test a clean sink, a known log level and its own path.
     *
     * Raises the level to INFO because most tests log at INFO, which the
     * default WARN would discard before any sink is consulted.
     */
    void
    SetUp() override {
        /* The process may already have a sink from its own pre-main
         * initialization; drop it so each test starts from a known state. */
        nixl::shutdownLogFile();

        prev_min_level_ = absl::MinLogLevel();
        prev_stderr_threshold_ = absl::StderrThreshold();

        /* Most tests log at INFO, which the default WARN level would discard
         * before any sink is consulted. Keep stderr quiet so a passing run does
         * not bury the real test output in deliberate log records. */
        absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
        absl::SetStderrThreshold(absl::LogSeverityAtLeast::kError);

        path_ = std::filesystem::temp_directory_path() /
            ("nixl-log-" + std::to_string(::getpid()) + "-" +
             testing::UnitTest::GetInstance()->current_test_info()->name() + ".log");
        std::filesystem::remove(path_);
    }

    /**
     * @brief Unregisters the sink, restores the log levels and deletes the file.
     *
     * Runs even when a test fails, so one failure cannot leave a sink pointing
     * at a file the next test is about to remove.
     */
    void
    TearDown() override {
        nixl::shutdownLogFile();
        absl::SetMinLogLevel(prev_min_level_);
        absl::SetStderrThreshold(prev_stderr_threshold_);
        std::filesystem::remove(path_);
    }

    /**
     * @brief Points NIXL_LOG_FILE at this test's scratch file and registers it.
     * @return Whatever nixl::initLogFile() reported, so a test can assert on it.
     */
    bool
    enableLogFile() {
        env_.addVar("NIXL_LOG_FILE", path_.string());
        return nixl::initLogFile();
    }

    /**
     * @brief Reads the whole log file.
     * @return Its contents, or an empty string if it does not exist.
     */
    std::string
    readLogFile() const {
        std::ifstream file(path_);
        std::ostringstream contents;
        contents << file.rdbuf();
        return contents.str();
    }

    /**
     * @brief Splits the log file into lines, dropping the trailing newline.
     * @return One entry per record, which lets a test count records and check
     *         that none was torn across a line boundary.
     */
    std::vector<std::string>
    readLogLines() const {
        std::ifstream file(path_);
        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);) {
            lines.push_back(line);
        }
        return lines;
    }

    /**
     * @brief Reports whether the log file was created at all.
     * @return true if the path exists, used to prove the disabled paths create
     *         nothing rather than an empty file.
     */
    bool
    logFileExists() const {
        return std::filesystem::exists(path_);
    }

    std::filesystem::path path_;
    gtest::ScopedEnv env_;

private:
    absl::LogSeverityAtLeast prev_min_level_ = absl::LogSeverityAtLeast::kInfo;
    absl::LogSeverityAtLeast prev_stderr_threshold_ = absl::LogSeverityAtLeast::kInfo;
};

/** @brief The base case: a record emitted with the sink registered reaches the file. */
TEST_F(nixlLogFileTest, WritesRecordToFile) {
    ASSERT_TRUE(enableLogFile());

    NIXL_INFO << "a record for the file";

    EXPECT_THAT(readLogFile(), HasSubstr("a record for the file"));
}

/**
 * @brief A file line carries the same prefix Abseil puts on stderr.
 *
 * This is what lets a file line be matched against the surrounding console
 * output: the severity letter first, then the source site.
 */
TEST_F(nixlLogFileTest, RecordCarriesSeverityAndSourceLocation) {
    ASSERT_TRUE(enableLogFile());

    NIXL_INFO << "located record";

    const auto lines = readLogLines();
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0][0], 'I');
    EXPECT_THAT(lines[0], HasSubstr("nixl_log_file_test.cpp:"));
    EXPECT_THAT(lines[0], HasSubstr("located record"));
}

/** @brief Records appear one per line, in the order they were emitted. */
TEST_F(nixlLogFileTest, EachRecordIsOneLine) {
    ASSERT_TRUE(enableLogFile());

    NIXL_INFO << "first";
    NIXL_INFO << "second";
    NIXL_INFO << "third";

    const auto lines = readLogLines();
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_THAT(lines[0], HasSubstr("first"));
    EXPECT_THAT(lines[1], HasSubstr("second"));
    EXPECT_THAT(lines[2], HasSubstr("third"));
}

/**
 * @brief The file supplements stderr instead of diverting it.
 *
 * The central compatibility claim of the feature: existing tooling that scrapes
 * a process's console must see exactly what it saw before.
 */
TEST_F(nixlLogFileTest, AddsToStderrRatherThanReplacingIt) {
    /* Abseil writes to stderr from its own default handler rather than through
     * a sink, so watch the real thing. */
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    ASSERT_TRUE(enableLogFile());

    testing::internal::CaptureStderr();
    NIXL_INFO << "record for both outputs";
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_THAT(captured, HasSubstr("record for both outputs"));
    EXPECT_THAT(readLogFile(), HasSubstr("record for both outputs"));
}

/** @brief Registering the file sink does not displace other registered sinks. */
TEST_F(nixlLogFileTest, LeavesOtherSinksUntouched) {
    countingSink other;
    ASSERT_TRUE(enableLogFile());

    NIXL_INFO << "record for every sink";

    EXPECT_EQ(other.count(), 1u);
    EXPECT_THAT(other.text(), HasSubstr("record for every sink"));
    EXPECT_THAT(readLogFile(), HasSubstr("record for every sink"));
}

/**
 * @brief NIXL_LOG_LEVEL governs the file exactly as it governs stderr.
 *
 * The level gates a record before any sink is consulted, so the file needs no
 * filtering of its own and cannot drift from what stderr would have shown.
 */
TEST_F(nixlLogFileTest, HonoursLogLevel) {
    const gtest::LogIgnoreGuard lig("warning that should be written");
    ASSERT_TRUE(enableLogFile());

    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
    NIXL_INFO << "info that should be dropped";
    NIXL_WARN << "warning that should be written";

    const std::string contents = readLogFile();
    EXPECT_THAT(contents, HasSubstr("warning that should be written"));
    EXPECT_THAT(contents, testing::Not(HasSubstr("info that should be dropped")));
}

/** @brief With the variable unset, no sink is registered and no file is created. */
TEST_F(nixlLogFileTest, DisabledWhenEnvVarUnset) {
    /* Deliberately no enableLogFile(). */
    env_.addVar("NIXL_LOG_FILE", "");
    ::unsetenv("NIXL_LOG_FILE");

    EXPECT_FALSE(nixl::initLogFile());

    NIXL_INFO << "record with no file configured";
    EXPECT_FALSE(logFileExists());
}

/**
 * @brief An empty value is treated as unset rather than as a filename.
 *
 * Matters because exporting a variable with no value is easy to do by accident
 * in a shell script or container spec.
 */
TEST_F(nixlLogFileTest, DisabledWhenEnvVarEmpty) {
    env_.addVar("NIXL_LOG_FILE", "");

    EXPECT_FALSE(nixl::initLogFile());

    NIXL_INFO << "record with an empty path";
    EXPECT_FALSE(logFileExists());
}

/**
 * @brief A path that cannot be opened degrades to no file, not to a failure.
 *
 * A log file we could not open must not take the process, or the rest of
 * logging, down with it.
 */
TEST_F(nixlLogFileTest, UnopenablePathIsNotFatal) {
    const gtest::LogIgnoreGuard lig("Could not open NIXL_LOG_FILE");
    const auto bad = std::filesystem::temp_directory_path() / "nixl-no-such-dir" / "x.log";
    env_.addVar("NIXL_LOG_FILE", bad.string());

    EXPECT_FALSE(nixl::initLogFile());

    countingSink other;
    NIXL_INFO << "logging still works";
    EXPECT_EQ(other.count(), 1u);
    EXPECT_FALSE(std::filesystem::exists(bad));
}

/** @brief Repeated init calls leave a single registration, so records are not duplicated. */
TEST_F(nixlLogFileTest, InitIsIdempotent) {
    ASSERT_TRUE(enableLogFile());
    EXPECT_TRUE(nixl::initLogFile());
    EXPECT_TRUE(nixl::initLogFile());

    NIXL_INFO << "written once";

    /* A sink registered twice would duplicate every record. */
    const auto lines = readLogLines();
    EXPECT_EQ(lines.size(), 1u);
}

/**
 * @brief Shutdown really unregisters, and a second call is harmless.
 *
 * Both halves matter: the destructor-attribute hook may run after a caller has
 * already shut the sink down explicitly.
 */
TEST_F(nixlLogFileTest, ShutdownStopsWritingAndIsIdempotent) {
    ASSERT_TRUE(enableLogFile());
    NIXL_INFO << "before shutdown";

    nixl::shutdownLogFile();
    nixl::shutdownLogFile();

    NIXL_INFO << "after shutdown";

    const std::string contents = readLogFile();
    EXPECT_THAT(contents, HasSubstr("before shutdown"));
    EXPECT_THAT(contents, testing::Not(HasSubstr("after shutdown")));
}

/**
 * @brief Reopening the same path appends instead of truncating.
 *
 * A restarted process should add to the record rather than erase what the
 * previous one reported.
 */
TEST_F(nixlLogFileTest, AppendsAcrossSessions) {
    ASSERT_TRUE(enableLogFile());
    NIXL_INFO << "from the first session";
    nixl::shutdownLogFile();

    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "from the second session";

    const std::string contents = readLogFile();
    EXPECT_THAT(contents, HasSubstr("from the first session"));
    EXPECT_THAT(contents, HasSubstr("from the second session"));
}

/**
 * @brief Each record is durable as soon as it is logged.
 *
 * Read back while the sink is still registered and without an explicit flush. A
 * process that crashes or hangs never reaches shutdown, so a buffered record
 * would be lost exactly when the log matters most.
 */
TEST_F(nixlLogFileTest, RecordsAreReadableWithoutWaitingForShutdown) {
    ASSERT_TRUE(enableLogFile());

    NIXL_INFO << "readable immediately";

    EXPECT_THAT(readLogFile(), HasSubstr("readable immediately"));
}

/**
 * @brief Concurrent writers produce whole lines, never torn ones.
 *
 * Abseil holds only a reader lock while dispatching to sinks, so Send() runs
 * concurrently and the sink must serialize writes itself.
 */
TEST_F(nixlLogFileTest, ConcurrentRecordsAreNotInterleaved) {
    ASSERT_TRUE(enableLogFile());

    constexpr unsigned num_threads = 8;
    constexpr unsigned per_thread = 50;

    std::vector<std::thread> threads;
    for (unsigned t = 0; t < num_threads; ++t) {
        threads.emplace_back([t]() {
            for (unsigned i = 0; i < per_thread; ++i) {
                NIXL_INFO << "payload " << t << ":" << i;
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    const auto lines = readLogLines();
    ASSERT_EQ(lines.size(), num_threads * per_thread);

    /* Every line must be a whole record. A torn or interleaved write would
     * leave a line that does not end in its own payload. */
    const std::regex record("^I.*payload [0-7]:[0-9]+$");
    for (const auto &line : lines) {
        EXPECT_TRUE(std::regex_match(line, record)) << "malformed line: " << line;
    }
}

} // namespace
