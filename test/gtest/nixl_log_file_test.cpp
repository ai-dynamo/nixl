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

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <string_view>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
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

// POSIX leaves this to the application to declare, and glibc only exposes it
// from unistd.h under _GNU_SOURCE, so declare it rather than depend on which
// feature macros happen to be set. At global scope, not in the namespace below,
// so it refers to the process environment and not a new internal symbol.
extern "C" char **environ;

namespace {

using testing::HasSubstr;

// Set by the child process in RecordsFromStaticDestructorsReachTheFile, so the
// destructor below only logs in the one process that is testing for it.
constexpr const char *late_record_env_var = "NIXL_TEST_LATE_RECORD";
constexpr const char *late_record_text = "record from a static destructor";

/**
 * @brief Logs from a static destructor, to check the file outlives teardown.
 *
 * At file scope, so it is destroyed near the end of the exit sequence: if the
 * sink is still registered here, it was for every earlier static destructor.
 */
struct lateLogger {
    ~lateLogger() {
        if (std::getenv(late_record_env_var) != nullptr) {
            NIXL_INFO << late_record_text;
        }
    }
};

lateLogger late_logger;

/** @brief Counts what Abseil hands to other sinks, to show none is displaced. */
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
     * @brief Records @p entry. Stores the bare message, so assertions do not
     *        depend on timestamps. Serialized: Abseil calls this from whichever
     *        thread logged.
     */
    void
    Send(const absl::LogEntry &entry) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        text_.append(std::string(entry.text_message())).append("\n");
    }

    /** @brief Records containing @p marker, so another thread cannot inflate a count. */
    size_t
    countMatching(std::string_view marker) const {
        const std::lock_guard<std::mutex> lock(mutex_);

        size_t matching = 0;
        for (size_t at = text_.find(marker); at != std::string::npos;
             at = text_.find(marker, at + 1)) {
            ++matching;
        }
        return matching;
    }

    /** @brief Messages so far, one per line. A copy: Send() may be appending. */
    std::string
    text() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return text_;
    }

private:
    mutable std::mutex mutex_;
    std::string text_;
};

/** @brief Fixture for the NIXL_LOG_FILE tests; see SetUp() for the isolation it gives. */
class nixlLogFileTest : public testing::Test {
protected:
    /** @brief Gives each test a clean sink, a known level and its own path. */
    void
    SetUp() override {
        // The process may already have a sink from its pre-main setup.
        nixl::shutdownLogFile();

        prevMinLevel_ = absl::MinLogLevel();
        prevStderrThreshold_ = absl::StderrThreshold();

        // INFO because most tests log there; stderr quiet so a passing run is
        // not buried in deliberate records.
        absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
        absl::SetStderrThreshold(absl::LogSeverityAtLeast::kError);

        path_ = std::filesystem::temp_directory_path() /
            ("nixl-log-" + std::to_string(::getpid()) + "-" +
             testing::UnitTest::GetInstance()->current_test_info()->name() + ".log");
        std::filesystem::remove(path_);
    }

    /** @brief Undoes SetUp, even on failure, so no sink outlives its test. */
    void
    TearDown() override {
        nixl::shutdownLogFile();
        absl::SetMinLogLevel(prevMinLevel_);
        absl::SetStderrThreshold(prevStderrThreshold_);
        std::filesystem::remove(path_);
    }

    /** @brief Points NIXL_LOG_FILE at this test's file; returns what init said. */
    bool
    enableLogFile() {
        env_.addVar("NIXL_LOG_FILE", path_.string());
        return nixl::initLogFile();
    }

    /** @brief The log file's contents, or empty if it does not exist. */
    std::string
    readLogFile() const {
        std::ifstream file(path_);
        std::ostringstream contents;
        contents << file.rdbuf();
        return contents.str();
    }

    /** @brief One entry per record, for counting and checking none was torn. */
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
     * @brief The log file's lines containing @p marker. Selecting on the test's
     *        own text keeps another thread's record out of an exact count.
     */
    std::vector<std::string>
    linesMatching(std::string_view marker) const {
        std::vector<std::string> matching;
        for (const auto &line : readLogLines()) {
            if (line.find(marker) != std::string::npos) {
                matching.push_back(line);
            }
        }
        return matching;
    }

    /** @brief Whether the file exists, to show a disabled path creates nothing. */
    bool
    logFileExists() const {
        return std::filesystem::exists(path_);
    }

    std::filesystem::path path_;
    gtest::ScopedEnv env_;

private:
    absl::LogSeverityAtLeast prevMinLevel_ = absl::LogSeverityAtLeast::kInfo;
    absl::LogSeverityAtLeast prevStderrThreshold_ = absl::LogSeverityAtLeast::kInfo;
};

/** @brief The base case: a record emitted with the sink registered reaches the file. */
TEST_F(nixlLogFileTest, WritesRecordToFile) {
    ASSERT_TRUE(enableLogFile());

    NIXL_INFO << "a record for the file";

    EXPECT_THAT(readLogFile(), HasSubstr("a record for the file"));
}

/** @brief A file line carries the same prefix as stderr, so the two can be matched. */
TEST_F(nixlLogFileTest, RecordCarriesSeverityAndSourceLocation) {
    ASSERT_TRUE(enableLogFile());

    NIXL_INFO << "located record";

    const auto lines = linesMatching("located record");
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0][0], 'I');
    EXPECT_THAT(lines[0], HasSubstr("nixl_log_file_test.cpp:"));
}

/** @brief Records appear one per line, in the order they were emitted. */
TEST_F(nixlLogFileTest, EachRecordIsOneLine) {
    ASSERT_TRUE(enableLogFile());

    // A shared prefix, so the three can be selected as a group below.
    NIXL_INFO << "in order: first";
    NIXL_INFO << "in order: second";
    NIXL_INFO << "in order: third";

    const auto lines = linesMatching("in order:");
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_THAT(lines[0], HasSubstr("first"));
    EXPECT_THAT(lines[1], HasSubstr("second"));
    EXPECT_THAT(lines[2], HasSubstr("third"));
}

/**
 * @brief The file supplements stderr rather than diverting it: tooling that
 *        scrapes the console must see what it saw before.
 */
TEST_F(nixlLogFileTest, AddsToStderrRatherThanReplacingIt) {
    // Abseil writes stderr from its default handler, not a sink.
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

    EXPECT_EQ(other.countMatching("record for every sink"), 1u);
    EXPECT_THAT(readLogFile(), HasSubstr("record for every sink"));
}

/**
 * @brief NIXL_LOG_LEVEL governs the file as it does stderr: the level gates a
 *        record before any sink is consulted, so the two cannot drift.
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
    // Deliberately no enableLogFile().
    env_.addVar("NIXL_LOG_FILE", "");
    ::unsetenv("NIXL_LOG_FILE");

    EXPECT_FALSE(nixl::initLogFile());

    NIXL_INFO << "record with no file configured";
    EXPECT_FALSE(logFileExists());
}

/** @brief An empty value is unset, not a filename: easy to export by accident. */
TEST_F(nixlLogFileTest, DisabledWhenEnvVarEmpty) {
    env_.addVar("NIXL_LOG_FILE", "");

    EXPECT_FALSE(nixl::initLogFile());

    NIXL_INFO << "record with an empty path";
    EXPECT_FALSE(logFileExists());
}

/** @brief An unopenable path costs the file, not the process or the rest of logging. */
TEST_F(nixlLogFileTest, UnopenablePathIsNotFatal) {
    const gtest::LogIgnoreGuard lig("Could not open NIXL_LOG_FILE");

    // From path_, which carries the pid and test name, so a concurrent run
    // cannot create the directory and turn the open into a success.
    auto missingDir = path_;
    missingDir += ".missing";
    std::filesystem::remove_all(missingDir);
    ASSERT_FALSE(std::filesystem::exists(missingDir));

    const auto bad = missingDir / "x.log";
    env_.addVar("NIXL_LOG_FILE", bad.string());

    EXPECT_FALSE(nixl::initLogFile());

    countingSink other;
    NIXL_INFO << "logging still works";
    EXPECT_EQ(other.countMatching("logging still works"), 1u);
    EXPECT_FALSE(std::filesystem::exists(bad));
}

/** @brief Repeated init calls leave a single registration, so records are not duplicated. */
TEST_F(nixlLogFileTest, InitIsIdempotent) {
    ASSERT_TRUE(enableLogFile());
    EXPECT_TRUE(nixl::initLogFile());
    EXPECT_TRUE(nixl::initLogFile());

    NIXL_INFO << "written once";

    // A sink registered twice would duplicate every record.
    EXPECT_EQ(linesMatching("written once").size(), 1u);
}

/** @brief Shutdown unregisters, and a second call is harmless: the hook may follow one. */
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

/** @brief Reopening appends: a restart adds to the record rather than erasing it. */
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
 * @brief Each record is durable as soon as it is logged, read back without an
 *        explicit flush: a process that hangs never reaches shutdown.
 */
TEST_F(nixlLogFileTest, RecordsAreReadableWithoutWaitingForShutdown) {
    ASSERT_TRUE(enableLogFile());

    NIXL_INFO << "readable immediately";

    EXPECT_THAT(readLogFile(), HasSubstr("readable immediately"));
}

/** @brief %h and %p expand, so one setting gives every worker its own file. */
TEST_F(nixlLogFileTest, ExpandsHostAndProcessIntoThePath) {
    char host[256] = {};
    ASSERT_EQ(::gethostname(host, sizeof(host) - 1), 0);

    const std::string pattern = path_.string() + "-%h-%p";
    const std::filesystem::path expanded =
        path_.string() + "-" + host + "-" + std::to_string(::getpid());
    std::filesystem::remove(expanded);

    env_.addVar("NIXL_LOG_FILE", pattern);
    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "record for the expanded path";
    nixl::shutdownLogFile();

    EXPECT_TRUE(std::filesystem::exists(expanded)) << "expected " << expanded;
    EXPECT_FALSE(std::filesystem::exists(pattern)) << "the raw pattern must not be used as a name";

    std::filesystem::remove(expanded);
}

/**
 * @brief %t separates runs: ids are recycled and the file is appended to, so a
 *        restart given an earlier id would otherwise continue its file.
 */
TEST_F(nixlLogFileTest, ExpandsTheRunMarkerIntoThePath) {
    const std::string pattern = path_.string() + "-%t";
    const std::string prefix = path_.filename().string() + "-";

    const auto now = [] {
        struct timespec ts {};
        ::clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
    };
    const auto named = [&prefix, this] {
        std::vector<std::string> found;
        for (const auto &entry : std::filesystem::directory_iterator(path_.parent_path())) {
            const std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) == 0) {
                found.push_back(name.substr(prefix.size()));
            }
        }
        return found;
    };

    for (const auto &stale : named()) {
        std::filesystem::remove(path_.parent_path() / (prefix + stale));
    }

    const uint64_t before = now();
    env_.addVar("NIXL_LOG_FILE", pattern);
    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "record for the marked path";
    nixl::shutdownLogFile();
    const uint64_t after = now();

    auto written = named();
    ASSERT_EQ(written.size(), 1u) << "expected exactly one file named for the run marker";
    const uint64_t marker = std::stoull(written.front());
    EXPECT_GE(marker, before) << "the marker predates the call";
    EXPECT_LE(marker, after) << "the marker postdates the call";

    // Sampled once: rebinding in this process must reuse the name, not start a
    // second file.
    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "record after rebinding";
    nixl::shutdownLogFile();

    written = named();
    EXPECT_EQ(written.size(), 1u) << "the run marker moved within one process";

    for (const auto &leftover : written) {
        std::filesystem::remove(path_.parent_path() / (prefix + leftover));
    }
}

/**
 * @brief A fork without exec keeps writing the parent's file.
 *
 * Documented rather than desired: the path is expanded once at library load, so
 * a child inherits the name and the open file. Pinned here so the documented
 * limit cannot quietly stop being true. Workers started through exec, the usual
 * case for GPU work, are unaffected.
 */
TEST_F(nixlLogFileTest, ForkWithoutExecKeepsWritingTheParentsFile) {
    const std::string pattern = path_.string() + "-fork-%p";
    const std::filesystem::path parent_file =
        path_.string() + "-fork-" + std::to_string(::getpid());
    std::filesystem::remove(parent_file);

    env_.addVar("NIXL_LOG_FILE", pattern);
    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "record from the parent";

    const pid_t child = ::fork();
    ASSERT_NE(child, -1) << "fork failed";
    if (child == 0) {
        NIXL_INFO << "record from the child";
        // _exit: the child must not run this process's teardown again.
        ::_exit(0);
    }

    // With a deadline: a child that inherited a held mutex blocks on its first
    // record forever, and an unbounded wait would hang the run rather than fail.
    int status = 0;
    pid_t reaped = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((reaped = ::waitpid(child, &status, WNOHANG)) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (reaped == 0) {
        ::kill(child, SIGKILL);
        ::waitpid(child, &status, 0);
        nixl::shutdownLogFile();
        std::filesystem::remove(parent_file);
        FAIL() << "the forked child never exited; it most likely blocked writing its record";
    }
    ASSERT_EQ(reaped, child) << "waitpid failed";
    nixl::shutdownLogFile();

    const std::filesystem::path child_file = path_.string() + "-fork-" + std::to_string(child);
    EXPECT_FALSE(std::filesystem::exists(child_file))
        << "a child file appeared, so fork now reinitializes: update the documentation";

    std::ifstream in(parent_file);
    std::ostringstream contents;
    contents << in.rdbuf();
    EXPECT_THAT(contents.str(), HasSubstr("record from the parent"));
    EXPECT_THAT(contents.str(), HasSubstr("record from the child"));

    std::filesystem::remove(parent_file);
    std::filesystem::remove(child_file);
}

/** @brief %% is a literal percent, and an unknown escape survives rather than being swallowed. */
TEST_F(nixlLogFileTest, LeavesLiteralAndUnknownEscapesAlone) {
    const std::string pattern = path_.string() + "-%%-%z";
    const std::filesystem::path expanded = path_.string() + "-%-%z";
    std::filesystem::remove(expanded);

    env_.addVar("NIXL_LOG_FILE", pattern);
    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "record for the literal path";
    nixl::shutdownLogFile();

    EXPECT_TRUE(std::filesystem::exists(expanded)) << "expected " << expanded;

    std::filesystem::remove(expanded);
}

/**
 * @brief At the limit the file rotates, keeping the newest records: a log that
 *        answers "what happened before this hung" needs its tail. One
 *        generation is kept, which bounds the total.
 */
TEST_F(nixlLogFileTest, RotatesAtTheLimitAndKeepsTheNewestRecords) {
    constexpr std::uintmax_t limit = 2048;
    const std::filesystem::path rotated = path_.string() + ".1";
    std::filesystem::remove(rotated);

    env_.addVar("NIXL_LOG_FILE", path_.string());
    env_.addVar("NIXL_LOG_FILE_SIZE", "2K");
    ASSERT_TRUE(nixl::initLogFile());

    for (unsigned i = 0; i < 200; ++i) {
        NIXL_INFO << "rotation record " << i;
    }

    ASSERT_TRUE(std::filesystem::exists(rotated)) << "nothing was rotated";
    EXPECT_LE(std::filesystem::file_size(path_), limit) << "the live file outgrew the limit";

    // Newest in the live file, earlier ones in the rotated file.
    EXPECT_THAT(readLogFile(), HasSubstr("rotation record 199"));

    std::ifstream previous(rotated);
    std::ostringstream contents;
    contents << previous.rdbuf();
    EXPECT_THAT(contents.str(), testing::Not(HasSubstr("rotation record 199")));

    // One generation only, so the total on disk stays bounded.
    EXPECT_FALSE(std::filesystem::exists(path_.string() + ".2"));

    std::filesystem::remove(rotated);
}

/**
 * @brief A rotation it cannot do stops the sink, and says so, rather than
 *        ignoring the limit. The file is left alone: those records are all
 *        there will be.
 */
TEST_F(nixlLogFileTest, StopsLoggingWhenItCannotRotate) {
    if (::geteuid() == 0) {
        GTEST_SKIP() << "root bypasses the directory permission this relies on";
    }
    constexpr std::uintmax_t limit = 2048;

    // Writable first so the file can be created, then searchable but not
    // writable: rename needs the directory, writing only needs the file.
    const std::filesystem::path directory = path_.string() + "-norename";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::filesystem::path log = directory / "log";

    env_.addVar("NIXL_LOG_FILE", log.string());
    env_.addVar("NIXL_LOG_FILE_SIZE", "2K");
    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "record that creates the file";

    std::filesystem::permissions(
        directory, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);

    testing::internal::CaptureStderr();
    for (unsigned i = 0; i < 200; ++i) {
        NIXL_INFO << "unrenamable record " << i;
    }
    const std::string captured = testing::internal::GetCapturedStderr();

    // Reported, once, and the limit still holds.
    const std::string report = "could not rotate NIXL_LOG_FILE";
    size_t reports = 0;
    for (size_t at = captured.find(report); at != std::string::npos;
         at = captured.find(report, at + 1)) {
        ++reports;
    }
    EXPECT_EQ(reports, 1u) << "stderr was:\n" << captured;
    EXPECT_LE(std::filesystem::file_size(log), limit) << "the limit was abandoned";
    EXPECT_FALSE(std::filesystem::exists(log.string() + ".1")) << "the rename cannot have worked";

    // Stopped, keeping what it had: the earliest records survive.
    std::ifstream kept(log);
    std::ostringstream contents;
    contents << kept.rdbuf();
    EXPECT_THAT(contents.str(), HasSubstr("unrenamable record 0"));
    EXPECT_THAT(contents.str(), testing::Not(HasSubstr("unrenamable record 199")));

    nixl::shutdownLogFile();
    std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
    std::filesystem::remove_all(directory);
}

/** @brief With no NIXL_LOG_FILE_SIZE the file grows, exactly as it used to. */
TEST_F(nixlLogFileTest, GrowsWithoutLimitWhenNoSizeIsSet) {
    ASSERT_TRUE(enableLogFile());

    for (unsigned i = 0; i < 200; ++i) {
        NIXL_INFO << "unbounded record " << i;
    }

    EXPECT_FALSE(std::filesystem::exists(path_.string() + ".1"));
    EXPECT_GT(std::filesystem::file_size(path_), 2048u);
}

/**
 * @brief An unparsable size is reported and leaves the file unbounded: an
 *        invented limit would throw away records the operator meant to keep.
 */
TEST_F(nixlLogFileTest, IgnoresAnUnparsableSizeAndSaysSo) {
    const gtest::LogIgnoreGuard lig("Ignoring NIXL_LOG_FILE_SIZE");
    const std::string report = "Ignoring NIXL_LOG_FILE_SIZE";

    env_.addVar("NIXL_LOG_FILE", path_.string());

    // Asserted on the report rather than the absence of rotation: "-1" read as
    // unsigned wraps to a limit so large the file never rotates, which is
    // indistinguishable from unbounded. Only the report tells them apart.
    for (const std::string bad : {"sometime next week", "-1", "-1024", "64X", " 64", "+64"}) {
        countingSink watcher;

        env_.addVar("NIXL_LOG_FILE_SIZE", bad);
        ASSERT_TRUE(nixl::initLogFile())
            << "a bad size must not cost the log file: '" << bad << "'";
        EXPECT_EQ(watcher.countMatching(report), 1u) << "'" << bad << "' was not reported";

        nixl::shutdownLogFile();
        env_.popVar();
    }

    // Ignored, so the file grows as it would with no limit at all.
    env_.addVar("NIXL_LOG_FILE_SIZE", "sometime next week");
    ASSERT_TRUE(nixl::initLogFile());

    for (unsigned i = 0; i < 200; ++i) {
        NIXL_INFO << "unparsable size record " << i;
    }

    EXPECT_FALSE(std::filesystem::exists(path_.string() + ".1"));
    EXPECT_GT(std::filesystem::file_size(path_), 2048u);
}

/**
 * @brief Setup failures reach an operator who asked for errors only: at
 *        NIXL_LOG_LEVEL=ERROR a warning would be filtered out, leaving no way
 *        to find out why the file never appeared.
 */
TEST_F(nixlLogFileTest, ReportsSetupFailuresAtErrorSeverity) {
    const gtest::LogIgnoreGuard size_guard("Ignoring NIXL_LOG_FILE_SIZE");
    const gtest::LogIgnoreGuard open_guard("Could not open NIXL_LOG_FILE");

    // Restored by TearDown, which puts back what SetUp saved.
    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);

    {
        countingSink watcher;
        env_.addVar("NIXL_LOG_FILE", path_.string());
        env_.addVar("NIXL_LOG_FILE_SIZE", "not a size");
        ASSERT_TRUE(nixl::initLogFile());
        EXPECT_EQ(watcher.countMatching("Ignoring NIXL_LOG_FILE_SIZE"), 1u)
            << "an ignored size limit went unreported";
        nixl::shutdownLogFile();
        env_.popVar();
        env_.popVar();
    }

    {
        const std::filesystem::path directory = path_.string() + "-absent";
        std::filesystem::remove_all(directory);

        countingSink watcher;
        env_.addVar("NIXL_LOG_FILE", (directory / "log").string());
        EXPECT_FALSE(nixl::initLogFile());
        EXPECT_EQ(watcher.countMatching("Could not open NIXL_LOG_FILE"), 1u)
            << "a log file that never opened went unreported";
        env_.popVar();
    }
}

/**
 * @brief A failed open is reported with the open's own reason: the size lookup
 *        the sink does on the way in would otherwise replace its errno.
 */
TEST_F(nixlLogFileTest, ReportsWhyAnUnopenablePathReallyFailed) {
    if (::geteuid() == 0) {
        GTEST_SKIP() << "root bypasses the directory permission this relies on";
    }
    const gtest::LogIgnoreGuard lig("Could not open NIXL_LOG_FILE");

    // Searchable but not writable: creating fails EACCES, while asking the
    // size of that missing file fails ENOENT.
    const std::filesystem::path directory = path_.string() + "-readonly";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    std::filesystem::permissions(
        directory, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);

    countingSink watcher;
    env_.addVar("NIXL_LOG_FILE", (directory / "log").string());
    EXPECT_FALSE(nixl::initLogFile());
    EXPECT_EQ(watcher.countMatching("Permission denied"), 1u)
        << "the report named the wrong reason:\n"
        << watcher.text();

    std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
    std::filesystem::remove_all(directory);
}

/**
 * @brief A write failure is reported once, then records are dropped. A failed
 *        stream no-ops every later write, so without a report the file would
 *        stop part way through and say nothing. /dev/full gives a real ENOSPC
 *        without a full filesystem. The report is on stderr because reporting
 *        must not depend on the machinery that just failed.
 */
TEST_F(nixlLogFileTest, ReportsAWriteFailureOnceThenDropsRecords) {
    // Checked rather than assumed: a missing /dev/full would be created as an
    // ordinary file that accepts every write.
    if (!std::filesystem::is_character_file("/dev/full")) {
        GTEST_SKIP() << "/dev/full is not available on this system";
    }

    env_.addVar("NIXL_LOG_FILE", "/dev/full");
    ASSERT_TRUE(nixl::initLogFile()) << "/dev/full should open like any other file";

    testing::internal::CaptureStderr();
    NIXL_INFO << "first record";
    NIXL_INFO << "second record";
    NIXL_INFO << "third record";
    const std::string captured = testing::internal::GetCapturedStderr();

    // Once, however many records follow, so a failing file cannot bury stderr.
    const std::string report = "could not write to NIXL_LOG_FILE";
    size_t reports = 0;
    for (size_t at = captured.find(report); at != std::string::npos;
         at = captured.find(report, at + 1)) {
        ++reports;
    }
    EXPECT_EQ(reports, 1u) << "stderr was:\n" << captured;

    // Names the file and why, so the report is actionable.
    EXPECT_THAT(captured, HasSubstr("/dev/full"));
    EXPECT_THAT(captured, HasSubstr("No space left on device"));
}

/**
 * @brief A record logged from a static destructor still reaches the file.
 *
 * The teardown hook is in .fini_array, which glibc runs after the exit-handler
 * queue holding static destructors. That is loader behaviour rather than a
 * language guarantee, so it is pinned here rather than assumed.
 *
 * Needs a real process exit, so it runs in a helper: this binary told to run no
 * tests, exec'd rather than forked, since a forked image would run the whole
 * teardown chain against locks inherited from the parent's test run. Nothing
 * calls initLogFile(), so this exercises the path a real process takes.
 */
TEST_F(nixlLogFileTest, RecordsFromStaticDestructorsReachTheFile) {
    // Built before the fork: only async-signal-safe calls may run between fork
    // and exec, and setenv() can allocate. open() and dup2() below are safe.
    const std::vector<std::string> overrides = {
        "NIXL_LOG_FILE=" + path_.string(),
        "NIXL_LOG_LEVEL=INFO",
        std::string(late_record_env_var) + "=1",
    };

    // Minus the names overridden below, so those win rather than duplicating.
    std::vector<std::string> child_env;
    for (char **entry = environ; *entry != nullptr; ++entry) {
        const std::string text(*entry);
        const std::string name = text.substr(0, text.find('=') + 1);
        const bool overridden =
            std::any_of(overrides.begin(), overrides.end(), [&name](const std::string &o) {
                return o.compare(0, name.size(), name) == 0;
            });
        if (!overridden) {
            child_env.push_back(text);
        }
    }
    child_env.insert(child_env.end(), overrides.begin(), overrides.end());

    std::vector<char *> envp;
    for (std::string &entry : child_env) {
        envp.push_back(entry.data());
    }
    envp.push_back(nullptr);

    std::string helper_name = "nixl_log_file_late_record_helper";
    std::string no_tests = "--gtest_filter=-*";
    std::vector<char *> argv{helper_name.data(), no_tests.data(), nullptr};

    const pid_t pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        // Child. Keep the helper's output out of the test's, before the exec.
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
        }

        ::execve("/proc/self/exe", argv.data(), envp.data());

        // Only on exec failure; distinct from any status the helper returns.
        _exit(127);
    }

    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status)) << "helper did not exit normally";
    ASSERT_NE(WEXITSTATUS(status), 127) << "could not exec the helper";
    ASSERT_EQ(WEXITSTATUS(status), 0) << "helper exited " << WEXITSTATUS(status);

    EXPECT_THAT(readLogFile(), HasSubstr(late_record_text));
}

/**
 * @brief Concurrent writers produce whole lines: Abseil holds only a reader
 *        lock while dispatching, so the sink must serialize writes itself.
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

    // Only this test's records, so an unrelated thread cannot fail the count.
    // A torn write still shows up in the shape and set checks below.
    const auto lines = linesMatching("payload ");
    ASSERT_EQ(lines.size(), num_threads * per_thread);

    // A torn write would leave a line not ending in its own payload.
    const std::regex record("^I.*payload ([0-9]+):([0-9]+)$");
    std::set<std::pair<unsigned, unsigned>> seen;
    for (const auto &line : lines) {
        std::smatch fields;
        EXPECT_TRUE(std::regex_match(line, fields, record)) << "malformed line: " << line;
        if (fields.size() == 3) {
            seen.emplace(std::stoul(fields[1]), std::stoul(fields[2]));
        }
    }

    // Count and shape alone would pass if one payload were written twice and
    // another lost, so compare the sets.
    for (unsigned t = 0; t < num_threads; ++t) {
        for (unsigned i = 0; i < per_thread; ++i) {
            EXPECT_TRUE(seen.count({t, i}) == 1) << "missing payload " << t << ":" << i;
        }
    }
    EXPECT_EQ(seen.size(), num_threads * per_thread) << "unexpected payloads present";
}

} // namespace
