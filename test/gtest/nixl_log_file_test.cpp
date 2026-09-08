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
#include <cstdlib>
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
 * @brief Logs from a static destructor, to check the log file outlives teardown.
 *
 * Declared at file scope so it is constructed before main() and therefore
 * destroyed near the end of the exit sequence, behind the static objects a real
 * process registers during startup. That is the stringent position: if the
 * NIXL_LOG_FILE sink is still registered when this runs, it was still
 * registered for every static destructor that ran before it.
 */
struct lateLogger {
    ~lateLogger() {
        if (std::getenv(late_record_env_var) != nullptr) {
            NIXL_INFO << late_record_text;
        }
    }
};

lateLogger late_logger;

/**
 * @brief Counts what Abseil hands to a sink other than the file sink.
 *
 * This is how the tests below check that adding a file does not displace
 * existing output.
 */
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
     *
     * Abseil requires Send() to be thread-safe and will call it from whichever
     * thread logged, so the append is serialized on mutex_, exactly as the real
     * sink in nixl_log.cpp does.
     */
    void
    Send(const absl::LogEntry &entry) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        text_.append(std::string(entry.text_message())).append("\n");
    }

    /**
     * @brief Number of records received whose message contains @p marker.
     * @return A count restricted to the caller's own records, so one logged by
     *         another thread cannot inflate it.
     */
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

    /**
     * @brief Concatenated messages received so far, one per line.
     * @return A copy, because a reference would hand the caller a member that
     *         another thread's Send() could be appending to.
     */
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
    /**
     * @brief Gives each test a clean sink, a known log level and its own path.
     *
     * Raises the level to INFO because most tests log at INFO, which the
     * default WARN would discard before any sink is consulted.
     */
    void
    SetUp() override {
        // The process may already have a sink from its own pre-main
        // initialization; drop it so each test starts from a known state.
        nixl::shutdownLogFile();

        prevMinLevel_ = absl::MinLogLevel();
        prevStderrThreshold_ = absl::StderrThreshold();

        // Most tests log at INFO, which the default WARN level would discard
        // before any sink is consulted. Keep stderr quiet so a passing run does
        // not bury the real test output in deliberate log records.
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
        absl::SetMinLogLevel(prevMinLevel_);
        absl::SetStderrThreshold(prevStderrThreshold_);
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
     * @brief The log file's lines that contain @p marker.
     * @return One entry per matching record. Any thread in the process can log
     *         while the sink is registered, so selecting on the caller's own
     *         text keeps an unrelated record out of an exact count.
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
    absl::LogSeverityAtLeast prevMinLevel_ = absl::LogSeverityAtLeast::kInfo;
    absl::LogSeverityAtLeast prevStderrThreshold_ = absl::LogSeverityAtLeast::kInfo;
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
 * @brief The file supplements stderr instead of diverting it.
 *
 * The central compatibility claim of the feature: existing tooling that scrapes
 * a process's console must see exactly what it saw before.
 */
TEST_F(nixlLogFileTest, AddsToStderrRatherThanReplacingIt) {
    // Abseil writes to stderr from its own default handler rather than through
    // a sink, so watch the real thing.
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
    // Deliberately no enableLogFile().
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

    // Derived from path_, which already carries this process's pid and the test
    // name, so a concurrent run cannot create the directory and turn the open
    // into a success. Cleared first in case an earlier run left it behind.
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
 * @brief %h and %p expand, so one setting can serve many processes.
 *
 * The point of the escapes: a disaggregated run sets one NIXL_LOG_FILE for
 * every worker and still gets a file per worker.
 */
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
 * @brief %% is a literal percent, and an unknown escape is left as written.
 *
 * A path is free to contain a percent that was never meant as an escape, so an
 * unrecognized one has to survive rather than be swallowed.
 */
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
 * @brief At the limit the file is rotated, and the newest records are kept.
 *
 * Which end survives is the whole design question. A log that answers "what
 * happened just before this hung" has to keep its tail, so the live file holds
 * the newest records and the previous generation sits beside it. Exactly one
 * generation is kept, which is what bounds the total.
 */
TEST_F(nixlLogFileTest, RotatesAtTheLimitAndKeepsTheNewestRecords) {
    constexpr uintmax_t limit = 2048;
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

    // The newest record is in the live file, and the rotated one holds what
    // came before it.
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
 * @brief A rotation it cannot do stops the sink, and says so.
 *
 * Carrying on would mean either ignoring the limit that was asked for or
 * quietly redefining rotation, so the sink stops instead. What it keeps is the
 * records written up to the limit; the file is left alone rather than
 * truncated, since those records are all there will be.
 */
TEST_F(nixlLogFileTest, StopsLoggingWhenItCannotRotate) {
    if (::geteuid() == 0) {
        GTEST_SKIP() << "root bypasses the directory permission this relies on";
    }
    constexpr uintmax_t limit = 2048;

    // Writable to begin with, so the log file can be created, and then made
    // searchable but not writable: renaming needs permission on the directory,
    // which is now gone, while writing and truncating need it on the file,
    // which it still has.
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

    // Stopped, keeping what it had rather than emptying the file: the earliest
    // records survive and the ones after the failed rotation are dropped.
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
 * @brief A size that cannot be parsed is reported and leaves the file unbounded.
 *
 * Falling back to some invented limit would quietly throw away records the
 * operator meant to keep, which is worse than ignoring the setting and saying
 * so.
 */
TEST_F(nixlLogFileTest, IgnoresAnUnparsableSizeAndSaysSo) {
    const gtest::LogIgnoreGuard lig("Ignoring NIXL_LOG_FILE_SIZE");
    const std::string report = "Ignoring NIXL_LOG_FILE_SIZE";

    env_.addVar("NIXL_LOG_FILE", path_.string());

    // The report is what is asserted on, rather than the absence of rotation,
    // because "-1" is the case that needs catching: read as an unsigned value
    // it wraps to the largest limit there is, and a file with a limit that
    // large never rotates -- which looks exactly like an unbounded one. Only
    // the report distinguishes a rejected setting from a disastrously
    // misparsed one.
    for (const std::string bad : {"sometime next week", "-1", "-1024", "64X", " 64", "+64"}) {
        countingSink watcher;

        env_.addVar("NIXL_LOG_FILE_SIZE", bad);
        ASSERT_TRUE(nixl::initLogFile())
            << "a bad size must not cost the log file: '" << bad << "'";
        EXPECT_EQ(watcher.countMatching(report), 1u) << "'" << bad << "' was not reported";

        nixl::shutdownLogFile();
        env_.popVar();
    }

    // Having been ignored, the limit leaves the file growing as it would with
    // no limit set at all.
    env_.addVar("NIXL_LOG_FILE_SIZE", "sometime next week");
    ASSERT_TRUE(nixl::initLogFile());

    for (unsigned i = 0; i < 200; ++i) {
        NIXL_INFO << "unparsable size record " << i;
    }

    EXPECT_FALSE(std::filesystem::exists(path_.string() + ".1"));
    EXPECT_GT(std::filesystem::file_size(path_), 2048u);
}

/**
 * @brief A failed open is reported with the open's own reason.
 *
 * The sink asks the file's size on the way in, to count an appended-to file's
 * existing contents against the limit. Asking before checking that the open
 * succeeded would replace the errno the report is built from, and name the
 * wrong reason for the failure.
 */
TEST_F(nixlLogFileTest, ReportsWhyAnUnopenablePathReallyFailed) {
    if (::geteuid() == 0) {
        GTEST_SKIP() << "root bypasses the directory permission this relies on";
    }
    const gtest::LogIgnoreGuard lig("Could not open NIXL_LOG_FILE");

    // Searchable but not writable, which is what separates the two reasons:
    // creating a file here fails with EACCES, while asking the size of that
    // same missing file fails with ENOENT.
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
 * @brief A write failure is reported once, and then records are dropped.
 *
 * A stream that has failed treats every later write as a silent no-op, so
 * without a report the file would stop part way through and say nothing about
 * why. /dev/full opens like any other file and fails every write with ENOSPC,
 * which gives the real failure without needing a full filesystem.
 *
 * The report has to reach stderr directly. This path runs inside a log sink
 * holding the sink's own mutex, so reporting through NIXL_WARN would re-enter
 * it on the same thread and deadlock; a test that hangs here is that bug.
 */
TEST_F(nixlLogFileTest, ReportsAWriteFailureOnceThenDropsRecords) {
    // Insisted on rather than assumed: opening a missing /dev/full in append
    // mode would create an ordinary file that accepts every write, so the test
    // would fail for a reason that has nothing to do with the code, and leave a
    // stray file in /dev behind it.
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

    // Said once, however many records follow, so a failing file cannot bury
    // the stderr output that is still working.
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
 * The teardown hook is an __attribute__((destructor)), so it lands in
 * .fini_array, which glibc runs after draining the exit-handler queue that
 * __cxa_atexit registers static destructors on. That ordering is loader
 * behaviour rather than a language guarantee, so this test pins it down instead
 * of leaving it as an assumption in a comment.
 *
 * Needs a real process exit, which gtest cannot do in-process, so it runs in a
 * helper process. That helper is exec'd rather than just forked: continuing in
 * a forked image would run the whole static-teardown chain, UCX, gRPC,
 * telemetry and Abseil included, against threads and locks inherited from the
 * parent's test run, and a mutex held by a thread that did not survive the fork
 * stays held forever in the child.
 *
 * The helper is this same binary, told to run no tests, so all it does is
 * start up and shut down. Nothing calls initLogFile(): the library's own
 * constructor registers the sink from NIXL_LOG_FILE, which makes this a test of
 * the path a real process takes. If the ordering ever reverses, the sink is
 * torn down before lateLogger runs and the record goes missing.
 */
TEST_F(nixlLogFileTest, RecordsFromStaticDestructorsReachTheFile) {
    // Everything the child needs is built here, before the fork. Between fork()
    // and exec() only async-signal-safe calls are allowed, and setenv() can
    // allocate, which would hang the child if another thread held the allocator
    // lock at the moment of the fork. open() and dup2() below are safe.
    const std::vector<std::string> overrides = {
        "NIXL_LOG_FILE=" + path_.string(),
        "NIXL_LOG_LEVEL=INFO",
        std::string(late_record_env_var) + "=1",
    };

    // Inherit the environment, minus the names being overridden, so the values
    // below win outright rather than relying on how getenv treats duplicates.
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
        // Child. Anything the helper prints is its own business, so keep it out
        // of the test output. Redirected before the exec so it survives it.
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
        }

        ::execve("/proc/self/exe", argv.data(), envp.data());

        // Only reached if the exec failed, and kept distinct from any status
        // the helper itself could return.
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

    // Only this test's records, so one logged by an unrelated thread cannot
    // fail the count. A torn write is still caught: if it kept the payload text
    // the shape check below rejects it, and if it lost the text the count and
    // the set check report it missing.
    const auto lines = linesMatching("payload ");
    ASSERT_EQ(lines.size(), num_threads * per_thread);

    // Every line must be a whole record. A torn or interleaved write would
    // leave a line that does not end in its own payload.
    const std::regex record("^I.*payload ([0-9]+):([0-9]+)$");
    std::set<std::pair<unsigned, unsigned>> seen;
    for (const auto &line : lines) {
        std::smatch fields;
        EXPECT_TRUE(std::regex_match(line, fields, record)) << "malformed line: " << line;
        if (fields.size() == 3) {
            seen.emplace(std::stoul(fields[1]), std::stoul(fields[2]));
        }
    }

    // Checking the line count and each line's shape would still pass if one
    // payload were written twice and another lost, so compare the set of
    // payloads actually present against the set that should be.
    for (unsigned t = 0; t < num_threads; ++t) {
        for (unsigned i = 0; i < per_thread; ++i) {
            EXPECT_TRUE(seen.count({t, i}) == 1) << "missing payload " << t << ":" << i;
        }
    }
    EXPECT_EQ(seen.size(), num_threads * per_thread) << "unexpected payloads present";
}

} // namespace
