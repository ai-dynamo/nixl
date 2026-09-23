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
#include <cerrno>
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
#include <system_error>
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

constexpr const char *late_record_text = "record from a static destructor";
constexpr const char *helper_mode_env_var = "NIXL_LOG_FILE_TEST_HELPER";
constexpr const char *helper_path_env_var = "NIXL_LOG_FILE_TEST_PATH";
constexpr const char *run_marker_helper_mode = "run-marker";
constexpr const char *fatal_helper_mode = "fatal";
constexpr const char *fatal_setup_helper_mode = "fatal-setup";
constexpr const char *late_record_helper_mode = "late-record";
constexpr const char *single_record_helper_mode = "single-record";
constexpr const char *single_record_text = "record from the helper";
constexpr const char *fatal_record_text = "fatal record for the file";

struct helperProcessResult {
    pid_t pid = -1;
    int status = 0;
    int launchError = 0;
    int waitError = 0;
    bool timedOut = false;
    std::string output;
};

/**
 * @brief Whether a helper can be started by re-execing this binary.
 *
 * The helpers exec /proc/self/exe, which is Linux-only. Tests that need one
 * skip elsewhere rather than reporting the missing path as a failure.
 */
bool
canExecHelper() {
    std::error_code error;
    return std::filesystem::exists("/proc/self/exe", error);
}

/** @brief Contents of @p path, or empty if it cannot be read. */
std::string
readFile(const std::filesystem::path &path) {
    std::ifstream file(path);
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

/** @brief Runs one helper test in a fresh executable with a bounded wait. */
helperProcessResult
runHelper(const std::string &mode,
          const std::string &filter,
          const std::filesystem::path &data_path,
          const std::filesystem::path &output_path) {
    helperProcessResult result;

    const std::vector<std::string> overrides = {
        std::string("NIXL_LOG_FILE=") + (mode == late_record_helper_mode ? data_path.string() : ""),
        "NIXL_LOG_FILE_SIZE=",
        "NIXL_LOG_FILE_ERROR_IS_FATAL=",
        "NIXL_LOG_LEVEL=INFO",
        std::string(helper_mode_env_var) + "=" + mode,
        std::string(helper_path_env_var) + "=" + data_path.string(),
    };

    // Prepare everything before fork: only async-signal-safe calls are made
    // between fork and exec in the child.
    std::vector<std::string> child_env;
    for (char **entry = environ; *entry != nullptr; ++entry) {
        const std::string text(*entry);
        // Helpers must not inherit repetition, sharding, or parent report files.
        if (text.rfind("GTEST_", 0) == 0 || text.rfind("XML_OUTPUT_FILE=", 0) == 0 ||
            text.rfind("TEST_PREMATURE_EXIT_FILE=", 0) == 0) {
            continue;
        }
        const std::string name = text.substr(0, text.find('=') + 1);
        const bool overridden =
            std::any_of(overrides.begin(), overrides.end(), [&name](const std::string &value) {
                return value.compare(0, name.size(), name) == 0;
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

    std::string helper_name = "nixl_log_file_helper";
    std::string filter_arg = "--gtest_filter=" + filter;
    std::string no_color = "--gtest_color=no";
    std::vector<char *> argv{helper_name.data(), filter_arg.data(), no_color.data(), nullptr};

    const int output_fd =
        ::open(output_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (output_fd < 0) {
        result.launchError = errno;
        return result;
    }

    result.pid = ::fork();
    if (result.pid < 0) {
        result.launchError = errno;
        ::close(output_fd);
        return result;
    }

    if (result.pid == 0) {
        if (::dup2(output_fd, STDOUT_FILENO) < 0 || ::dup2(output_fd, STDERR_FILENO) < 0) {
            ::_exit(126);
        }
        if (output_fd > STDERR_FILENO) {
            ::close(output_fd);
        }
        ::execve("/proc/self/exe", argv.data(), envp.data());
        ::_exit(127);
    }
    ::close(output_fd);

    pid_t reaped = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((reaped = ::waitpid(result.pid, &result.status, WNOHANG)) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (reaped == 0) {
        result.timedOut = true;
        ::kill(result.pid, SIGKILL);
        if (::waitpid(result.pid, &result.status, 0) < 0) {
            result.waitError = errno;
        }
    } else if (reaped < 0) {
        result.waitError = errno;
    }

    result.output = readFile(output_path);
    std::filesystem::remove(output_path);
    return result;
}

/** @brief Restores the process working directory on every test exit path. */
class scopedCurrentPath {
public:
    scopedCurrentPath() : original_(std::filesystem::current_path()) {}

    ~scopedCurrentPath() {
        restore();
    }

    bool
    restore() {
        if (restored_) {
            return true;
        }

        std::error_code error;
        std::filesystem::current_path(original_, error);
        restored_ = !error;
        return restored_;
    }

private:
    std::filesystem::path original_;
    bool restored_ = false;
};

/**
 * @brief Logs from a static destructor, to check the file outlives teardown.
 *
 * At file scope, so it is destroyed near the end of the exit sequence: if the
 * sink is still registered here, it was for every earlier static destructor.
 */
struct lateLogger {
    ~lateLogger() {
        const char *mode = std::getenv(helper_mode_env_var);
        if (mode != nullptr && mode == std::string_view(late_record_helper_mode)) {
            NIXL_INFO << late_record_text;
        }
    }
};

lateLogger late_logger;

TEST(nixlLogFileHelper, ExpandsTheRunMarkerIntoThePath) {
    const char *mode = std::getenv(helper_mode_env_var);
    if (mode == nullptr || mode != std::string_view(run_marker_helper_mode)) {
        GTEST_SKIP() << "only run by the run-marker regression";
    }

    const char *configured_path = std::getenv(helper_path_env_var);
    ASSERT_NE(configured_path, nullptr);

    const std::filesystem::path base(configured_path);
    const std::string pattern = base.string() + "-%t";
    const std::string prefix = base.filename().string() + "-";
    const auto named = [&prefix, &base] {
        std::vector<std::string> found;
        for (const auto &entry : std::filesystem::directory_iterator(base.parent_path())) {
            const std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) == 0) {
                found.push_back(name.substr(prefix.size()));
            }
        }
        return found;
    };
    for (const auto &stale : named()) {
        std::filesystem::remove(base.parent_path() / (prefix + stale));
    }

    const auto now = [] {
        struct timespec ts {};
        ::clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
    };

    gtest::ScopedEnv env;
    env.addVar("NIXL_LOG_FILE", pattern);
    env.addVar("NIXL_LOG_FILE_SIZE", "");

    const uint64_t before = now();
    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "record for the marked path";
    nixl::shutdownLogFile();
    const uint64_t after = now();

    auto written = named();
    ASSERT_EQ(written.size(), 1u) << "expected exactly one file named for the run marker";
    const uint64_t marker = std::stoull(written.front());
    EXPECT_GE(marker, before) << "the marker predates its first use";
    EXPECT_LE(marker, after) << "the marker postdates its first use";

    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "record after rebinding";
    nixl::shutdownLogFile();

    written = named();
    EXPECT_EQ(written.size(), 1u) << "the run marker moved within one process";
    if (written.size() == 1) {
        const std::string contents = readFile(base.parent_path() / (prefix + written.front()));
        EXPECT_THAT(contents, HasSubstr("record for the marked path"));
        EXPECT_THAT(contents, HasSubstr("record after rebinding"));
    }

    for (const auto &leftover : written) {
        std::filesystem::remove(base.parent_path() / (prefix + leftover));
    }
}

TEST(nixlLogFileHelper, EmitsFatalRecord) {
    const char *mode = std::getenv(helper_mode_env_var);
    if (mode == nullptr || mode != std::string_view(fatal_helper_mode)) {
        GTEST_SKIP() << "only run by the fatal-record regression";
    }

    const char *configured_path = std::getenv(helper_path_env_var);
    ASSERT_NE(configured_path, nullptr);

    gtest::ScopedEnv env;
    env.addVar("NIXL_LOG_FILE", configured_path);
    env.addVar("NIXL_LOG_FILE_SIZE", "");
    ASSERT_TRUE(nixl::initLogFile());

    NIXL_FATAL << fatal_record_text;
}

TEST(nixlLogFileHelper, DiesWhenTheLogFileCannotOpenAndErrorsAreFatal) {
    const char *mode = std::getenv(helper_mode_env_var);
    if (mode == nullptr || mode != std::string_view(fatal_setup_helper_mode)) {
        GTEST_SKIP() << "only run by the fatal-setup regression";
    }

    const char *configured_path = std::getenv(helper_path_env_var);
    ASSERT_NE(configured_path, nullptr);

    gtest::ScopedEnv env;
    env.addVar("NIXL_LOG_FILE_ERROR_IS_FATAL", "1");
    // A path under a directory that does not exist cannot be opened.
    env.addVar("NIXL_LOG_FILE", std::string(configured_path) + ".missing/x.log");

    // Expected not to return: the report is fatal. Returning at all means the
    // setting was not honoured, which the parent reports as a normal exit.
    nixl::initLogFile();
}

TEST(nixlLogFileHelper, EmitsSingleRecord) {
    const char *mode = std::getenv(helper_mode_env_var);
    if (mode == nullptr || mode != std::string_view(single_record_helper_mode)) {
        GTEST_SKIP() << "only run by the helper-isolation regression";
    }

    const char *configured_path = std::getenv(helper_path_env_var);
    ASSERT_NE(configured_path, nullptr);

    gtest::ScopedEnv env;
    env.addVar("NIXL_LOG_FILE", configured_path);
    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << single_record_text;
}

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

        // Tests start unlimited and non-fatal regardless of the invoking shell.
        // Individual tests can push on top, and ScopedEnv restores both layers.
        env_.addVar("NIXL_LOG_FILE_SIZE", "");
        env_.addVar("NIXL_LOG_FILE_ERROR_IS_FATAL", "");

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

/**
 * @brief Unset, empty, false and unrecognised values all keep a setup failure
 *        survivable: an abort has to be asked for with a recognised true.
 */
TEST_F(nixlLogFileTest, FatalSetupErrorsStayOffForOtherValues) {
    const gtest::LogIgnoreGuard lig("Could not open NIXL_LOG_FILE");

    // From path_, which carries the pid and test name, so a concurrent run
    // cannot create the directory and turn the open into a success.
    auto missingDir = path_;
    missingDir += ".missing";
    std::filesystem::remove_all(missingDir);
    ASSERT_FALSE(std::filesystem::exists(missingDir));
    const auto bad = missingDir / "x.log";

    for (const char *value : {"", "0", "false", "no", "off", "disable", "n", "not a boolean"}) {
        env_.addVar("NIXL_LOG_FILE_ERROR_IS_FATAL", value);
        env_.addVar("NIXL_LOG_FILE", bad.string());

        EXPECT_FALSE(nixl::initLogFile()) << "'" << value << "' made the failure fatal";

        env_.popVar();
        env_.popVar();
    }
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

/** @brief A direct write makes each record readable without waiting for shutdown. */
TEST_F(nixlLogFileTest, RecordsAreReadableWithoutWaitingForShutdown) {
    ASSERT_TRUE(enableLogFile());

    NIXL_INFO << "readable immediately";

    EXPECT_THAT(readLogFile(), HasSubstr("readable immediately"));
}

TEST_F(nixlLogFileTest, HelpersIgnoreParentGoogleTestEnvironment) {
    if (!canExecHelper()) {
        GTEST_SKIP() << "re-execing this binary needs /proc/self/exe";
    }

    const std::filesystem::path parent_report = path_.string() + ".parent-report";
    const std::string sentinel = "parent report";
    std::ofstream(parent_report) << sentinel;

    env_.addVar("GTEST_FILTER", "NoSuchTest.*");
    env_.addVar("GTEST_REPEAT", "2");
    env_.addVar("GTEST_TOTAL_SHARDS", "2");
    env_.addVar("GTEST_SHARD_INDEX", "1");
    env_.addVar("GTEST_SHARD_STATUS_FILE", parent_report.string());
    env_.addVar("GTEST_OUTPUT", "xml:" + parent_report.string());
    env_.addVar("XML_OUTPUT_FILE", parent_report.string());
    env_.addVar("TEST_PREMATURE_EXIT_FILE", parent_report.string());

    const std::filesystem::path output = path_.string() + ".helper-output";
    const auto result =
        runHelper(single_record_helper_mode, "nixlLogFileHelper.EmitsSingleRecord", path_, output);
    const std::string report_after = readFile(parent_report);
    std::filesystem::remove(parent_report);

    ASSERT_EQ(result.launchError, 0) << "could not launch helper: errno " << result.launchError;
    ASSERT_FALSE(result.timedOut) << "helper exceeded its 10-second deadline\n" << result.output;
    ASSERT_EQ(result.waitError, 0) << "waitpid failed: errno " << result.waitError;
    ASSERT_TRUE(WIFEXITED(result.status)) << "helper did not exit normally\n" << result.output;
    ASSERT_EQ(WEXITSTATUS(result.status), 0) << "helper failed:\n" << result.output;
    EXPECT_EQ(linesMatching(single_record_text).size(), 1u) << result.output;
    EXPECT_EQ(report_after, sentinel);
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
    if (!canExecHelper()) {
        GTEST_SKIP() << "re-execing this binary needs /proc/self/exe";
    }

    // A directory of its own: the helper has to scan for the name the marker
    // produced, and in a shared temp directory that walk would trip over
    // entries other runs are deleting underneath it.
    const std::filesystem::path directory = path_.string() + "-marker";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const std::filesystem::path output = path_.string() + ".helper-output";
    const auto result = runHelper(run_marker_helper_mode,
                                  "nixlLogFileHelper.ExpandsTheRunMarkerIntoThePath",
                                  directory / "log",
                                  output);

    // Cleaned up even when the helper failed before doing so.
    std::filesystem::remove_all(directory);

    ASSERT_EQ(result.launchError, 0) << "could not launch helper: errno " << result.launchError;
    ASSERT_FALSE(result.timedOut) << "run-marker helper exceeded its 10-second deadline\n"
                                  << result.output;
    ASSERT_EQ(result.waitError, 0) << "waitpid failed: errno " << result.waitError;
    ASSERT_TRUE(WIFEXITED(result.status)) << "run-marker helper terminated abnormally\n"
                                          << result.output;
    ASSERT_NE(WEXITSTATUS(result.status), 127) << "could not exec the run-marker helper";
    EXPECT_EQ(WEXITSTATUS(result.status), 0) << "run-marker helper failed:\n" << result.output;
}

/** @brief %% lets a path contain a literal percent. */
TEST_F(nixlLogFileTest, ExpandsLiteralPercent) {
    const std::string pattern = path_.string() + "-%%";
    const std::filesystem::path expanded = path_.string() + "-%";
    std::filesystem::remove(expanded);

    env_.addVar("NIXL_LOG_FILE", pattern);
    ASSERT_TRUE(nixl::initLogFile());
    NIXL_INFO << "record for the literal path";
    nixl::shutdownLogFile();

    EXPECT_TRUE(std::filesystem::exists(expanded)) << "expected " << expanded;

    std::filesystem::remove(expanded);
}

/** @brief Unknown and incomplete escapes are rejected so future extensions are unambiguous. */
TEST_F(nixlLogFileTest, RejectsUnknownAndIncompleteEscapes) {
    const gtest::LogIgnoreGuard lig("Invalid NIXL_LOG_FILE");
    const std::string report = "Invalid NIXL_LOG_FILE";

    for (const std::string suffix : {"-%z", "-%"}) {
        const std::filesystem::path pattern = path_.string() + suffix;
        std::filesystem::remove(pattern);
        countingSink watcher;

        env_.addVar("NIXL_LOG_FILE", pattern.string());
        EXPECT_FALSE(nixl::initLogFile()) << "'" << pattern << "' was accepted";
        EXPECT_EQ(watcher.countMatching(report), 1u) << "'" << pattern << "' was not reported";
        EXPECT_FALSE(std::filesystem::exists(pattern)) << "'" << pattern << "' was created";

        env_.popVar();
    }
}

/**
 * @brief At the limit the file rotates, keeping the newest records: a log that
 *        answers "what happened before this hung" needs its tail. One
 *        generation is kept, which bounds the total.
 */
TEST_F(nixlLogFileTest, RotatesAtTheLimitAndKeepsTheNewestRecords) {
    constexpr std::uintmax_t limit = 4096;
    const std::filesystem::path rotated = path_.string() + ".1";
    std::filesystem::remove(rotated);

    env_.addVar("NIXL_LOG_FILE", path_.string());
    env_.addVar("NIXL_LOG_FILE_SIZE", "4K");
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

/** @brief Existing oversized files survive until bounded generations replace them. */
TEST_F(nixlLogFileTest, PreservesExistingOversizedFilesUntilRotationReplacesThem) {
    constexpr std::uintmax_t limit = 4096;
    const std::filesystem::path rotated = path_.string() + ".1";

    for (const bool oversized_active : {false, true}) {
        const std::string initial_active =
            oversized_active ? std::string(2 * limit, 'a') : "earlier live record\n";
        const std::string initial_backup(3 * limit, 'b');
        std::ofstream(path_) << initial_active;
        std::ofstream(rotated) << initial_backup;

        env_.addVar("NIXL_LOG_FILE", path_.string());
        env_.addVar("NIXL_LOG_FILE_SIZE", "4K");
        ASSERT_TRUE(nixl::initLogFile());
        EXPECT_EQ(readLogFile(), initial_active);
        EXPECT_EQ(readFile(rotated), initial_backup);

        NIXL_INFO << "first record under the new limit";
        EXPECT_THAT(readLogFile(), HasSubstr("first record under the new limit"));
        EXPECT_EQ(readFile(rotated), oversized_active ? initial_active : initial_backup);

        for (unsigned i = 0; i < 200; ++i) {
            NIXL_INFO << "bounded replacement record " << i;
        }
        nixl::shutdownLogFile();

        EXPECT_LE(std::filesystem::file_size(path_), limit);
        EXPECT_LE(std::filesystem::file_size(rotated), limit);
        EXPECT_THAT(readLogFile(), HasSubstr("bounded replacement record 199"));

        env_.popVar();
        env_.popVar();
        std::filesystem::remove(rotated);
    }
}

/**
 * @brief A relative path is bound when the sink opens, not resolved again when
 *        rotation happens after the process changes directory.
 */
TEST_F(nixlLogFileTest, KeepsRelativePathAcrossWorkingDirectoryChanges) {
    constexpr std::string_view relative_name = "relative.log";
    constexpr std::string_view live_sentinel = "unrelated live file";
    constexpr std::string_view backup_sentinel = "unrelated backup file";

    for (const bool populate_destination : {false, true}) {
        const std::string variant = populate_destination ? "populated" : "empty";
        const std::filesystem::path directory_a = path_.string() + "-" + variant + "-a";
        const std::filesystem::path directory_b = path_.string() + "-" + variant + "-b";
        std::filesystem::remove_all(directory_a);
        std::filesystem::remove_all(directory_b);
        std::filesystem::create_directories(directory_a);
        std::filesystem::create_directories(directory_b);

        const std::filesystem::path log_a = directory_a / relative_name;
        const std::filesystem::path rotated_a = log_a.string() + ".1";
        const std::filesystem::path log_b = directory_b / relative_name;
        const std::filesystem::path rotated_b = log_b.string() + ".1";
        if (populate_destination) {
            std::ofstream(log_b) << live_sentinel;
            std::ofstream(rotated_b) << backup_sentinel;
        }

        scopedCurrentPath current_path;
        std::filesystem::current_path(directory_a);
        env_.addVar("NIXL_LOG_FILE", std::string(relative_name));
        env_.addVar("NIXL_LOG_FILE_SIZE", "4K");
        ASSERT_TRUE(nixl::initLogFile());
        NIXL_INFO << "record opened in directory A";

        std::filesystem::current_path(directory_b);
        for (unsigned i = 0; i < 200; ++i) {
            NIXL_INFO << "relative-path rotation record " << i;
        }
        nixl::shutdownLogFile();
        ASSERT_TRUE(current_path.restore()) << "could not restore the working directory";

        ASSERT_TRUE(std::filesystem::exists(rotated_a)) << "rotation did not stay in directory A";
        EXPECT_THAT(readFile(log_a), HasSubstr("relative-path rotation record 199"));

        if (populate_destination) {
            EXPECT_EQ(readFile(log_b), live_sentinel);
            EXPECT_EQ(readFile(rotated_b), backup_sentinel);
        } else {
            EXPECT_FALSE(std::filesystem::exists(log_b));
            EXPECT_FALSE(std::filesystem::exists(rotated_b));
        }

        env_.popVar();
        env_.popVar();
        std::filesystem::remove_all(directory_a);
        std::filesystem::remove_all(directory_b);
    }
}

/** @brief A record too large for an empty file remains on stderr but is omitted here. */
TEST_F(nixlLogFileTest, DropsRecordLargerThanLimit) {
    constexpr std::uintmax_t limit = 4096;
    const std::filesystem::path rotated = path_.string() + ".1";
    std::filesystem::remove(rotated);

    env_.addVar("NIXL_LOG_FILE", path_.string());
    env_.addVar("NIXL_LOG_FILE_SIZE", std::to_string(limit));
    ASSERT_TRUE(nixl::initLogFile());

    const std::string payload(5000, 'x');
    NIXL_INFO << "oversized record " << payload;

    EXPECT_EQ(std::filesystem::file_size(path_), 0u);
    EXPECT_THAT(readLogFile(), testing::Not(HasSubstr(payload)));
    EXPECT_FALSE(std::filesystem::exists(rotated));

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
    constexpr std::uintmax_t limit = 4096;

    // Writable first so the file can be created, then searchable but not
    // writable: rename needs the directory, writing only needs the file.
    const std::filesystem::path directory = path_.string() + "-norename";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::filesystem::path log = directory / "log";

    env_.addVar("NIXL_LOG_FILE", log.string());
    env_.addVar("NIXL_LOG_FILE_SIZE", "4K");
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

/** @brief An unparsable explicit limit is reported and prevents file logging. */
TEST_F(nixlLogFileTest, RejectsAnUnparsableSizeAndSaysSo) {
    const gtest::LogIgnoreGuard lig("Invalid NIXL_LOG_FILE_SIZE");
    const std::string report = "Invalid NIXL_LOG_FILE_SIZE";

    env_.addVar("NIXL_LOG_FILE", path_.string());

    for (const std::string bad : {
             "sometime next week",
             "-1",
             "-1024",
             "64X",
             " 64",
             "+64",
             "184467440737095516160",
         }) {
        countingSink watcher;

        env_.addVar("NIXL_LOG_FILE_SIZE", bad);
        EXPECT_FALSE(nixl::initLogFile()) << "'" << bad << "' was accepted";
        EXPECT_EQ(watcher.countMatching(report), 1u) << "'" << bad << "' was not reported";
        EXPECT_FALSE(logFileExists()) << "'" << bad << "' still created a log file";

        env_.popVar();
    }
}

/** @brief A parseable limit below 4096 bytes is rejected with a specific error. */
TEST_F(nixlLogFileTest, RejectsSizeBelowMinimumAndSaysSo) {
    const std::string report = "value is below the minimum of 4096 bytes";
    const gtest::LogIgnoreGuard lig(report);

    env_.addVar("NIXL_LOG_FILE", path_.string());

    for (const std::string bad : {"0", "4095", "3K"}) {
        countingSink watcher;

        env_.addVar("NIXL_LOG_FILE_SIZE", bad);
        EXPECT_FALSE(nixl::initLogFile()) << "'" << bad << "' was accepted";
        EXPECT_EQ(watcher.countMatching(report), 1u) << "'" << bad << "' had no specific error";
        EXPECT_FALSE(logFileExists()) << "'" << bad << "' still created a log file";

        env_.popVar();
    }
}

/**
 * @brief Setup failures reach an operator who asked for errors only: at
 *        NIXL_LOG_LEVEL=ERROR a warning would be filtered out, leaving no way
 *        to find out why the file never appeared.
 */
TEST_F(nixlLogFileTest, ReportsSetupFailuresAtErrorSeverity) {
    const gtest::LogIgnoreGuard size_guard("Invalid NIXL_LOG_FILE_SIZE");
    const gtest::LogIgnoreGuard open_guard("Could not open NIXL_LOG_FILE");

    // Restored by TearDown, which puts back what SetUp saved.
    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);

    {
        countingSink watcher;
        env_.addVar("NIXL_LOG_FILE", path_.string());
        env_.addVar("NIXL_LOG_FILE_SIZE", "not a size");
        EXPECT_FALSE(nixl::initLogFile());
        EXPECT_EQ(watcher.countMatching("Invalid NIXL_LOG_FILE_SIZE"), 1u)
            << "an invalid size limit went unreported";
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
 * @brief The second fatal dispatch contributes its stack trace rather than a
 *        duplicate of the fatal message.
 */
TEST_F(nixlLogFileTest, WritesFatalMessageOnceWithItsStackTrace) {
    if (!canExecHelper()) {
        GTEST_SKIP() << "re-execing this binary needs /proc/self/exe";
    }

    const std::filesystem::path output = path_.string() + ".helper-output";
    const auto result =
        runHelper(fatal_helper_mode, "nixlLogFileHelper.EmitsFatalRecord", path_, output);

    ASSERT_EQ(result.launchError, 0) << "could not launch helper: errno " << result.launchError;
    ASSERT_FALSE(result.timedOut) << "fatal helper exceeded its 10-second deadline\n"
                                  << result.output;
    ASSERT_EQ(result.waitError, 0) << "waitpid failed: errno " << result.waitError;
    ASSERT_TRUE(WIFSIGNALED(result.status)) << "fatal helper did not terminate by signal\n"
                                            << result.output;
    EXPECT_EQ(WTERMSIG(result.status), SIGABRT) << "fatal helper received an unexpected signal";

    const std::string contents = readLogFile();
    size_t fatal_messages = 0;
    for (size_t at = contents.find(fatal_record_text); at != std::string::npos;
         at = contents.find(fatal_record_text, at + 1)) {
        ++fatal_messages;
    }
    EXPECT_EQ(fatal_messages, 1u) << "the fatal message was not written exactly once:\n"
                                  << contents;

    EXPECT_THAT(contents, HasSubstr("*** Check failure stack trace: ***\n"))
        << "the fatal stack-trace content is missing";
}

/**
 * @brief NIXL_LOG_FILE_ERROR_IS_FATAL turns a setup failure into a fatal one:
 *        an operator who sets it would rather the process die than run without
 *        the log it asked for. Needs a helper because the process aborts.
 */
TEST_F(nixlLogFileTest, FatalSetupErrorTerminatesTheProcess) {
    if (!canExecHelper()) {
        GTEST_SKIP() << "re-execing this binary needs /proc/self/exe";
    }

    const std::filesystem::path output = path_.string() + ".helper-output";
    const auto result = runHelper(fatal_setup_helper_mode,
                                  "nixlLogFileHelper.DiesWhenTheLogFileCannotOpenAndErrorsAreFatal",
                                  path_,
                                  output);

    ASSERT_EQ(result.launchError, 0) << "could not launch helper: errno " << result.launchError;
    ASSERT_FALSE(result.timedOut) << "fatal-setup helper exceeded its 10-second deadline\n"
                                  << result.output;
    ASSERT_EQ(result.waitError, 0) << "waitpid failed: errno " << result.waitError;
    ASSERT_TRUE(WIFSIGNALED(result.status)) << "fatal-setup helper did not terminate by signal\n"
                                            << result.output;
    EXPECT_EQ(WTERMSIG(result.status), SIGABRT)
        << "fatal-setup helper received an unexpected signal";
    EXPECT_THAT(result.output, HasSubstr("Could not open NIXL_LOG_FILE"))
        << "the fatal report never named the failure";
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
    if (!canExecHelper()) {
        GTEST_SKIP() << "re-execing this binary needs /proc/self/exe";
    }

    const std::filesystem::path output = path_.string() + ".helper-output";
    const auto result = runHelper(late_record_helper_mode, "-*", path_, output);

    ASSERT_EQ(result.launchError, 0) << "could not launch helper: errno " << result.launchError;
    ASSERT_FALSE(result.timedOut) << "late-record helper exceeded its 10-second deadline\n"
                                  << result.output;
    ASSERT_EQ(result.waitError, 0) << "waitpid failed: errno " << result.waitError;
    ASSERT_TRUE(WIFEXITED(result.status)) << "helper did not exit normally\n" << result.output;
    ASSERT_EQ(WEXITSTATUS(result.status), 0) << "helper failed:\n" << result.output;

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
