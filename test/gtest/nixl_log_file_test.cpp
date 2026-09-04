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
     * thread logged, so both members are serialized on mutex_, exactly as the
     * real sink in nixl_log.cpp does.
     */
    void
    Send(const absl::LogEntry &entry) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        text_.append(std::string(entry.text_message())).append("\n");
        ++count_;
    }

    /** @brief Number of records this sink has received. */
    size_t
    count() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return count_;
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
    size_t count_ = 0;
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
    EXPECT_EQ(other.count(), 1u);
    EXPECT_FALSE(std::filesystem::exists(bad));
}

/** @brief Repeated init calls leave a single registration, so records are not duplicated. */
TEST_F(nixlLogFileTest, InitIsIdempotent) {
    ASSERT_TRUE(enableLogFile());
    EXPECT_TRUE(nixl::initLogFile());
    EXPECT_TRUE(nixl::initLogFile());

    NIXL_INFO << "written once";

    // A sink registered twice would duplicate every record.
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

    const auto lines = readLogLines();
    ASSERT_EQ(lines.size(), num_threads * per_thread);

    // Every line must be a whole record. A torn or interleaved write would
    // leave a line that does not end in its own payload.
    const std::regex record("^I.*payload [0-7]:[0-9]+$");
    for (const auto &line : lines) {
        EXPECT_TRUE(std::regex_match(line, record)) << "malformed line: " << line;
    }
}

} // namespace
