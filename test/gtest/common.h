/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef TEST_GTEST_COMMON_H
#define TEST_GTEST_COMMON_H

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stack>
#include <optional>
#include <mutex>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_entry.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gtest {

inline bool
hasCudaGpu() {
#ifdef HAVE_CUDA
    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
#else
    return false;
#endif
}

constexpr const char *
GetMockBackendName() {
    return "MOCK_BACKEND";
}

class Logger {
public:
    Logger(const std::string &title = "INFO");
    ~Logger();

    template<typename T> Logger &operator<<(const T &value)
    {
        std::cout << value;
        return *this;
    }
};

class ScopedEnv {
public:
    void
    addVar(const std::string &name, const std::string &value);
    void
    popVar();

private:
    class Variable {
    public:
        Variable(const std::string &name, const std::string &value);
        Variable(Variable &&other);
        ~Variable();

        Variable(const Variable &other) = delete;
        Variable &operator=(const Variable &other) = delete;

    private:
        std::optional<std::string> m_prev_value;
        std::string m_name;
    };

    std::stack<Variable> m_vars;
};

class PortAllocator {
public:
    static constexpr uint16_t MIN_PORT = 10500;
    static constexpr uint16_t MAX_PORT = 65535;

private:
    PortAllocator() = default;
    ~PortAllocator() = default;
    PortAllocator(const PortAllocator &other) = delete;
    void
    operator=(const PortAllocator &) = delete;

public:
    static uint16_t
    next_tcp_port();
    static PortAllocator &
    instance();

    void
    set_min_port(uint16_t min_port);
    void
    set_max_port(uint16_t max_port);

private:
    static bool
    is_port_available(uint16_t port);

    std::mutex _mutex;
    uint16_t _port = MIN_PORT;
    uint16_t _min_port = MIN_PORT;
    uint16_t _max_port = MAX_PORT;
};

/**
 * @brief A scoped LogSink that captures log messages for testing assertions.
 *
 * This class registers itself with Abseil's logging system to intercept
 * log messages on construction and unregisters on destruction. It can be
 * used in tests to verify that expected warnings or errors are logged.
 *
 * Usage:
 *   scopedTestLogSink sink;
 *   // ... code that logs warnings ...
 *   EXPECT_EQ(sink.warningCount(), 1);
 *   EXPECT_EQ(sink.countWarnings("expected message"), 1);
 */
class scopedTestLogSink {
public:
    scopedTestLogSink();
    ~scopedTestLogSink();

    scopedTestLogSink(const scopedTestLogSink &) = delete;
    scopedTestLogSink &
    operator=(const scopedTestLogSink &) = delete;

    [[nodiscard]] size_t
    warningCount() const;

    [[nodiscard]] size_t
    countWarningsMatching(const std::string &substring) const;

private:
    class testLogSink : public absl::LogSink {
    public:
        void
        Send(const absl::LogEntry &entry) override;

        mutable std::mutex mutex_;
        std::vector<std::string> warnings_;
    };

    testLogSink sink_;
};

struct nixlTestParam {
    std::string backendName;
    bool progressThreadEnabled;
    unsigned numWorkers;
    unsigned numThreads;
    std::string engineConfig;
};

using nixl_test_t = testing::TestWithParam<nixlTestParam>;

} // namespace gtest

#define NIXL_INSTANTIATE_TEST(_test_name,               \
                              _test_case,               \
                              _backend,                 \
                              _progress_thread_enabled, \
                              _num_workers,             \
                              _num_threads,             \
                              _engine_config)           \
    INSTANTIATE_TEST_SUITE_P(                           \
        _test_name,                                     \
        _test_case,                                     \
        testing::ValuesIn(std::vector<nixlTestParam>(   \
            {{_backend, _progress_thread_enabled, _num_workers, _num_threads, _engine_config}})));

#endif /* TEST_GTEST_COMMON_H */
