/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace gtest {
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
    void addVar(const std::string &name, const std::string &value);

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

} // namespace gtest

#endif /* TEST_GTEST_COMMON_H */
