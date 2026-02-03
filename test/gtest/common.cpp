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

#include "common.h"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <memory>
#include <stack>
#include <optional>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <random>
#include "absl/log/log_sink_registry.h"

namespace gtest {

Logger::Logger(const std::string &title)
{
    std::cout << "[ " << std::setw(8) << title << " ] ";
}

Logger::~Logger()
{
    std::cout << std::endl;
}

void ScopedEnv::addVar(const std::string &name, const std::string &value)
{
    m_vars.emplace(name, value);
}

void
ScopedEnv::popVar() {
    m_vars.pop();
}

ScopedEnv::Variable::Variable(const std::string &name, const std::string &value)
    : m_name(name)
{
    const char* backup = getenv(name.c_str());

    if (backup != nullptr) {
        m_prev_value = backup;
    }

    setenv(name.c_str(), value.c_str(), 1);
}

ScopedEnv::Variable::Variable(Variable &&other)
    : m_prev_value(std::move(other.m_prev_value)),
      m_name(std::move(other.m_name))
{
    // The moved-from object should be invalidated
    assert(other.m_name.empty());
}

ScopedEnv::Variable::~Variable()
{
    if (m_name.empty()) {
        return;
    }

    if (m_prev_value) {
        setenv(m_name.c_str(), m_prev_value->c_str(), 1);
    } else {
        unsetenv(m_name.c_str());
    }
}

PortAllocator &
PortAllocator::instance() {
    static PortAllocator _instance;
    return _instance;
}

void
PortAllocator::set_min_port(uint16_t min_port) {
    _min_port = min_port;
    _port = _min_port;
}

void
PortAllocator::set_max_port(uint16_t max_port) {
    _max_port = max_port;
}

bool
PortAllocator::is_port_available(uint16_t port) {
    struct sockaddr_in addr = {
        .sin_family = AF_INET, .sin_port = htons(port), .sin_addr = {.s_addr = INADDR_ANY}};

    const auto sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    const auto ret = bind(sock_fd, (struct sockaddr *)&addr, sizeof(addr));
    close(sock_fd);
    return ret == 0;
}

uint16_t
PortAllocator::next_tcp_port() {
    PortAllocator &instance = PortAllocator::instance();
    std::lock_guard<std::mutex> lock(instance._mutex);
    const int port_range = instance._max_port - instance._min_port;

    for (int scanned = 0; scanned < port_range; scanned++) {
        if (is_port_available(instance._port)) {
            return instance._port++;
        }

        instance._port++;

        if (instance._port >= instance._max_port) {
            instance._port = instance._min_port;
        }
    }

    throw std::runtime_error("No port available in range: " + std::to_string(instance._min_port) +
                             " - " + std::to_string(instance._max_port));
}

scopedTestLogSink::scopedTestLogSink() {
    absl::AddLogSink(&sink_);
}

scopedTestLogSink::~scopedTestLogSink() {
    absl::RemoveLogSink(&sink_);
}

void
scopedTestLogSink::testLogSink::Send(const absl::LogEntry &entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (entry.log_severity() == absl::LogSeverity::kWarning) {
        warnings_.emplace_back(entry.text_message());
    }
}

size_t
scopedTestLogSink::warningCount() const {
    std::lock_guard<std::mutex> lock(sink_.mutex_);
    return sink_.warnings_.size();
}

size_t
scopedTestLogSink::countWarningsMatching(const std::string &substring) const {
    std::lock_guard<std::mutex> lock(sink_.mutex_);
    size_t count = 0;
    for (const auto &w : sink_.warnings_) {
        if (w.find(substring) != std::string::npos) {
            ++count;
        }
    }
    return count;
}

} // namespace gtest
