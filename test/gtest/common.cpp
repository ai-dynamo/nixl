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

PortAllocator::PortAllocator() : _port(_get_first_port()) {}

bool PortAllocator::_is_port_available(uint16_t port) {
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    int ret = bind(sock_fd, (struct sockaddr *)&addr, sizeof(addr));
    close(sock_fd);
    return ret == 0;
}

uint16_t PortAllocator::next_tcp_port() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_instance) {
        _instance = std::make_unique<PortAllocator>();
    }

    int max_port = MIN_PORT + _get_concurrent_id() * (PORT_RANGE + 1) - 1;

    while (!_is_port_available(++_instance->_port) && _instance->_port <= max_port);

    if (_instance->_port >= max_port) {
        // Please increase PORT_RANGE in common.h and .ci/scripts/common.sh to avoid this error
        throw std::runtime_error("Reached max port within executor port range, consider increasing PORT_RANGE");
    }

    return _instance->_port;
}

uint16_t PortAllocator::_get_first_port() {
    return MIN_PORT + _get_concurrent_id() * PORT_RANGE + OFFSET;
}

int PortAllocator::_get_concurrent_id() {
    char *jenkins_executor_number = getenv("EXECUTOR_NUMBER");
    char *gitlab_concurrent_id = getenv("CI_CONCURRENT_ID");

    if (jenkins_executor_number) {
        return std::stoi(jenkins_executor_number);
    } else if (gitlab_concurrent_id) {
        return std::stoi(gitlab_concurrent_id);
    }

    return rand() % ((MAX_PORT - MIN_PORT) / PORT_RANGE);
}
} // namespace gtest
