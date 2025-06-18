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
#include <stack>
#include <optional>

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

} // namespace gtest
