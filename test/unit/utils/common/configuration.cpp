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

#undef NDEBUG

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <stdlib.h>
#include <unistd.h>

#include "common/configuration.h"

namespace {

const std::string pid = std::to_string(::getpid());
const std::string bool_name = "bool" + pid;
const std::string number_name = "number" + pid;
const std::string string_name = "string" + pid;

} // namespace

int
main()
{
    const auto pid = ::getpid();
    const auto file = "nixl_test_" + std::to_string(pid) + ".cfg";
    const auto path = std::filesystem::temp_directory_path() / file;
    {
        std::ofstream ofs(path, std::ios::out | std::ios::trunc);
        ofs << bool_name << " = true\n";
        ofs << number_name << " = 42\n";
        ofs << string_name << " = \"hello\"\n";
    }
    ::setenv("NIXL_CONFIG_FILE", path.native().c_str(), 1);
    {
        const auto value = nixl::config::getValue<bool>(bool_name);
        assert(value == true);
    }
    {
        const auto value = nixl::config::getValue<unsigned>(number_name);
        assert(value == 42);
    }
    {
        const auto value = nixl::config::getValue<std::string>(string_name);
        assert(value == "hello");
    }
    return 0;
}
