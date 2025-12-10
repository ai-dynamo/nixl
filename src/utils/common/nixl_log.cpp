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

#include "nixl_log.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"
#include "absl/strings/ascii.h"
#include "absl/container/flat_hash_map.h"
#include <cstdlib>
#include <string>
#include <string_view>

namespace {

// Structure to hold logging settings
struct LogLevelSettings {
    absl::LogSeverityAtLeast min_severity;
    int vlog_level;
};

// Default log level if nothing else is specified
constexpr std::string_view kDefaultLogLevel = "WARN";

// Function to initialize logging, run before main() via constructor attribute.
void InitializeNixlLogging() __attribute__((constructor));

void InitializeNixlLogging()
{
    // Map from log level string to settings
    const absl::flat_hash_map<std::string_view, LogLevelSettings> kLogLevelMap = {
        {"TRACE", {absl::LogSeverityAtLeast::kInfo, 2}},
        {"DEBUG", {absl::LogSeverityAtLeast::kInfo, 1}},
        {"INFO",  {absl::LogSeverityAtLeast::kInfo, 0}},
        {"WARN",  {absl::LogSeverityAtLeast::kWarning, 0}},
        {"ERROR", {absl::LogSeverityAtLeast::kError, 0}},
        {"FATAL", {absl::LogSeverityAtLeast::kFatal, 0}},
    };

    // This is the fallback log level, an option of last resort if nothing else is specified.
    std::string_view level_to_use = kDefaultLogLevel;
    bool invalid_env_var = false;

    // Check environment variable, it has priority over compile-time default.
    const char* env_log_level = std::getenv("NIXL_LOG_LEVEL");
    std::string env_level_str_upper;
    if (env_log_level != nullptr) {
        env_level_str_upper = absl::AsciiStrToUpper(env_log_level);
        if (kLogLevelMap.contains(env_level_str_upper)) {
            level_to_use = env_level_str_upper;
        } else {
            // Fall back to kDefaultLogLevel if env var is invalid
            invalid_env_var = true;
        }
    }

    // Apply the settings
    auto it = kLogLevelMap.find(level_to_use);
    const LogLevelSettings& settings = (it != kLogLevelMap.end()) ? it->second : kLogLevelMap.at(kDefaultLogLevel);
    absl::SetMinLogLevel(settings.min_severity);
    absl::SetVLogLevel("*", settings.vlog_level);
    absl::SetStderrThreshold(settings.min_severity);
    absl::InitializeLog();

#ifdef NIXL_VERSION
    NIXL_INFO << "NIXL version: " << NIXL_VERSION
#ifdef NIXL_GIT_HASH
              << " (git: " << NIXL_GIT_HASH << ")"
#endif
        ;
#endif

    if (invalid_env_var) {
        NIXL_WARN << "Invalid NIXL_LOG_LEVEL environment variable, using default log level: " << kDefaultLogLevel;
    }
}

} // anonymous namespace
