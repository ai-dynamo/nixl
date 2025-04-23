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
#include <cstdlib>
#include <string>

namespace {

// Function to initialize logging, run before main() via constructor attribute.
void InitializeNixlLogging() __attribute__((constructor));

void InitializeNixlLogging()
{
    // Initialize Abseil logging system.
    absl::InitializeLog();

    // First, check for runtime override via environment variable
    const char* env_log_level = std::getenv("NIXL_LOG_LEVEL");
    if (env_log_level != nullptr) {
        std::string level_str = absl::AsciiStrToUpper(env_log_level);
        if (level_str == "TRACE") {
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
            absl::SetVLogLevel("*", 2);
        } else if (level_str == "DEBUG") {
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
            absl::SetVLogLevel("*", 1);
        } else if (level_str == "INFO") {
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
            absl::SetVLogLevel("*", 0);
        } else if (level_str == "WARNING") {
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
            absl::SetVLogLevel("*", 0);
        } else if (level_str == "ERROR") {
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
            absl::SetVLogLevel("*", 0);
        } else if (level_str == "FATAL") {
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kFatal);
            absl::SetVLogLevel("*", 0);
        } else {
             // Unknown level from env var, default to WARNING
             absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
             absl::SetVLogLevel("*", 0);
        }
    } else {
        // If environment variable is not set, use the compile-time default level
        #if defined(LOG_LEVEL_TRACE)
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
            absl::SetVLogLevel("*", 2);
        #elif defined(LOG_LEVEL_DEBUG)
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
            absl::SetVLogLevel("*", 1);
        #elif defined(LOG_LEVEL_INFO)
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
            absl::SetVLogLevel("*", 0);
        #elif defined(LOG_LEVEL_WARNING)
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
            absl::SetVLogLevel("*", 0);
        #elif defined(LOG_LEVEL_ERROR)
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
            absl::SetVLogLevel("*", 0);
        #elif defined(LOG_LEVEL_FATAL)
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kFatal);
            absl::SetVLogLevel("*", 0);
        #else
            // Default compile-time level if none of the above are defined (should match meson default)
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError); // Default to Error
            absl::SetVLogLevel("*", 0);
        #endif
    }
}

} // anonymous namespace
