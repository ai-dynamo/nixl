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
#ifndef __NIXL_LOG_H
#define __NIXL_LOG_H

#include "absl/log/log.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"

/* Initialize logging. Should be called at the start of main() */
inline void InitLogging() {
    absl::InitializeLog();
}

/*-----------------------------------------------------------------------------*
 * Logging Macros (Abseil Stream-style)
 *-----------------------------------------------------------------------------*
 * Ordered by severity (highest to lowest)
 * Usage: NIXL_INFO << "Message part 1 " << variable << " message part 2";
 */

/* --- Fatal Severity --- */

/*
 * Logs a message and terminates the program unconditionally.
 * Maps to Abseil LOG(FATAL). Use for unrecoverable errors.
 */
#define NIXL_FATAL LOG(FATAL)

/* --- High Severity --- */

/* Logs messages unconditionally (maps to Abseil ERROR level) */
#define NIXL_ERROR LOG(ERROR)

/* --- Medium Severity --- */

/* Logs messages unconditionally (maps to Abseil WARNING level) */
#define NIXL_WARN LOG(WARNING)

/* --- Informational --- */

/*
 * Logs messages unconditionally (maps to Abseil INFO level)
 * Controlled at runtime by Abseil flags (e.g., --minloglevel)
 * Controlled at compile time by ABSL_MIN_LOG_LEVEL (strips if level < INFO)
 */
#define NIXL_INFO LOG(INFO)

/* --- Debug / Low Severity (Typically Debug Builds Only) --- */

/*
 * Logs messages only in debug builds (when NDEBUG is not defined)
 * Maps to Abseil DVLOG(1). Conceptually lower severity than INFO.
 * Stripped entirely if NDEBUG is defined OR ABSL_MIN_LOG_LEVEL > INFO.
 * Controlled by --v=1 or higher in debug builds.
 */
#define NIXL_DEBUG DVLOG(1)

/*
 * Logs verbose trace messages only in debug builds (when NDEBUG is not defined)
 * Maps to Abseil DVLOG(2). Conceptually the lowest severity.
 * Stripped entirely if NDEBUG is defined OR ABSL_MIN_LOG_LEVEL > INFO.
 * Example Prefix: NIXL_TRACE << "[Component] " << "Detailed message";
 * Controlled by --v=2 or higher in debug builds.
 */
#define NIXL_TRACE DVLOG(2)


/*-----------------------------------------------------------------------------*
 * Assertion Macros
 *-----------------------------------------------------------------------------*/

/*
 * Check condition in all builds (debug and release). Typically for critical invariants.
 * Terminates if condition is false.
 * Maps directly to Abseil CHECK. Allows streaming additional context:
 * NIXL_ASSERT_ALWAYS(size > 0) << "Size must be positive, got " << size;
 */
#define NIXL_ASSERT_ALWAYS(condition) CHECK(condition)


/*
 * Check condition in debug builds only (when NDEBUG is not defined). Used for heavier checks.
 * Terminates if condition is false.
 * Maps directly to Abseil DCHECK. Allows streaming additional context:
 * NIXL_ASSERT(ptr != nullptr) << "Pointer must not be null";
 */
#define NIXL_ASSERT(condition) DCHECK(condition)


#endif /* __NIXL_LOG_H */
