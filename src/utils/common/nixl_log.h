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

#include <system_error>
#include "absl/log/log.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"

/*-----------------------------------------------------------------------------*
 * Logging Macros (Abseil Stream-style)
 *-----------------------------------------------------------------------------*
 * Ordered by severity (highest to lowest)
 * Usage: NIXL_INFO << "Message part 1 " << variable << " message part 2";
 */

/*
 * Logs a message and terminates the program unconditionally.
 * Maps to Abseil LOG(FATAL). Use for unrecoverable errors.
 */
#define NIXL_FATAL LOG(FATAL)

/*
 * Like NIXL_FATAL, but also prints the errno message.
 */
#define NIXL_PFATAL NIXL_FATAL.WithPerror()

/* Logs messages unconditionally (maps to Abseil ERROR level) */
#define NIXL_ERROR LOG(ERROR)

/*
 * Like NIXL_ERROR, but also prints the errno message.
 */
#define NIXL_PERROR NIXL_ERROR.WithPerror()

/*
 * Like NIXL_ERROR, but prefixed with current function name and a colon
 */
#define NIXL_ERROR_FUNC NIXL_ERROR << __FUNCTION__ << ": "

/*
 * Logs messages unconditionally (maps to Abseil WARNING level)
 */
#define NIXL_WARN LOG(WARNING)

/*
 * Like NIXL_WARN, but also prints the errno message.
 */
#define NIXL_PWARN NIXL_WARN.WithPerror()

/*
 * Logs messages unconditionally (maps to Abseil INFO level)
 */
#define NIXL_INFO LOG(INFO)

/*
 * Like NIXL_INFO, but also prints the errno message.
 */
#define NIXL_PINFO NIXL_INFO.WithPerror()

/*
 * Logs messages unconditionally (maps to Abseil verbosity level 1)
 */
#define NIXL_DEBUG VLOG(1)

/*
 * Like NIXL_DEBUG, but also prints the errno message.
 */
#define NIXL_PDEBUG NIXL_DEBUG.WithPerror()

/*
 * Logs messages unconditionally (maps to Abseil verbosity level 2)
 * Stripped from release buids.
 */
#define NIXL_TRACE DVLOG(2)

/*
 * Like NIXL_TRACE, but also prints the errno message.
 */
#define NIXL_PTRACE NIXL_TRACE.WithPerror()

/*-----------------------------------------------------------------------------*
 * Assertion Macros
 *-----------------------------------------------------------------------------*/

/*
 * Check condition in all builds (debug and release). For critical invariants.
 * Terminates program if condition is false.
 * Allows streaming additional context:
 *      NIXL_ASSERT_ALWAYS(size > 0) << "Size must be positive, got " << size;
 */
#define NIXL_ASSERT_ALWAYS(condition) CHECK(condition)

/*
 * Check condition in debug builds only. Used for heavier checks.
 * Terminates program if condition is false.
 * Allows streaming additional context:
 *      NIXL_ASSERT(ptr != nullptr) << "Pointer must not be null";
 */
#define NIXL_ASSERT(condition) DCHECK(condition)

/*-----------------------------------------------------------------------------*
 * Helper Functions
 *-----------------------------------------------------------------------------*/

/*
 * Get the error message for the given error number. Thread-safe.
 * @param err: The error number.
 * @return: The error message.
 */
static inline std::string nixl_strerror(int err) {
    return std::error_code(err, std::generic_category()).message();
}

#endif /* __NIXL_LOG_H */
