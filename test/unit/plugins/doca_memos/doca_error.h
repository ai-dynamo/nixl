/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * DOCA KV Mock - Error Types
 * Mock implementation of DOCA error types
 */

#ifndef NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_ERROR_H
#define NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_ERROR_H

#include <doca_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DOCA API return codes
 */
typedef enum doca_error {
    DOCA_SUCCESS = 0,
    DOCA_ERROR_UNKNOWN = 1,
    DOCA_ERROR_NOT_PERMITTED = 2,
    DOCA_ERROR_IN_USE = 3,
    DOCA_ERROR_NOT_SUPPORTED = 4,
    DOCA_ERROR_AGAIN = 5,
    DOCA_ERROR_INVALID_VALUE = 6,
    DOCA_ERROR_NO_MEMORY = 7,
    DOCA_ERROR_INITIALIZATION = 8,
    DOCA_ERROR_TIME_OUT = 9,
    DOCA_ERROR_SHUTDOWN = 10,
    DOCA_ERROR_CONNECTION_RESET = 11,
    DOCA_ERROR_CONNECTION_ABORTED = 12,
    DOCA_ERROR_CONNECTION_INPROGRESS = 13,
    DOCA_ERROR_NOT_CONNECTED = 14,
    DOCA_ERROR_NO_LOCK = 15,
    DOCA_ERROR_NOT_FOUND = 16,
    DOCA_ERROR_IO_FAILED = 17,
    DOCA_ERROR_BAD_STATE = 18,
    DOCA_ERROR_UNSUPPORTED_VERSION = 19,
    DOCA_ERROR_OPERATING_SYSTEM = 20,
    DOCA_ERROR_DRIVER = 21,
    DOCA_ERROR_UNEXPECTED = 22,
    DOCA_ERROR_ALREADY_EXIST = 23,
    DOCA_ERROR_FULL = 24,
    DOCA_ERROR_EMPTY = 25,
    DOCA_ERROR_IN_PROGRESS = 26,
    DOCA_ERROR_TOO_BIG = 27,
    DOCA_ERROR_AUTHENTICATION = 28,
    DOCA_ERROR_BAD_CONFIG = 29,
    DOCA_ERROR_SKIPPED = 30,
    DOCA_ERROR_DEVICE_FATAL_ERROR = 31
} doca_error_t;

/**
 * Returns the string representation of an error code name.
 */
const char *
doca_error_get_name(doca_error_t error);

/**
 * Returns the description string of an error code.
 */
const char *
doca_error_get_descr(doca_error_t error);

#ifdef __cplusplus
}
#endif

#endif /* NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_ERROR_H */
