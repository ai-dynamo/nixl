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
 * DOCA KV Mock - Compatibility Definitions
 * Mock implementation of DOCA compatibility macros
 */

#ifndef NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_COMPAT_H
#define NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_COMPAT_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__linux__)

#define DOCA_USED __attribute__((used))
#define DOCA_STABLE __attribute__((visibility("default"))) DOCA_USED

#ifndef DOCA_ALLOW_EXPERIMENTAL_API
#define DOCA_EXPERIMENTAL                                           \
    __attribute__((deprecated("Symbol is defined as experimental"), \
                   section(".text.experimental"))) DOCA_STABLE
#else
#define DOCA_EXPERIMENTAL __attribute__((section(".text.experimental"))) DOCA_STABLE
#endif

#else /* Windows or other */

#define __attribute__(_x_)
#define DOCA_STABLE
#define DOCA_EXPERIMENTAL

#endif

/* Compiler optimization hints */
#define doca_likely(x) __builtin_expect(!!(x), 1)
#define doca_unlikely(x) __builtin_expect(!!(x), 0)

#ifdef __cplusplus
}
#endif

#endif /* NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_COMPAT_H */
