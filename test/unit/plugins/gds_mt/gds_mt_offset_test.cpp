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

#include <cassert>
#include <cstdint>
#include "gds_mt_backend.h"

namespace {

void
testInteriorDescriptorUsesRegisteredBase() {
    alignas(4096) char registered[4096]{};
    const uintptr_t descriptor_addr = reinterpret_cast<uintptr_t> (registered) + 1024;

    gdsMtResolvedBuffer resolved{};
    const nixl_status_t status = gdsMtResolveRegisteredBuffer(registered,
                                                              sizeof (registered),
                                                              descriptor_addr,
                                                              512,
                                                              resolved);

    assert (status == NIXL_SUCCESS);
    assert (resolved.devPtrBase == registered);
    assert (resolved.devPtrOffset == 1024);
}

void
testExactBaseUsesZeroOffset() {
    alignas(4096) char registered[4096]{};

    gdsMtResolvedBuffer resolved{};
    const nixl_status_t status = gdsMtResolveRegisteredBuffer(registered,
                                                              sizeof (registered),
                                                              reinterpret_cast<uintptr_t> (registered),
                                                              sizeof (registered),
                                                              resolved);

    assert (status == NIXL_SUCCESS);
    assert (resolved.devPtrBase == registered);
    assert (resolved.devPtrOffset == 0);
}

void
testDescriptorBeforeRegisteredBaseIsRejected() {
    alignas(4096) char registered[4096]{};

    gdsMtResolvedBuffer resolved{};
    const nixl_status_t status = gdsMtResolveRegisteredBuffer(registered,
                                                              sizeof (registered),
                                                              reinterpret_cast<uintptr_t> (registered) - 1,
                                                              1,
                                                              resolved);

    assert (status == NIXL_ERR_INVALID_PARAM);
}

void
testDescriptorPastRegisteredEndIsRejected() {
    alignas(4096) char registered[4096]{};

    gdsMtResolvedBuffer resolved{};
    const nixl_status_t status = gdsMtResolveRegisteredBuffer(registered,
                                                              sizeof (registered),
                                                              reinterpret_cast<uintptr_t> (registered) + 3584,
                                                              1024,
                                                              resolved);

    assert (status == NIXL_ERR_INVALID_PARAM);
}

} // namespace

int
main() {
    testInteriorDescriptorUsesRegisteredBase();
    testExactBaseUsesZeroOffset();
    testDescriptorBeforeRegisteredBaseIsRejected();
    testDescriptorPastRegisteredEndIsRejected();
    return 0;
}
