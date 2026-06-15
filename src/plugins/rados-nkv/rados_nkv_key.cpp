/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
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

#include "rados_nkv_key.h"

#include <algorithm>

bool
radosNkvDeriveKey(const std::string &token_seq, uint8_t key_len, std::vector<uint8_t> &out) {
    if (token_seq.empty() || key_len == 0) {
        return false;
    }

    // FNV-1a, 128-bit (offset basis and prime per the FNV spec).
    const __uint128_t k_offset =
        (static_cast<__uint128_t>(0x6c62272e07bb0142ULL) << 64) | 0x62b821756295c58dULL;
    const __uint128_t k_prime =
        (static_cast<__uint128_t>(0x0000000001000000ULL) << 64) | 0x000000000000013bULL;

    __uint128_t h = k_offset;
    for (unsigned char c : token_seq) {
        h ^= static_cast<__uint128_t>(c);
        h *= k_prime;
    }

    // Serialize big-endian, then keep the first key_len bytes.
    uint8_t buf[16];
    for (int i = 15; i >= 0; --i) {
        buf[i] = static_cast<uint8_t>(h & 0xff);
        h >>= 8;
    }
    out.assign(buf, buf + std::min<size_t>(key_len, sizeof(buf)));
    return true;
}
