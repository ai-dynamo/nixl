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

#ifndef NIXL_SRC_PLUGINS_RADOS_NKV_RADOS_NKV_KEY_H
#define NIXL_SRC_PLUGINS_RADOS_NKV_RADOS_NKV_KEY_H

#include <cstdint>
#include <string>
#include <vector>

/**
 * Derive the fixed-length NVMe KV key from a token sequence (the OBJ_SEG
 * descriptor's metaInfo blob).
 *
 * Uses a 128-bit FNV-1a hash emitted big-endian and truncated to @p key_len, so
 * an arbitrary-length token sequence maps to a stable, fixed-size key. This is
 * intentionally free of any SPDK dependency so it can be unit-tested without a
 * KV target, and reproduced by tooling (e.g. the RADOS round-trip script
 * computes the same hash to find the expected object id).
 *
 * @param token_seq The token sequence to hash (the descriptor's metaInfo).
 * @param key_len   Desired key length in bytes; clamped to 16 (the hash width).
 * @param out       On success, set to the derived key (@p key_len bytes).
 * @return true on success; false if @p token_seq is empty or @p key_len is 0
 *         (no key produced).
 */
bool
radosNkvDeriveKey(const std::string &token_seq, uint8_t key_len, std::vector<uint8_t> &out);

#endif // NIXL_SRC_PLUGINS_RADOS_NKV_RADOS_NKV_KEY_H
