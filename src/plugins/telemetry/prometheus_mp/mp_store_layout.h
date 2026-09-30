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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_STORE_LAYOUT_H
#define NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_STORE_LAYOUT_H

/**
 * @file mp_store_layout.h
 * @brief The bytes a multi-process telemetry store is made of.
 *
 * Private to the store implementation: everything else works with
 * storeWriter/storeSnapshot from mp_store.h and never sees this layout. Kept
 * apart so the on-disk format -- the one part two differently-built processes
 * have to agree on, and the one part that cannot be changed without a schema
 * bump -- reads as a single self-contained contract.
 */

#include "mp_store.h"

#include "common/nixl_log.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace nixl::telemetry::mp {

// "NIXLMPS1" as a little-endian tag; changing the layout must change either this
// or MP_STORE_SCHEMA_VERSION so stale-format files are rejected.
inline constexpr uint64_t MP_STORE_MAGIC = 0x3153504d4c58494eULL;

inline constexpr std::size_t MP_MAX_AGENT_NAME = 256;
inline constexpr std::size_t MP_MAX_HOSTNAME = 128;
inline constexpr std::size_t MP_MAX_LOCAL_RANK = 64;

// Fixed on-disk layout. Plain trivially-copyable POD operated on with __atomic
// builtins (not std::atomic) so it is safe to memset/reinterpret over an mmap'd
// region shared between processes. Field order keeps every uint64 8-byte aligned.
struct storeLayout {
    uint64_t magic;
    uint32_t schemaVersion;
    uint32_t slotCount;
    int64_t pid;
    uint64_t lastUpdateNs;
    uint64_t instance;
    // 64-bit purely so the double array that follows stays 8-byte aligned
    // without implicit padding.
    uint64_t bucketCount;
    char agentName[MP_MAX_AGENT_NAME];
    char hostname[MP_MAX_HOSTNAME];
    char localRank[MP_MAX_LOCAL_RANK];
    double bucketBounds[MP_STORE_MAX_BUCKETS];
    uint64_t counters[MP_STORE_SLOT_COUNT];
    uint64_t gauges[MP_STORE_SLOT_COUNT];
    uint64_t histBuckets[MP_STORE_SLOT_COUNT][MP_STORE_MAX_BUCKETS + 1];
    uint64_t histSums[MP_STORE_SLOT_COUNT];
};

// schemaVersion and slotCount alone do not catch a reordered field or a
// changed MP_MAX_* cap, both of which move every offset after them while a
// peer still validates the header. Failing the build is the only way to force
// the version bump that makes such a file rejectable.
static_assert(sizeof(storeLayout) == 6800,
              "storeLayout is an on-disk format: bump MP_STORE_SCHEMA_VERSION when its size "
              "changes, then update this assertion");

inline void
copyField(char *dst, std::size_t cap, const std::string &src, const char *what) {
    if (src.size() >= cap) {
        NIXL_WARN << "prometheus_mp: " << what << " '" << src << "' exceeds " << (cap - 1)
                  << " chars; truncating in telemetry store";
    }
    const std::size_t n = std::min(src.size(), cap - 1);
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

[[nodiscard]] inline std::string
readField(const char *src, std::size_t cap) {
    const std::size_t n = ::strnlen(src, cap);
    return std::string(src, n);
}

} // namespace nixl::telemetry::mp

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_MP_MP_STORE_LAYOUT_H
