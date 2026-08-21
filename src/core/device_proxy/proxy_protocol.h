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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H
#define NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H

#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

/** @brief Operation kinds carried by a proxy submission. */
enum class nixl_proxy_opcode_t : uint8_t {
    PUT = 0,
    ATOMIC_ADD = 1,
};

/** @brief Runtime control states visible to GPU producers. */
enum class nixl_proxy_control_state_t : uint32_t {
    RUNNING = 0,
    SHUTDOWN = 1,
};

struct nixlProxyDeviceContextData;

/** @brief GPU-visible proxy memory-view metadata. */
struct nixlProxyDeviceMemView {
    uint32_t proxyMemViewId = 0;
    uint32_t directPtrCount = 0;
    const nixlProxyDeviceContextData *context = nullptr;
    void *directPtrs[0];
};

/** @brief Fixed-size GPU-to-CPU proxy work-ring record. */
struct alignas(64) nixlProxySubmission {
    uint64_t opIdx = 0;
    uint64_t value = 0;
    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    uint64_t size = 0;
    nixl_proxy_opcode_t opcode = nixl_proxy_opcode_t::PUT;
    uint8_t flags = 0;
    uint16_t channelId = 0;
    uint32_t reserved = 0;
    uint32_t srcIndex = 0;
    uint32_t dstIndex = 0;
    uint32_t srcProxyMemViewId = 0;
    uint32_t dstProxyMemViewId = 0;
};

static_assert(sizeof(nixlProxySubmission) == 64, "nixlProxySubmission must be 64 bytes");
static_assert(offsetof(nixlProxySubmission, opIdx) == 0,
              "opIdx must be the first word because it publishes record readiness");

/** @brief GPU-visible aliases for one proxy work ring. */
struct nixlProxyWorkRing {
    /** Mapped host records: GPU writes via device alias; CPU worker reads host alias. */
    nixlProxySubmission *records = nullptr;
    /** Device-resident producer index; only the GPU updates it. */
    uint64_t *producerIdx = nullptr;
    /** Authoritative consumer index; CPU publishes through GDRCopy or mapped host memory. */
    uint64_t *consumerIdx = nullptr;
    /** Device-resident cached consumer index; GPU refreshes from consumer_idx only when full. */
    uint64_t *consumerIdxCache = nullptr;
    /** The depth of the work ring. */
    uint32_t depth = 0;
};

/** @brief Completion frontier and status for one proxy channel. */
struct alignas(16) nixlProxyCompletionSlot {
    uint64_t completedIdx = 0;
    nixl_status_t nextStatus = NIXL_IN_PROG;
};

/** @brief GPU-visible work and completion state for one channel. */
struct nixlProxyChannelView {
    nixlProxyWorkRing *workRing = nullptr;
    /** Mapped pinned host memory (device alias); host writes via host pointer with atomics. */
    nixlProxyCompletionSlot *completionSlot = nullptr;
};

/** @brief GPU-visible context shared by all proxy channels. */
struct nixlProxyDeviceContextData {
    nixlProxyChannelView *channels = nullptr;
    uint32_t maxPeers = 0;
    uint32_t numChannels = 0;
    uint64_t *shutdownWord = nullptr;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H
