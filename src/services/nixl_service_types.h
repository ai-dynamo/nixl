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
#ifndef NIXL_SERVICE_TYPES_H
#define NIXL_SERVICE_TYPES_H

#include "nixl_types.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"

#include <variant>
#include <string>
#include <unordered_map>
#include <functional>

struct nixlMarshalDirectConfig {};

struct nixlMarshalStagingConfig {};

struct nixlMarshalDeltaConfig {};

enum class nixl_marshal_compress_algo_t {
    ANS,
    /**
     * @brief ANS Delta compression algorithm.
     *
     * @details
     * The ANS Delta compression algorithm gets an extra "other" pointer to a memory buffer that
     * contains the previous value. Prior to ANS compression, the "src" is compared to the "other"
     * pointer and unchanged values are treated as zeros, which makes the compression more
     * efficient.
     */
    ANS_DELTA,
    BITCOMP,
};

struct nixlMarshalCompressConfig {
    nixl_marshal_compress_algo_t algo = nixl_marshal_compress_algo_t::ANS;
};

using nixl_marshal_config_t = std::variant<nixlMarshalDirectConfig,
                                           nixlMarshalStagingConfig,
                                           nixlMarshalDeltaConfig,
                                           nixlMarshalCompressConfig>;

struct nixlMarshalDirectOptArgs {};

struct nixlMarshalStagingOptArgs {};

struct nixlMarshalDeltaOptArgs {
    // The "ref" pointers' length must match the corresponding "src" pointer length.
    std::byte *senderRef = nullptr;
    std::byte *receiverRef = nullptr;
    nixl_mem_t senderMemType{};
    nixl_mem_t receiverMemType{};
    // Supported values are 1/2/4/8 bytes.
    size_t elementSize = 0;
};

struct nixlMarshalCompressOptArgs {
    // Only valid for the ANS_DELTA compression algorithm, must be present.
    std::optional<nixlMarshalDeltaOptArgs> delta;
};

using nixl_marshal_opt_args_t = std::variant<nixlMarshalDirectOptArgs,
                                             nixlMarshalStagingOptArgs,
                                             nixlMarshalDeltaOptArgs,
                                             nixlMarshalCompressOptArgs>;

namespace MarshalBackendSizing {
/**
 * @brief The number of slots per transfer.
 *        This is the number of slots that will be used for a single transfer.
 */
constexpr size_t slots_per_transfer = 2;

/** @brief Minimum alignment for staging-slot base addresses and strides. */
constexpr size_t slot_stride_alignment = 8;
} // namespace MarshalBackendSizing

struct nixlServiceAgentConfig : nixlAgentConfig {
    nixl_marshal_config_t mode = nixlMarshalDirectConfig{};
};

struct nixl_service_opt_args_t : nixl_opt_args_t {
    /**
     * @brief The marshal optional arguments.
     *
     * @details
     * The marshal optional arguments are used to configure the marshal backend.
     * The marshal must be direct or the same as the marshal configuration.
     *
     * @note If not set or extra_params is nullptr, the transfer uses the marshal
     *       mode configured in nixlServiceAgentConfig (via makeXferReq and createXferReq).
     */
    std::optional<nixl_marshal_opt_args_t> marshalOptArgs = std::nullopt;
};

#endif // NIXL_SERVICE_TYPES_H
