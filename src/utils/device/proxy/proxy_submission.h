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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_SUBMISSION_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_SUBMISSION_H

#include <cstdint>

#include "nixl_types.h"
#include "proxy_transport.h"
#include "proxy_protocol.h"

namespace nixl {

/**
 * @brief Resolve a ring command's view tokens into the submission a backend posts to `peer`.
 *
 * Reads the registry entries the command names through their tokens and never
 * touches the registry's map, so it runs on worker threads without a lock. The
 * caller keeps those views registered until the command completes. `channel` is the
 * ring's channel; the command does not carry one.
 *
 * @param[out] out Unchanged on failure.
 * @retval NIXL_ERR_NOT_FOUND A view token is not set.
 * @retval NIXL_ERR_INVALID_PARAM Wrong view role, descriptor index or range.
 * @retval NIXL_ERR_NOT_SUPPORTED Unknown opcode.
 */
[[nodiscard]] nixl_status_t
resolveSubmission(const nixlProxyCommand &command,
                  uint32_t channel,
                  uint32_t peer,
                  proxyBackendSubmission &out) noexcept;

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_SUBMISSION_H
