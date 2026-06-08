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

#ifndef NIXL_SRC_PLUGINS_UCX_TYPES_H
#define NIXL_SRC_PLUGINS_UCX_TYPES_H

#include <deque>
#include <map>
#include <memory>
#include <variant>

class nixlUcxConnection;

using ucx_connection_ptr_t = std::shared_ptr<nixlUcxConnection>;

namespace nixl::ucx {

struct recvMapKey;

struct recvMapPostValue;
struct recvMapRndvValue;

using recv_map_value_t = std::variant<std::deque<recvMapPostValue>, std::deque<recvMapRndvValue>>;

using recv_map_t = std::map<recvMapKey, recv_map_value_t, std::less<void>>;

} // namespace nixl::ucx

#endif
