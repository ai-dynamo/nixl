/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Keep this translation unit free of torch: configuration.h reaches absl/log,
// which defines CHECK, LOG_IF and DCHECK, and torch defines them too.

#include "ep_config.hpp"

#include "configuration.h"

namespace nixl_ep {

bool
ht_avoid_record_stream_enabled() {
    return nixl::config::getValueDefaulted<bool>("NIXL_EP_HT_AVOID_RECORD_STREAM", false);
}

} // namespace nixl_ep
