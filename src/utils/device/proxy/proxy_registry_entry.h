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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_REGISTRY_ENTRY_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_REGISTRY_ENTRY_H

#include <vector>

#include "backend_aux.h"
#include "device/device_ops.h"
#include "nixl_types.h"

namespace nixl {

/**
 * The registry's device memview: kernels read the token and direct pointers through
 * it. The same type as nixlMemViewH; the name says which of its meanings is meant.
 */
using proxyViewHandle = nixlMemViewH;

/**
 * One prepared memory view. The registry creates and owns entries; a ring command
 * names one by its address (the 64-bit host token), and resolveSubmission() reads
 * it through that token without a lookup.
 */
struct registryEntry {
    struct storedDesc {
        /** metadataP is borrowed and must outlive submissions. */
        nixlMetaDesc desc;
        bool usable = true;
    };

    proxyViewHandle proxy_memview = nullptr;
    deviceMem proxy_memview_mem;
    bool remote = false;
    std::vector<storedDesc> descs;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_REGISTRY_ENTRY_H
