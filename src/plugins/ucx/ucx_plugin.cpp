/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "backend/backend_plugin.h"
#include "common/configuration.h"
#include "common/nixl_log.h"
#include "ucx_backend.h"

#include <dlfcn.h>
#include <string>

extern "C" {
#include <ucp/api/ucp.h>
}

// Plugin type alias for convenience
using ucx_plugin_t = nixlBackendPluginCreator<nixlUcxEngine>;

namespace {
constexpr const char *kExpectedUcxSonameVar = "NIXL_UCX_EXPECTED_SONAME";

std::string
getUcxSymbolPath() {
    Dl_info info{};

    if (dladdr(reinterpret_cast<void *>(ucp_get_version_string), &info) == 0 ||
        info.dli_fname == nullptr) {
        return "<unknown>";
    }

    return info.dli_fname;
}

bool
validateUcxBinding() {
    const std::string symbol_path = getUcxSymbolPath();
    NIXL_INFO << "NIXL UCX backend bound to UCX " << ucp_get_version_string()
              << " at " << symbol_path;

    const auto expected_soname =
        nixl::config::getValueOptional<std::string>(kExpectedUcxSonameVar);
    if (!expected_soname || expected_soname->empty()) {
        return true;
    }

    if (symbol_path.find(*expected_soname) != std::string::npos) {
        return true;
    }

    NIXL_ERROR << kExpectedUcxSonameVar << "=" << *expected_soname
               << " but NIXL UCX backend bound to " << symbol_path;
    return false;
}

nixlBackendPlugin *
createUcxPlugin() {
    if (!validateUcxBinding()) {
        return nullptr;
    }

    return ucx_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                "UCX",
                                "0.1.0",
                                get_ucx_backend_common_options(),
                                {DRAM_SEG, VRAM_SEG});
}
} // namespace

#ifdef STATIC_PLUGIN_UCX
nixlBackendPlugin *
createStaticUCXPlugin() {
    return createUcxPlugin();
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return createUcxPlugin();
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
