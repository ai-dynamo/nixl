/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include "posix_backend.h"
#include "backend/backend_plugin.h"

// Plugin type alias for convenience
using posix_plugin_t = nixlBackendPluginCreator<nixlPosixEngine>;

namespace {
nixl_b_params_t
makePosixPluginParams() {
    nixl_b_params_t params = {{"ios_pool_size", "65536"}, {"kernel_queue_size", "256"}};
#ifdef HAVE_LINUXAIO
    params["use_aio"] = "true";
#endif
#ifdef HAVE_LIBURING
    params["use_uring"] =
#ifdef HAVE_LINUXAIO
        "false";
#else
        "true";
#endif
#endif
#ifdef HAVE_POSIXAIO
    params["use_posix_aio"] =
#if defined(HAVE_LINUXAIO) || defined(HAVE_LIBURING)
        "false";
#else
        "true";
#endif
#endif
    return params;
}

const nixl_b_params_t posix_plugin_params = makePosixPluginParams();
} // namespace

#ifdef STATIC_PLUGIN_POSIX
nixlBackendPlugin *
createStaticPOSIXPlugin() {
    return posix_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "POSIX", "0.1.0", posix_plugin_params, {DRAM_SEG, FILE_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return posix_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "POSIX", "0.1.0", posix_plugin_params, {DRAM_SEG, FILE_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
