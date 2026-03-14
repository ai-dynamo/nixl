/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Dell Technologies Inc. All rights reserved.
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
#include "libblkio_backend.h"
#include "backend/backend_plugin.h"

using blkio_plugin_t = nixlBackendPluginCreator<nixlLibblkioEngine>;

namespace {

[[nodiscard]] nixl_b_params_t
get_libblkio_backend_options()
{
    nixl_b_params_t params;
    params["api_type"] = "IO_URING";
    params["device_list"] = "";
    params["direct_io"] = "0";
    return params;
}

} // namespace

#ifdef STATIC_PLUGIN_LIBBLKIO
nixlBackendPlugin *
createStaticLIBBLKIOPlugin()
{
    return blkio_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                  "LIBBLKIO",
                                  "0.1.0",
                                  get_libblkio_backend_options(),
                                  {DRAM_SEG, BLK_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init()
{
    return blkio_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                  "LIBBLKIO",
                                  "0.1.0",
                                  get_libblkio_backend_options(),
                                  {DRAM_SEG, BLK_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini()
{
}
#endif
