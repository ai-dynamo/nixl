/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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
#include "nixl_types.h"

#include "tcpxo_backend.h"

// Plugin type alias for convenience
using tcpxo_plugin_t = nixlBackendPluginCreator<tcpxo::nixlTcpxoEngine>;

#ifdef STATIC_PLUGIN_TCPXO
nixlBackendPlugin *
createStaticTCPXOPlugin() {
    return tcpxo_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                  "TCPXO",
                                  "0.1.0",
                                  {},
                                  {
                                      VRAM_SEG,
                                  });
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return tcpxo_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                  "TCPXO",
                                  "0.1.0",
                                  {},
                                  {
                                      VRAM_SEG,
                                  });
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
