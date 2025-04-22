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
#include "backend/backend_engine.h"
#include "backend/backend_plugin.h"

namespace mocks {
namespace basic_plugin {

static constexpr const char *plugin_name = "MOCK_BASIC";
static constexpr const char *plugin_version = "0.0.1";

static nixlBackendEngine *create_engine(const nixlBackendInitParams *) {
  return nullptr;
}

static void destroy_engine(nixlBackendEngine *) {}

static const char *get_plugin_name() { return plugin_name; }

static const char *get_plugin_version() { return plugin_version; }

static nixl_b_params_t get_backend_options() { return nixl_b_params_t(); }

static nixlBackendPlugin plugin = {
  NIXL_PLUGIN_API_VERSION,
  create_engine,
  destroy_engine,
  get_plugin_name,
  get_plugin_version,
  get_backend_options
};
} // namespace basic_plugin

} // namespace mocks

extern "C" nixlBackendPlugin *nixl_plugin_init() {
  return &mocks::basic_plugin::plugin;
}

extern "C" void nixl_plugin_fini() {}
