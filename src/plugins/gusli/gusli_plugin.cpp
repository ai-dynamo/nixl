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
#include "gusli_backend.h"
static nixlBackendEngine* create_gusli_engine(const nixlBackendInitParams* init_params) {
	return new nixlGusliEngine(init_params);
}
static void destroy_gusli_engine(nixlBackendEngine *engine) { delete engine; }
static const char* get_plugin_name() { return "GUSLI"; }	// PLUGIN_NAME
static const char* get_plugin_version() { return "0.1.0"; }	// PLUGIN_VERSION
static nixl_b_params_t get_backend_options() { return nixl_b_params_t(); }
static nixlBackendPlugin plugin = {
	NIXL_PLUGIN_API_VERSION,
	create_gusli_engine,
	destroy_gusli_engine,
	get_plugin_name,
	get_plugin_version,
	get_backend_options,
	__getSupportedGusliMems
};

#ifdef STATIC_PLUGIN_GUSLI
	nixlBackendPlugin* createStaticGusliPlugin() { return &plugin; }
#else
	extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin* nixl_plugin_init() { return &plugin; }
	extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {}
#endif
