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

#include "nixl_types.h"
#include "obj_backend.h"
#include "backend/backend_plugin.h"
#include "common/nixl_log.h"

namespace {
nixl_b_params_t
get_obj_options() {
    nixl_b_params_t params;
    params["access_key"] = "AWS access key ID (required)";
    params["secret_key"] = "AWS secret access key (required)";
    params["session_token"] = "AWS session token (optional)";
    return params;
}

nixl_mem_list_t
get_obj_mems() {
    return {DRAM_SEG, OBJ_SEG};
}
}

// Define the complete OBJ plugin using the template
NIXL_DEFINE_PLUGIN(OBJ, nixlObjEngine, "0.1.0", get_obj_options, get_obj_mems)
