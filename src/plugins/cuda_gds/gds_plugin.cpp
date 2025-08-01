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
#include "gds_backend.h"

namespace {
nixl_b_params_t
get_gds_options() {
    nixl_b_params_t params;
    return params;
}

nixl_mem_list_t
get_gds_mems() {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    mems.push_back(FILE_SEG);
    return mems;
}
} // namespace

// Define the complete GDS plugin using the template
NIXL_DEFINE_PLUGIN(GDS, nixlGdsEngine, "0.1.1", get_gds_options, get_gds_mems)
