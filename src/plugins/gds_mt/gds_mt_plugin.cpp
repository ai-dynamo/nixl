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
#include "gds_mt_backend.h"
#include "common/nixl_log.h"
#include <exception>

namespace {
nixl_b_params_t
get_gds_mt_options() {
    return {};
}

nixl_mem_list_t
get_gds_mt_mems() {
    return {DRAM_SEG, VRAM_SEG, FILE_SEG};
}
} // namespace

// Define the complete GDS_MT plugin using the template
NIXL_DEFINE_PLUGIN(GDS_MT, nixlGdsMtEngine, "0.1.0", get_gds_mt_options, get_gds_mt_mems)
