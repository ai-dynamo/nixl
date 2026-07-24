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

#ifndef WQSKV_HELPERS_H
#define WQSKV_HELPERS_H

#include "backend/backend_engine.h"
#include "kv_interface.h"
#include "nixl_types.h"
#include <string>
#include <vector>

namespace wqskv {

// Returns customParams["config_path"] when set and non-empty; otherwise falls
// back to the WDS_BACKEND_CONFIG_PATH environment variable. Returns an empty
// string when neither source provides a path.
std::string
resolveConfigPath(const nixlBackendInitParams *init_params);

// Loads vendor lib config from a JSON file into `opts`. The schema mirrors
// mooncake/mooncake-store/src/wds/wds_backend.cpp so a single config file can
// be shared between consumers. Returns false on open/parse error or invalid
// thread_mode / bind_cpus content.
bool
loadKVCacheOptionsFromJson(const std::string &conf_path, KVCacheOptions &opts);

// Splits a '\n'-delimited blob into `out`. Returns true and populates `out`
// only when the resulting key count matches `expected_count` exactly;
// otherwise clears `out` and returns false.
bool
parseCustomParamKeys(const std::string &blob, int expected_count, std::vector<std::string> &out);

// Validates prepXfer's operation type and that both descriptor lists are
// DRAM_SEG (the only mem type WQSKV supports).
bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote);

} // namespace wqskv

#endif // WQSKV_HELPERS_H
