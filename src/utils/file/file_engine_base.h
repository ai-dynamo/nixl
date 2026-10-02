/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef NIXL_SRC_UTILS_FILE_FILE_ENGINE_BASE_H
#define NIXL_SRC_UTILS_FILE_FILE_ENGINE_BASE_H

#include <string>

#include <nixl.h>
#include <nixl_types.h>

#include "backend/backend_engine.h"

/**
 * @brief Shared trivial nixlBackendEngine overrides for local file backends.
 *
 * Local storage backends (cuFile/GDS, hipFile/AIS) are all local-only, notif-less
 * engines over the same {DRAM, VRAM, FILE} memory set, so the connection and
 * metadata entry points collapse to constants. Concrete engines derive from this
 * and add their platform-specific registration and transfer behavior.
 */
class FileEngineBase : public nixlBackendEngine {
public:
    explicit FileEngineBase(const nixlBackendInitParams *init_params)
        : nixlBackendEngine(init_params) {}

    bool
    supportsNotif() const override {
        return false;
    }

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG, VRAM_SEG, FILE_SEG};
    }

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }
};

#endif // NIXL_SRC_UTILS_FILE_FILE_ENGINE_BASE_H
