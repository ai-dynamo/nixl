/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "respkv_engine.h"

#include "common/nixl_log.h"

#include <absl/strings/str_format.h>

nixl_status_t
nixlRespKvEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    if (!input) {
        output = nullptr;
        return NIXL_ERR_INVALID_PARAM;
    }
    output = input;
    return NIXL_SUCCESS;
}

bool
nixlRespKvEngine::validatePrepCommon(const nixl_xfer_op_t &operation,
                                     const nixl_meta_dlist_t &local,
                                     const nixl_meta_dlist_t &remote,
                                     const std::string &remote_agent) const {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (local.descCount() == 0) {
        NIXL_ERROR << "Error: Transfer descriptor lists must not be empty";
        return false;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local and remote descriptor counts must match (%d != %d)",
            local.descCount(),
            remote.descCount());
        return false;
    }

    if (remote_agent != localAgent) {
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            localAgent,
            remote_agent);
    }

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    return true;
}
