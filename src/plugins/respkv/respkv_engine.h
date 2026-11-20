/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_RESPKV_RESPKV_ENGINE_H
#define NIXL_SRC_PLUGINS_RESPKV_RESPKV_ENGINE_H

#include "backend/backend_engine.h"

#include <string>

/** Thin base for RESP-family local KV plugins (REDIS, VALKEY, …). */
class nixlRespKvEngine : public nixlBackendEngine {
public:
    explicit nixlRespKvEngine(const nixlBackendInitParams *init_params)
        : nixlBackendEngine(init_params) {}

    ~nixlRespKvEngine() override = default;

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG};
    }

    nixl_status_t
    connect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;

protected:
    /** Shared prep checks. Does not validate remote segment type. */
    bool
    validatePrepCommon(const nixl_xfer_op_t &operation,
                       const nixl_meta_dlist_t &local,
                       const nixl_meta_dlist_t &remote,
                       const std::string &remote_agent) const;
};

#endif
