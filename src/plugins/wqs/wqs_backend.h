// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_WQS_WQS_BACKEND_H
#define NIXL_SRC_PLUGINS_WQS_WQS_BACKEND_H

#include "backend/backend_engine.h"
#include "wqs_session.h"

#include <memory>

inline constexpr const char *wqs_plugin_name = "WQS";
inline constexpr const char *wqs_plugin_version = "0.1.0";

/** Local whole-object transfers between caller DRAM and the WQS staging pool. */
class nixlWqsEngine final : public nixlBackendEngine {
public:
    explicit nixlWqsEngine(const nixlBackendInitParams *params);
    nixlWqsEngine(const nixlBackendInitParams *params, std::shared_ptr<nixlWqsSession> session);
    ~nixlWqsEngine() override;

    [[nodiscard]] bool
    supportsLocal() const override {
        return true;
    }

    [[nodiscard]] bool
    supportsRemote() const override {
        return false;
    }

    [[nodiscard]] bool
    supportsNotif() const override {
        return false;
    }

    [[nodiscard]] nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG, OBJ_SEG};
    }

    [[nodiscard]] nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &type, nixlBackendMD *&out) override;
    [[nodiscard]] nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;
    [[nodiscard]] nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;
    [[nodiscard]] nixl_status_t
    unloadMD(nixlBackendMD *meta) override;
    [[nodiscard]] nixl_status_t
    connect(const std::string &agent) override;
    [[nodiscard]] nixl_status_t
    disconnect(const std::string &agent) override;
    [[nodiscard]] nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    [[nodiscard]] nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *args = nullptr) const override;
    [[nodiscard]] nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *args = nullptr) const override;
    [[nodiscard]] nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    [[nodiscard]] nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

private:
    class impl;
    std::unique_ptr<impl> impl_;
};

#endif // NIXL_SRC_PLUGINS_WQS_WQS_BACKEND_H
