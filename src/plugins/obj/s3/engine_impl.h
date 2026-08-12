/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_ENGINE_IMPL_H
#define OBJ_PLUGIN_S3_ENGINE_IMPL_H

#include "obj_backend.h"

class DefaultObjEngineImpl : public nixlObjEngineImpl {
public:
    explicit DefaultObjEngineImpl(const nixlBackendInitParams *init_params);
    DefaultObjEngineImpl(const nixlBackendInitParams *init_params,
                         std::shared_ptr<iS3Client> s3_client,
                         std::shared_ptr<iS3Client> s3_client_crt);
    ~DefaultObjEngineImpl() override;

    nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG, OBJ_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;
    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;
    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;
    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

protected:
    virtual iS3Client *
    getClient() const;
    virtual iS3Client *
    getClientForSize(size_t data_len) const;

    std::shared_ptr<asioThreadPoolExecutor> executor_;
    std::shared_ptr<iS3Client> s3Client_;
    std::unordered_map<uint64_t, std::string> devIdToObjKey_;
    size_t crtMinLimit_;
};

#endif // OBJ_PLUGIN_S3_ENGINE_IMPL_H
