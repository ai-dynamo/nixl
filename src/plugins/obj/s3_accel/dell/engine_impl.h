/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H
#define OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H

#include "s3_accel/engine_impl.h"
#include "s3_accel/dell/client.h"
#include <cuobjclient.h>

class S3DellObsObjEngineImpl : public S3AccelObjEngineImpl {
public:
    explicit S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params);
    S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params,
                         std::shared_ptr<iS3Client> s3_client,
                         std::shared_ptr<iS3Client> s3_client_accel = nullptr);

    nixl_status_t registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) override;

    nixl_status_t deregisterMem(nixlBackendMD *meta) override;

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
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    nixl_mem_list_t
    getSupportedMems() const {
        return {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    }

protected:
    iS3Client *
    getClient() const override;

private:
    std::shared_ptr<awsS3DellObsClient> s3Client_;
    std::shared_ptr<cuObjClient> cuClient_;
    std::unordered_map<std::string, std::string> objKeyToRDMADesc_;
};

#endif // OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H
