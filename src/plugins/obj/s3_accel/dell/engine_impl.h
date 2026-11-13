/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H
#define NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H

#include "s3_accel/engine_impl.h"

/**
 * @brief Dell ObjectScale accelerated OBJ engine.
 *
 * Keeps Dell's x-rdma-info wire protocol, but uses the shared cuObject
 * registration/token lifecycle and standard S3 transfer orchestration.
 */
class S3DellObsObjEngineImpl : public S3AccelObjEngineImpl {
public:
    /**
     * @brief Creates the engine using the process-wide cuObject client.
     * @param init_params Backend initialization parameters.
     * @throws std::runtime_error If no connected cuObject RDMA fabric is available.
     */
    explicit S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params);

    /**
     * @brief Creates the engine with an injected S3 client and cuObject client.
     * @param init_params Backend initialization parameters.
     * @param s3_client S3 client to use; a Dell client is created when null.
     * @param rdma Non-owning cuObject client; the process-wide instance is used when null.
     *             It must outlive this engine and every RDMA token obtained through it,
     *             including tokens held by in-flight requests.
     * @throws std::runtime_error If the cuObject client is not connected.
     */
    S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params,
                           std::shared_ptr<iS3Client> s3_client,
                           nixl_obj_rdma::SharedCuObjClient *rdma = nullptr);

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

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

protected:
    nixl_obj_rdma::SharedCuObjClient *
    rdmaClient() const override;

private:
    nixl_obj_rdma::SharedCuObjClient *rdma_;
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H
