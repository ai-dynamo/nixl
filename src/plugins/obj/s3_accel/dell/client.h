/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_DELL_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_DELL_CLIENT_H

#include "s3_accel/client.h"
#include "cuobj_token_manager.h"
#include "nixl_types.h"

/**
 * S3 client for Dell ObjectScale with RDMA acceleration (Pattern B).
 *
 * Overrides the standard iS3Client methods putObjectAsync / getObjectAsync
 * to inject an RDMA token (x-rdma-info header) and send an empty HTTP body.
 * The Dell ObjectScale server reads from / writes to the client's memory
 * directly over RDMA.
 *
 * This class does NOT introduce a separate RDMA interface — it fulfills the
 * standard iS3Client contract.  DefaultObjEngineImpl::postXfer calls
 * getClient()->putObjectAsync() / getObjectAsync() and the RDMA path is
 * entirely transparent.
 */
class awsS3DellObsClient : public awsS3AccelClient {
public:
    /**
     * @param custom_params  S3 configuration (bucket, endpoint, credentials, ...).
     * @param token_mgr      Shared token manager for RDMA token generation.
     * @param executor       Optional async executor for the AWS SDK.
     */
    awsS3DellObsClient(nixl_b_params_t *custom_params,
                        std::shared_ptr<CuObjTokenManager> token_mgr,
                        std::shared_ptr<Aws::Utils::Threading::Executor>
                            executor = nullptr);

    virtual ~awsS3DellObsClient() = default;

    /**
     * Asynchronously put an object to Dell ObjectScale using RDMA.
     *
     * Generates an RDMA token via CuObjTokenManager, sets the x-rdma-info
     * header, and sends a PutObject request with ContentLength(0).
     * The server performs RDMA_READ from the client's registered memory.
     */
    void
    putObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   put_object_callback_t callback) override;

    /**
     * Asynchronously get an object from Dell ObjectScale using RDMA.
     *
     * Same pattern as putObjectAsync but for GET: the server performs
     * RDMA_WRITE into the client's registered memory.
     */
    void
    getObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) override;

private:
    std::shared_ptr<CuObjTokenManager> tokenMgr_;
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_CLIENT_H
