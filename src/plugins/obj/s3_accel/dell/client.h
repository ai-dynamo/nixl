/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_DELL_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_DELL_CLIENT_H

#include "s3_accel/client.h"
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>

// Dell's body-less S3 requests use x-rdma-info rather than x-amz-rdma-token.
class awsS3DellObsClient : public awsS3AccelClient {
public:
    /**
     * @brief Creates a Dell ObjectScale RDMA client.
     * @param custom_params Backend parameters (bucket, credentials, endpoint, ...).
     * @param rdma Non-owning reference to the cuObject client used to mint tokens. It must
     *             outlive this client and every token obtained through it, including tokens
     *             held by in-flight requests whose completion callbacks have not yet run.
     * @param executor Optional executor for asynchronous requests.
     */
    awsS3DellObsClient(nixl_b_params_t *custom_params,
                       nixl_obj_rdma::SharedCuObjClient &rdma,
                       std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    void
    putObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   put_object_callback_t callback) override;
    void
    getObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) override;

protected:
    // The SDK dispatch boundary lets tests inspect the actual request and hold
    // completion pending without a server or RDMA NIC.
    virtual void
    submitPut(const Aws::S3::Model::PutObjectRequest &request, put_object_callback_t callback);
    virtual void
    submitGet(const Aws::S3::Model::GetObjectRequest &request, get_object_callback_t callback);

private:
    nixl_obj_rdma::SharedCuObjClient &rdma_;
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_CLIENT_H
