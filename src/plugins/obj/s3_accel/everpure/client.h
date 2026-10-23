/*
 * Copyright 2026 Everpure, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_EVERPURE_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_EVERPURE_CLIENT_H

#include <memory>
#include <string>
#include <string_view>
#include <cstdint>
#include <aws/s3/S3Client.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include "s3_accel/client.h"
#include "rdma_interface.h"
#include "nixl_types.h"

/**
 * S3 Accelerated Object Client for cuObject-compatible RDMA endpoints -
 * inherits from Accelerated S3 Client, presenting Put/GetObject over RDMA
 * via the cuObject API (cuobjclient / libcuobjclient). Defaults are tuned
 * for FlashBlade, but every protocol detail is overridable so another
 * cuObject-compatible endpoint can reuse it under the same `everpure` type.
 *
 * The RDMA descriptor travels as a bare request header (`x-amz-rdma-token`
 * by default) -- see client.cpp. Header names and this client's connection
 * pool/timeout budget (separate from the default S3 engine's) override via
 * environment variable (`NIXL_EVERPURE_RDMA_TOKEN_HEADER`,
 * `NIXL_EVERPURE_RDMA_REPLY_HEADER`, `NIXL_EVERPURE_MAX_CONNECTIONS`,
 * `NIXL_EVERPURE_CONNECT_TIMEOUT_MS`, `NIXL_EVERPURE_REQUEST_TIMEOUT_MS`).
 * The `x-amz-content-sha256` header's value overrides via custom_params
 * instead (`content_sha256_value`).
 */
class awsS3EverpureClient : public awsS3AccelClient, public iEverpureS3RdmaClient {
public:
    /**
     * Constructor that creates an AWS S3 client for RDMA-accelerated
     * transfers from custom parameters.
     * @param custom_params Custom parameters containing S3 configuration
     * @param executor Optional executor for async operations
     */
    awsS3EverpureClient(nixl_b_params_t *custom_params,
                        std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    virtual ~awsS3EverpureClient() = default;

    /**
     * Asynchronously puts an object to the S3 endpoint using RDMA
     * acceleration.
     *
     * @param key The object key to store
     * @param data_ptr Pointer to the data buffer
     * @param data_len Length of the data to transfer
     * @param offset Offset within the object (must be 0; whole-object writes only)
     * @param rdma_desc RDMA descriptor for acceleration
     * @param callback Callback function invoked on completion
     */
    void
    putObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       put_object_callback_t callback);

    /**
     * Asynchronously gets an object from the S3 endpoint using RDMA
     * acceleration.
     *
     * @param key The object key to retrieve
     * @param data_ptr Pointer to the buffer to fill
     * @param data_len Length of the data to transfer
     * @param offset Offset within the object
     * @param rdma_desc RDMA descriptor for acceleration
     * @param callback Callback function invoked on completion
     */
    void
    getObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       get_object_callback_t callback);

private:
    std::string rdmaTokenHeader_;
    std::string contentSha256Value_;
    std::string rdmaReplyHeader_;
};

#endif // NIXL_OBJ_PLUGIN_S3_EVERPURE_CLIENT_H
