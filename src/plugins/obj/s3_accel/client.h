/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_ACCEL_CLIENT_H
#define OBJ_PLUGIN_S3_ACCEL_CLIENT_H

#include <memory>
#include <string_view>

#include "s3/client.h"
#include "object/rdma/rdma.h"
#include "nixl_types.h"

/**
 * S3 Accelerated Object Client - the generic, protocol-compliant S3-over-RDMA
 * client, and the base for vendor-specific accelerated clients.
 *
 * When `accelerated=true` is requested with no `type` (or `type=s3`), this
 * client moves the object payload out-of-band over the published `x-amz-rdma-*`
 * protocol instead of streaming the body over HTTP: putObjectAsync/getObjectAsync
 * mint a cuObject token for the (engine-pinned) buffer and drive the RDMA
 * control plane. There is no HTTP fallback on that path - an unavailable fabric
 * or control plane fails construction, and a server decline is a hard error.
 *
 * Vendor clients (selected by an explicit `type`) inherit and manage their own
 * RDMA path; for them the generic fast path stays disabled
 * (isGenericAccelRequested is false), so this client behaves as the plain HTTP
 * base.
 */
class awsS3AccelClient : public awsS3Client {
public:
    awsS3AccelClient(nixl_b_params_t *custom_params,
                     std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    ~awsS3AccelClient() override = default;

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

    // The executor is captured at construction and the RDMA fast path is bound to
    // it, so it cannot be swapped afterward. Override the base (which throws) with
    // a no-op so an injected accelerated client can still be constructed.
    void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) override;

    /**
     * @brief Whether the generic S3-over-RDMA fast path is fully usable
     *        (generic accel requested, cuObject fabric + control plane +
     *        executor all ready). The engine uses this to gate VRAM_SEG
     *        advertisement and buffer pinning.
     */
    [[nodiscard]] bool
    supportsRdma() const;

private:
    [[nodiscard]] bool
    rdmaReady() const;

    std::shared_ptr<Aws::Utils::Threading::Executor> executor_;
    bool rdma_requested_ = false;
    nixl_obj_rdma::SharedCuObjClient *rdma_ = nullptr;
    std::unique_ptr<nixl_obj_rdma::S3RdmaControlPlane> rdmaCp_;
};

#endif // OBJ_PLUGIN_S3_ACCEL_CLIENT_H
