/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_EVERPURE_ENGINE_IMPL_H
#define NIXL_OBJ_PLUGIN_S3_EVERPURE_ENGINE_IMPL_H

#if defined(HAVE_CUOBJ_CLIENT)

#include "s3_accel/engine_impl.h"
#include "s3_accel/everpure/client.h"
#include <cuobjclient.h>

/**
 * RDMA-accelerated S3 object engine for cuObject-compatible endpoints,
 * registered under `type: everpure` -- see client.h.
 *
 * Talks to the cuObject client library directly (cuobjclient /
 * libcuobjclient, via <cuobjclient.h>).
 */
class S3EverpureObjEngineImpl : public S3AccelObjEngineImpl {
public:
    explicit S3EverpureObjEngineImpl(const nixlBackendInitParams *init_params);
    S3EverpureObjEngineImpl(const nixlBackendInitParams *init_params,
                            std::shared_ptr<iS3Client> s3_client);

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

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

    // The cuObject path moves data straight out of GPU memory, so VRAM_SEG
    // is added on top of the DRAM_SEG/OBJ_SEG pair the base engine already
    // advertises.
    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    }

protected:
    iS3Client *
    getClient() const override;

private:
    /// Fails fast with NIXL_ERR_BACKEND when the cuObject client dropped its
    /// connection; shared by every entry point that touches cuClient_.
    nixl_status_t
    requireCuObjReady() const;

    std::shared_ptr<iS3Client> s3Client_;
    std::shared_ptr<cuObjClient> cuClient_;
};

#endif // HAVE_CUOBJ_CLIENT

#endif // NIXL_OBJ_PLUGIN_S3_EVERPURE_ENGINE_IMPL_H
