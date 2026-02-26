/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H
#define NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H

#include "s3_accel/engine_impl.h"
#include "s3_accel/dell/client.h"
#include <cuobjclient.h>

/**
 * S3 Dell ObjectScale Engine Implementation.
 * This class provides RDMA-accelerated S3 object storage operations specifically
 * for Dell ObjectScale storage systems. It inherits from S3AccelObjEngineImpl and
 * implements the cuObject API for GPU-direct storage operations.
 */
class S3DellObsObjEngineImpl : public S3AccelObjEngineImpl {
public:
    /**
     * Constructor that initializes the S3 Dell ObjectScale engine.
     * Creates both the S3 client and cuObject client for RDMA operations.
     *
     * @param init_params Backend initialization parameters
     */
    explicit S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params);
    /**
     * Constructor that accepts an injected S3 client (for testing).
     * Creates the cuObject client for RDMA operations using the provided S3 client.
     *
     * @param init_params Backend initialization parameters
     * @param s3_client Pre-configured S3 client (can be mock for testing)
     */
    S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params,
                           std::shared_ptr<iS3Client> s3_client);

    /**
     * Register memory with the backend for RDMA operations.
     * Supports OBJ_SEG, DRAM_SEG, and VRAM_SEG memory types.
     * For OBJ_SEG, creates object metadata and maps device ID to object key.
     * For DRAM_SEG/VRAM_SEG, obtains cuObject descriptor for the memory region.
     *
     * @param mem Memory blob descriptor
     * @param nixl_mem Memory type
     * @param out Output backend metadata handle
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    /**
     * Deregister memory from the backend.
     * Cleans up cuObject descriptors and removes device ID mappings.
     *
     * @param meta Backend metadata handle to deregister
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    /**
     * Prepare a transfer operation between local and remote memory.
     * Validates parameters, sets up RDMA contexts using cuObject, and creates
     * transfer request handles for subsequent execution.
     *
     * @param operation Transfer operation (NIXL_READ or NIXL_WRITE)
     * @param local Local memory descriptor list
     * @param remote Remote memory descriptor list
     * @param remote_agent Remote agent identifier
     * @param local_agent Local agent identifier
     * @param handle Output transfer request handle
     * @param opt_args Optional backend arguments
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

    /**
     * Post a transfer operation for execution.
     * Initiates asynchronous S3 operations with RDMA descriptors obtained
     * from the preparation phase. Uses futures/promises to bridge callback
     * and polling interfaces.
     *
     * @param operation Transfer operation (NIXL_READ or NIXL_WRITE)
     * @param local Local memory descriptor list
     * @param remote Remote memory descriptor list
     * @param remote_agent Remote agent identifier
     * @param handle Transfer request handle from prepXfer
     * @param opt_args Optional backend arguments
     * @return NIXL_IN_PROG if operation started, error code on failure
     */
    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    /**
     * Check the status of an ongoing transfer operation.
     * Polls the futures associated with the transfer request to determine
     * completion status.
     *
     * @param handle Transfer request handle to check
     * @return NIXL_SUCCESS if completed, NIXL_IN_PROG if ongoing, error code on failure
     */
    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    /**
     * Release a transfer request handle.
     * Cleans up resources associated with the transfer request.
     *
     * @param handle Transfer request handle to release
     * @return NIXL_SUCCESS on success
     */
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    /**
     * Get the list of supported memory types.
     *
     * @return List of supported memory segments (OBJ_SEG, DRAM_SEG, VRAM_SEG)
     */
    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    }

protected:
    /**
     * Get the S3 client instance.
     *
     * @return Pointer to the S3 client interface
     */
    iS3Client *
    getClient() const override;

private:
    /// S3 client for Dell ObjectScale operations
    std::shared_ptr<iS3Client> s3Client_;
    /// cuObject client for RDMA operations
    std::shared_ptr<cuObjClient> cuClient_;
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H
