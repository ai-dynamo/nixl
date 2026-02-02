/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_DELL_RDMA_INTERFACE_H
#define NIXL_OBJ_PLUGIN_S3_DELL_RDMA_INTERFACE_H

#include <string_view>
#include <cstdint>
#include <functional>
#include "nixl_types.h"

/**
 * Interface for S3 clients that support RDMA operations.
 * Only Dell-specific clients implement this interface.
 */
class iS3RdmaClient {
public:
    virtual ~iS3RdmaClient() = default;

    /**
     * Asynchronously put an object to S3 using RDMA.
     * @param key The object key
     * @param data_ptr Pointer to the data to upload
     * @param data_len Length of the data in bytes
     * @param offset Offset within the object
     * @param rdma_desc RDMA descriptor for the transfer
     * @param callback Callback function to handle the result
     */
    virtual void
    putObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       put_object_callback_t callback) = 0;

    /**
     * Asynchronously get an object from S3 using RDMA.
     * @param key The object key
     * @param data_ptr Pointer to the buffer to store the downloaded data
     * @param data_len Maximum length of data to read
     * @param offset Offset within the object to start reading from
     * @param rdma_desc RDMA descriptor for the transfer
     * @param callback Callback function to handle the result
     */
    virtual void
    getObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       get_object_callback_t callback) = 0;
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_RDMA_INTERFACE_H
