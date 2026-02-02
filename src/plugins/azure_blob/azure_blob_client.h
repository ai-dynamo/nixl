/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Microsoft Corporation.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AZURE_BLOB_CLIENT_H
#define AZURE_BLOB_CLIENT_H

#include <functional>
#include <memory>
#include <string_view>
#include <cstdint>
#include <azure/storage/blobs.hpp>
#include <asio.hpp>
#include "nixl_types.h"

using put_blob_callback_t = std::function<void(bool success)>;
using get_blob_callback_t = std::function<void(bool success)>;

/**
 * Abstract interface for Azure Blob client operations.
 * Provides async operations for PutBlob and GetBlob.
 */
class iBlobClient {
public:
    virtual ~iBlobClient() = default;

    /**
     * Set the executor for async operations.
     * @param executor The executor to use for async operations
     */
    virtual void
    setExecutor(std::shared_ptr<asio::thread_pool> executor) = 0;

    /**
     * Asynchronously put a blob to Azure Blob Storage.
     * @param blob_name The blob name
     * @param data_ptr Pointer to the data to upload
     * @param data_len Length of the data in bytes
     * @param offset Offset within the blob
     * @param callback Callback function to handle the result
     */
    virtual void
    putBlobAsync(std::string_view blob_name,
                 uintptr_t data_ptr,
                 size_t data_len,
                 size_t offset,
                 put_blob_callback_t callback) = 0;

    /**
     * Asynchronously get a blob from Azure Blob Storage.
     * @param blob_name The blob name
     * @param data_ptr Pointer to the buffer to store the downloaded data
     * @param data_len Maximum length of data to read
     * @param offset Offset within the object to start reading from
     * @param callback Callback function to handle the result
     */
    virtual void
    getBlobAsync(std::string_view blob_name,
                 uintptr_t data_ptr,
                 size_t data_len,
                 size_t offset,
                 get_blob_callback_t callback) = 0;

    /**
     * Check if the blob exists.
     * @param blob_name The blob name
     * @return true if the blob exists, false otherwise
     */
    virtual bool
    checkBlobExists(std::string_view blob_name) = 0;
};

/**
 * Concrete implementation of IBlobClient using Azure Blob SDK.
 */
class azureBlobClient : public iBlobClient {
public:
    /**
     * Constructor that creates an Azure BlobClient from custom parameters.
     * @param custom_params Custom parameters containing Azure Blob configuration
     * @param executor Optional executor for async operations
     */
    azureBlobClient(nixl_b_params_t *custom_params,
                    std::shared_ptr<asio::thread_pool> executor = nullptr);

    void
    setExecutor(std::shared_ptr<asio::thread_pool> executor) override;

    void
    putBlobAsync(std::string_view blob_name,
                 uintptr_t data_ptr,
                 size_t data_len,
                 size_t offset,
                 put_blob_callback_t callback) override;

    void
    getBlobAsync(std::string_view blob_name,
                 uintptr_t data_ptr,
                 size_t data_len,
                 size_t offset,
                 get_blob_callback_t callback) override;

    bool
    checkBlobExists(std::string_view blob_name) override;

private:
    std::shared_ptr<asio::thread_pool> executor_;
    std::unique_ptr<Azure::Storage::Blobs::BlobContainerClient> blobContainerClient_;
};

#endif // AZURE_BLOB_CLIENT_H
