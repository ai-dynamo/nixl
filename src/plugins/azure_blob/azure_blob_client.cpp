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

#include "azure_blob_client.h"
#include <asio.hpp>
#include <azure/core/http/curl_transport.hpp>
#include <azure/storage/blobs.hpp>
#include <azure/identity/default_azure_credential.hpp>
#include <optional>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <absl/strings/str_format.h>
#include "common/backend.h"
#include "common/configuration.h"
#include "nixl_types.h"

namespace {

[[nodiscard]] std::string
getAccountUrl(nixl_b_params_t *custom_params) {
    const auto str = nixl::getBackendParamDefaulted(custom_params, "account_url", std::string());
    if (!str.empty()) {
        return str;
    }

    return nixl::config::getValueDefaulted("AZURE_STORAGE_ACCOUNT_URL", std::string());
}

[[nodiscard]] std::string
getContainerName(nixl_b_params_t *custom_params) {
    const auto str = nixl::getBackendParamDefaulted(custom_params, "container_name", std::string());
    if (!str.empty()) {
        return str;
    }

    return nixl::config::getNonEmptyString("AZURE_STORAGE_CONTAINER_NAME");
}

[[nodiscard]] std::string
getConnectionString(nixl_b_params_t *custom_params) {
    const auto str = nixl::getBackendParamDefaulted(custom_params, "connection_string", std::string());
    if (!str.empty()) {
        return str;
    }

    return nixl::config::getValueDefaulted("AZURE_STORAGE_CONNECTION_STRING", std::string());
}

[[nodiscard]] std::string
getCaBundle(nixl_b_params_t *custom_params) {
    const auto str = nixl::getBackendParamDefaulted(custom_params, "ca_bundle", std::string());
    if (!str.empty()) {
        return str;
    }

    // Return empty string if not provided, which means use default CA bundle
    return nixl::config::getValueDefaulted("AZURE_CA_BUNDLE", std::string());
}

} // namespace

azureBlobClient::azureBlobClient(nixl_b_params_t *custom_params,
                                 std::shared_ptr<asio::thread_pool> executor) {
    executor_ = executor;
    const std::string accountUrl = ::getAccountUrl(custom_params);
    const std::string containerName = ::getContainerName(custom_params);
    const std::string connectionString = ::getConnectionString(custom_params);
    Azure::Storage::Blobs::BlobClientOptions options;
    options.Telemetry.ApplicationId = "azpartner-nixl/0.1.0";

    const std::string caBundle = ::getCaBundle(custom_params);
    if (!caBundle.empty()) {
        Azure::Core::Http::CurlTransportOptions curlOptions;
        curlOptions.CAInfo = caBundle;
        options.Transport.Transport =
            std::make_shared<Azure::Core::Http::CurlTransport>(curlOptions);
    }

    std::unique_ptr<Azure::Storage::Blobs::BlobServiceClient> blobServiceClient;
    if (!connectionString.empty()) {
        blobServiceClient = std::make_unique<Azure::Storage::Blobs::BlobServiceClient>(
            Azure::Storage::Blobs::BlobServiceClient::CreateFromConnectionString(connectionString,
                                                                                 options));
    } else if (!accountUrl.empty()) {
        blobServiceClient = std::make_unique<Azure::Storage::Blobs::BlobServiceClient>(
            accountUrl, std::make_shared<Azure::Identity::DefaultAzureCredential>(), options);
    } else {
        throw std::runtime_error(
            "Account URL not found. Please provide 'account_url' in custom_params or "
            "set AZURE_STORAGE_ACCOUNT_URL environment variable. If you are trying "
            "to connect to Azurite for local testing, you can alternatively provide "
            "a connection string via 'connection_string' in custom_params or "
            "AZURE_STORAGE_CONNECTION_STRING environment variable.");
    }

    blobContainerClient_ = std::make_unique<Azure::Storage::Blobs::BlobContainerClient>(
        blobServiceClient->GetBlobContainerClient(containerName));
}

void
azureBlobClient::setExecutor(std::shared_ptr<asio::thread_pool> executor) {
    throw std::runtime_error("azureBlobClient::setExecutor() not supported - Changing executor "
                             "after client creation is not supported");
}

void
azureBlobClient::putBlobAsync(std::string_view blob_name,
                              uintptr_t data_ptr,
                              size_t data_len,
                              size_t offset,
                              put_blob_callback_t callback) {
    // Azure Blob Storage doesn't support partial put operations with offset
    if (offset != 0) {
        callback(false);
        return;
    }

    std::string blob_name_str(blob_name);
    asio::post(*executor_, [this, blob_name_str, data_ptr, data_len, callback]() {
        try {
            auto blobClient = blobContainerClient_->GetBlockBlobClient(blob_name_str);
            blobClient.UploadFrom(reinterpret_cast<uint8_t *>(data_ptr), data_len);
            callback(true);
        }
        catch (const std::exception &e) {
            callback(false);
        }
    });
}

void
azureBlobClient::getBlobAsync(std::string_view blob_name,
                              uintptr_t data_ptr,
                              size_t data_len,
                              size_t offset,
                              get_blob_callback_t callback) {

    std::string blob_name_str(blob_name);
    asio::post(*executor_, [this, blob_name_str, data_ptr, data_len, offset, callback]() {
        try {
            auto blobClient = blobContainerClient_->GetBlockBlobClient(blob_name_str);
            Azure::Storage::Blobs::DownloadBlobToOptions options;
            Azure::Core::Http::HttpRange range;
            range.Offset = static_cast<int64_t>(offset);
            range.Length = static_cast<int64_t>(data_len);
            options.Range = range;
            blobClient.DownloadTo(reinterpret_cast<uint8_t *>(data_ptr), data_len, options);
            callback(true);
        }
        catch (const std::exception &e) {
            callback(false);
        }
    });
}

bool
azureBlobClient::checkBlobExists(std::string_view blob_name) {
    auto blobClient = blobContainerClient_->GetBlockBlobClient(std::string(blob_name));
    Azure::Storage::Blobs::GetBlobPropertiesOptions options;
    try {
        blobClient.GetProperties(options);
    }
    catch (const Azure::Core::RequestFailedException &e) {
        if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
            return false;
        } else {
            throw std::runtime_error("Failed to check if blob exists: " + std::string(e.what()));
        }
    }
    return true;
}
