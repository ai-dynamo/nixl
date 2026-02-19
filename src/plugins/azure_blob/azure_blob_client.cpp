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
#include "nixl_types.h"

namespace {

std::string
getAccountUrl(nixl_b_params_t *custom_params) {
    if (custom_params) {
        auto account_it = custom_params->find("account_url");
        if (account_it != custom_params->end() && !account_it->second.empty()) {
            return account_it->second;
        }
    }
    const char *env_account = std::getenv("AZURE_STORAGE_ACCOUNT_URL");
    if (env_account && env_account[0] != '\0') return std::string(env_account);
    throw std::runtime_error(
        "Account URL not found. Please provide 'account_url' in custom_params or "
        "set AZURE_STORAGE_ACCOUNT_URL environment variable");
}

std::string
getContainerName(nixl_b_params_t *custom_params) {
    if (custom_params) {
        auto container_it = custom_params->find("container_name");
        if (container_it != custom_params->end() && !container_it->second.empty()) {
            return container_it->second;
        }
    }

    const char *env_container = std::getenv("AZURE_STORAGE_CONTAINER_NAME");
    if (env_container && env_container[0] != '\0') return std::string(env_container);
    throw std::runtime_error(
        "Container name not found. Please provide 'container_name' in custom_params or "
        "set AZURE_STORAGE_CONTAINER_NAME environment variable");
}

std::string
getCaBundle(nixl_b_params_t *custom_params) {
    if (custom_params) {
        auto ca_bundle_it = custom_params->find("ca_bundle");
        if (ca_bundle_it != custom_params->end() && !ca_bundle_it->second.empty()) {
            return ca_bundle_it->second;
        }
    }
    const char *env_ca_bundle = std::getenv("AZURE_CA_BUNDLE");
    if (env_ca_bundle && env_ca_bundle[0] != '\0') return std::string(env_ca_bundle);
    return ""; // Return empty string if not provided, which means use default CA bundle
}

} // namespace

azureBlobClient::azureBlobClient(nixl_b_params_t *custom_params,
                                 std::shared_ptr<asio::thread_pool> executor) {
    executor_ = executor;
    std::string accountUrl = ::getAccountUrl(custom_params);
    std::string containerName = ::getContainerName(custom_params);
    Azure::Storage::Blobs::BlobClientOptions options;
    options.Telemetry.ApplicationId = "azpartner-nixl/0.1.0";

    std::string caBundle = ::getCaBundle(custom_params);
    if (!caBundle.empty()) {
        Azure::Core::Http::CurlTransportOptions curlOptions;
        curlOptions.CAInfo = caBundle;
        options.Transport.Transport =
            std::make_shared<Azure::Core::Http::CurlTransport>(curlOptions);
    }

    auto blobServiceClient = std::make_unique<Azure::Storage::Blobs::BlobServiceClient>(
        accountUrl, std::make_shared<Azure::Identity::DefaultAzureCredential>(), options);
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
