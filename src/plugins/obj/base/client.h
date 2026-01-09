/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OBJ_PLUGIN_BASE_CLIENT_H
#define OBJ_PLUGIN_BASE_CLIENT_H

#include <functional>
#include <memory>
#include <string_view>
#include <cstdint>
#include <aws/core/utils/Threading/Executor.h>

using put_object_callback_t = std::function<void(bool success)>;
using get_object_callback_t = std::function<void(bool success)>;

/**
 * Base abstract interface for object storage client operations.
 * This is the top-level interface that all object storage clients implement.
 */
class objectClient {
public:
    virtual ~objectClient() = default;

    /**
     * Set the executor for async operations.
     * @param executor The executor to use for async operations
     */
    virtual void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) = 0;

    /**
     * Asynchronously put an object to storage.
     * @param key The object key
     * @param data_ptr Pointer to the data to upload
     * @param data_len Length of the data in bytes
     * @param offset Offset within the object
     * @param callback Callback function to handle the result
     */
    virtual void
    putObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   put_object_callback_t callback) = 0;

    /**
     * Asynchronously get an object from storage.
     * @param key The object key
     * @param data_ptr Pointer to the buffer to store the downloaded data
     * @param data_len Maximum length of data to read
     * @param offset Offset within the object to start reading from
     * @param callback Callback function to handle the result
     */
    virtual void
    getObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) = 0;

    /**
     * Check if the object exists.
     * @param key The object key
     * @return true if the object exists, false otherwise
     */
    virtual bool
    checkObjectExists(std::string_view key) = 0;
};

#endif // OBJ_PLUGIN_BASE_CLIENT_H
