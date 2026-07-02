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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_REST_CLIENT_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_REST_CLIENT_H

#include <asio/thread_pool.hpp>
#include <asio/post.hpp>
#include <curl/curl.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include "obj_backend.h"
#include "nixl_types.h"

/**
 * Interface for Scality AI Connector clients that support RDMA operations.
 */
class iRestClient {
public:
    /**
     * Asynchronously put an object using RDMA.
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
     * Asynchronously get an object using RDMA.
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

    /**
     * Asynchronously check whether an object exists (HTTP HEAD).
     * @param key The object key
     * @param callback Receives true (exists), false (404), or std::nullopt (error)
     */
    virtual void
    checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) = 0;

protected:
    // Implementations are only ever owned through shared_ptr (never deleted via
    // an iRestClient*), so a protected non-virtual destructor suffices and
    // avoids the vtable cost of a public virtual one.
    ~iRestClient() = default;
};

/**
 * Scality AI Connector HTTP client with RDMA support.
 * Uses libcurl to perform HTTP PUT/GET requests, passing the RDMA
 * descriptor via the x-scal-rdma custom header.
 *
 * URL format: {endpoint}/{key}
 */
class RestClient : public iRestClient {
public:
    /**
     * Constructor.
     * @param custom_params Backend init params; must contain "endpoint_override".
     *                      Optional "num_threads" sizes the callback worker pool
     *                      (default: max(2, hardware_concurrency / 4)).
     */
    explicit RestClient(nixl_b_params_t *custom_params);

    ~RestClient();

    void
    putObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       put_object_callback_t callback) override;

    void
    getObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       get_object_callback_t callback) override;

    void
    checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) override;

private:
    /// Per-request state for an in-flight curl_multi transfer. Defined in client.cpp.
    struct RequestCtx;

    /// Base endpoint, e.g. "http://10.0.0.1:81"
    std::string endpoint_;

    // A single poller thread owns the curl multi handle and drives all transfers;
    // completed callbacks are offloaded to this pool (sized by numThreads_) so a
    // slow callback can't stall the event loop. curl multi handles are not
    // thread-safe: every curl_multi_* call runs on the poller thread, except
    // curl_multi_wakeup() which producers use to nudge it.
    std::size_t numThreads_;
    asio::thread_pool pool_; // callback worker pool

    CURLM *multi_ = nullptr;
    std::thread poller_;
    std::mutex queueMtx_;
    std::queue<std::unique_ptr<RequestCtx>> incoming_;
    std::atomic<bool> stop_{false};
    /// Handles currently added to multi_. Poller-thread access only (no lock).
    std::unordered_set<RequestCtx *> inflight_;

    /**
     * Build the full URL for a given key.
     * @param key Object key
     * @return Full URL string
     */
    [[nodiscard]] std::string
    buildUrl(std::string_view key) const;

    /**
     * Build a PUT/GET RDMA request context and hand it to the poller. PUT and GET
     * differ only in the curl method options, so they share this path.
     * @param op_name Operation label for logging (string literal)
     * @param key Object key
     * @param rdma_desc RDMA descriptor for the x-scal-rdma header
     * @param is_upload true for PUT (upload), false for GET
     * @param callback Result callback (invoked on the worker pool)
     */
    void
    submitRdmaRequest(const char *op_name,
                      std::string_view key,
                      std::string_view rdma_desc,
                      bool is_upload,
                      std::function<void(bool)> callback);

    /// Apply URL + method-specific curl options to a request's easy handle.
    static void
    buildEasy(RequestCtx *ctx);

    /// Push a fully-built request onto the queue and wake the poller.
    void
    enqueue(std::unique_ptr<RequestCtx> ctx);

    /// Body of the poller thread: drain queue, perform, reap, poll.
    void
    pollerLoop();

    /// Reap finished transfers (poller thread only) and dispatch their callbacks.
    void
    reapCompletions();

    /// Map a finished transfer to its callback dispatcher and free the context.
    /// Poller-thread only.
    void
    finishRequest(RequestCtx *ctx, CURLcode res, long http_code);

    /// Dispatch a finished HEAD transfer's callback on the worker pool.
    void
    dispatchHeadResult(RequestCtx *ctx, CURLcode res, long http_code);

    /// Dispatch a finished PUT/GET transfer's callback on the worker pool.
    void
    dispatchXferResult(RequestCtx *ctx, CURLcode res, long http_code);
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CLIENT_H
