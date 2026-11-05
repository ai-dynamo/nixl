/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CLIENT_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CLIENT_H

#include <asio/thread_pool.hpp>
#include <asio/post.hpp>
#include <curl/curl.h>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>
#include "obj_backend.h"
#include "nixl_types.h"

/**
 * Control-plane client of the Scality AI Connector: header-only HTTP requests
 * that name the object and carry the RDMA descriptor. Also the test injection
 * seam.
 */
class iRestClient {
public:
    virtual ~iRestClient() = default;

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
     * @param past_end_ok A 416 (range past the end of the object) counts as
     *                    success: nothing was there to read
     * @param callback Callback function to handle the result
     */
    virtual void
    getObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       bool past_end_ok,
                       get_object_callback_t callback) = 0;

    /**
     * Asynchronously get a byte range into host memory over plain HTTP.
     *
     * No registration and no RDMA descriptor: the response body is written
     * straight into the caller's buffer. Cheaper than RDMA for small reads, whose
     * cost is dominated by pinning. Intended for metadata, not bulk data.
     *
     * A response shorter than data_len succeeds: a range reaching past the end of
     * the object is answered as a complete 206 with fewer bytes. A body cut short
     * against its own Content-Length fails, as does one exceeding data_len.
     *
     * @param key The object key
     * @param dst Host buffer receiving the bytes; must hold data_len
     * @param data_len Bytes requested, and the capacity of dst
     * @param offset Offset within the object to start reading from
     * @param callback Receives true if the range was delivered
     */
    virtual void
    getObjectBodyAsync(std::string_view key,
                       void *dst,
                       size_t data_len,
                       size_t offset,
                       get_object_callback_t callback) = 0;

    /**
     * Asynchronously check whether an object exists (HTTP HEAD).
     * @param key The object key
     * @param callback Receives true (exists), false (404), or std::nullopt (error)
     */
    virtual void
    checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) = 0;
};

/// Default limit on a whole request, connection included.
constexpr std::size_t default_request_timeout_ms = 2000;

/// Upper bound on the default cap for concurrently-running requests; the default
/// is also clamped to a share of RLIMIT_NOFILE (see defaultMaxInflight).
constexpr std::size_t default_max_inflight = 512;

/**
 * libcurl implementation of iRestClient. PUT and GET carry the RDMA descriptor
 * in the x-scal-rdma header and an empty body: the server moves the bytes by
 * RDMA.
 *
 * URL format: {endpoint}/{key}
 */
class restClient : public iRestClient {
public:
    /**
     * @param custom_params Backend init params; must contain "endpoint_override".
     *                      Optional "num_threads" sizes the callback worker pool
     *                      (default: max(2, hardware_concurrency / 4)).
     *                      Optional "request_timeout_ms" bounds each request
     *                      (default default_request_timeout_ms).
     *                      Optional "max_inflight" caps concurrently-running
     *                      requests ("0" = unlimited, default
     *                      min(default_max_inflight, RLIMIT_NOFILE / 4)).
     */
    explicit restClient(nixl_b_params_t *custom_params);

    ~restClient() override;

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
                       bool past_end_ok,
                       get_object_callback_t callback) override;

    void
    getObjectBodyAsync(std::string_view key,
                       void *dst,
                       size_t data_len,
                       size_t offset,
                       get_object_callback_t callback) override;

    void
    checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) override;

private:
    /// Per-request state for an in-flight curl_multi transfer. Defined in client.cpp.
    struct requestCtx;

    /// Base endpoint, e.g. "http://10.0.0.1:10000"
    std::string endpoint_;

    // A single poller thread owns the curl multi handle and drives all transfers;
    // completed callbacks are offloaded to this pool (sized by numThreads_) so a
    // slow callback can't stall the event loop. curl multi handles are not
    // thread-safe: every curl_multi_* call runs on the poller thread, except
    // curl_multi_wakeup() which producers use to nudge it.
    std::size_t numThreads_;
    asio::thread_pool pool_;

    /// Limit on each whole request. A stalled request fails after this instead of
    /// holding its transfer: for a cache, the caller's fallback is faster.
    std::size_t requestTimeoutMs_;

    // Cap on requests running at once, since each holds a connection and a file
    // descriptor. Excess requests wait in pending_. 0 disables the cap.
    std::size_t maxInflight_;

    // Idle easy handles kept for reuse: curl_easy_reset() keeps the connection
    // alive, where curl_easy_cleanup() would close it. At most easyCacheCap_.
    std::mutex easyMtx_;
    std::vector<CURL *> easyCache_;
    std::size_t easyCacheCap_;

    CURLM *multi_ = nullptr;
    std::thread poller_;
    std::mutex queueMtx_;
    std::queue<std::unique_ptr<requestCtx>> incoming_;
    std::atomic<bool> stop_{false};
    /// Handles currently added to multi_. Poller-thread access only (no lock).
    std::unordered_set<requestCtx *> inflight_;
    /// Built requests waiting for a free in-flight slot. Poller-thread only.
    std::deque<std::unique_ptr<requestCtx>> pending_;
    // High-water marks, logged at teardown. Poller-thread only.
    std::size_t peakInflight_ = 0;
    std::size_t peakPending_ = 0;
    // Connection-reuse accounting, logged at teardown: new_connections well
    // below requests means the handle cache is doing its job. Poller-thread only.
    std::size_t totalRequests_ = 0;
    std::size_t newConnects_ = 0;
    /// The first-connect-failure diagnostic was emitted. Poller-thread only.
    bool fdExhaustionLogged_ = false;
    /// The process is out of file descriptors: set when a connect failure finds
    /// none to spare, cleared when a request gets a response or a later check
    /// finds descriptors again. While set, the backlog is failed at once.
    /// Poller-thread only.
    bool fdExhausted_ = false;
    /// Descriptors ran out at some point, for the teardown report. Poller-thread
    /// only.
    bool fdExhaustionSeen_ = false;
    /// Requests failed by the backlog drain, reported once at teardown rather
    /// than one log line each. Poller-thread only.
    std::size_t fdAbandoned_ = 0;

    std::string
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
                      std::function<void(bool)> callback,
                      size_t data_len = 0,
                      size_t offset = 0,
                      bool past_end_ok = false);

    /// Apply URL + method-specific curl options to a request's easy handle.
    void
    buildEasy(requestCtx *ctx) const;

    /// Take an idle handle from easyCache_, or create one. Any thread.
    CURL *
    acquireEasy();

    /// Reset a finished handle and return it to easyCache_, cleaning it up if the
    /// cache is full. The handle must already be removed from multi_. Any thread.
    void
    releaseEasy(CURL *easy);

    /// Push a fully-built request onto the queue and wake the poller.
    void
    enqueue(std::unique_ptr<requestCtx> ctx);

    /// Move pending requests into multi_ while in-flight slots are available.
    /// Poller-thread only.
    void
    startPending();

    /// Body of the poller thread: drain queue, perform, reap, poll.
    void
    pollerLoop();

    /// Reap finished transfers (poller thread only) and dispatch their callbacks.
    void
    reapCompletions();

    /// Map a finished transfer to its callback, dispatch it on the worker pool,
    /// and free the context. Poller-thread only.
    void
    finishRequest(requestCtx *ctx, CURLcode res, long http_code);
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CLIENT_H
