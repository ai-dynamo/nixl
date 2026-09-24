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

#include "client.h"
#include "common/config_traits.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <asio/post.hpp>
#include <curl/curl.h>
#include <algorithm>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

size_t
captureBody(void *ptr, size_t size, size_t nmemb, void *userdata) {
    auto *body = static_cast<std::string *>(userdata);
    body->append(static_cast<char *>(ptr), size * nmemb);
    return size * nmemb;
}

std::once_flag curl_init_flag;

/// A customParams integer, or nullopt when it is not one: unlike std::stoul,
/// NIXL's converter rejects a sign, spaces and trailing text.
std::optional<std::size_t>
parseSize(const std::string &value) {
    try {
        return nixl::config::configTraits<std::size_t>::convert(value);
    }
    catch (const std::runtime_error &) {
        return std::nullopt;
    }
}

std::size_t
parseNumThreads(nixl_b_params_t *params) {
    if (!params || params->count("num_threads") == 0) {
        return std::max(2u, std::thread::hardware_concurrency() / 4);
    }
    // A zero pool would accept callbacks but never run them, hanging every
    // transfer; reject non-positive / malformed values instead.
    const std::string &value = params->at("num_threads");
    const std::optional<std::size_t> parsed = parseSize(value);
    if (!parsed || *parsed == 0) {
        throw std::invalid_argument("restClient: num_threads must be a positive integer");
    }
    return *parsed;
}

std::size_t
parseRequestTimeoutMs(nixl_b_params_t *params) {
    if (!params || params->count("request_timeout_ms") == 0) {
        return default_request_timeout_ms;
    }
    const std::string &value = params->at("request_timeout_ms");
    const std::optional<std::size_t> parsed = parseSize(value);
    if (!parsed || *parsed == 0) {
        throw std::invalid_argument("restClient: request_timeout_ms must be a positive integer");
    }
    return *parsed;
}

/// libcurl's description of the failure, or the generic CURLcode text when the
/// error buffer was left empty.
std::string
curlErrorText(CURLcode res, const char *error_buf) {
    if (error_buf && error_buf[0] != '\0') {
        return error_buf;
    }
    return curl_easy_strerror(res);
}

} // namespace

enum class rest_method { PUT, GET, HEAD };

// Per-request state for an in-flight curl_multi transfer. Owns its easy handle,
// header list, and response buffer for the full transfer lifetime; the user
// callback (exactly one of the two, by method) is moved out on completion.
struct restClient::requestCtx {
    CURL *easy = nullptr;
    struct curl_slist *headers = nullptr;
    std::string url;
    std::string responseBody;
    const char *opName = "";
    rest_method method = rest_method::GET;
    std::function<void(bool)> boolCb; // Put/Get
    std::function<void(std::optional<bool>)> checkCb; // Head
    /// libcurl's own description of the failure, which carries the reason as
    /// text even on paths where CURLINFO_OS_ERRNO stays unset.
    char errorBuf[CURL_ERROR_SIZE] = {};

    ~requestCtx() {
        if (headers) {
            curl_slist_free_all(headers);
        }
        if (easy) {
            // Must already be removed from the multi handle by the poller.
            curl_easy_cleanup(easy);
        }
    }
};

void
restClient::buildEasy(requestCtx *ctx) const {
    CURL *curl = ctx->easy;
    ctx->errorBuf[0] = '\0';
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, ctx->errorBuf);
    curl_easy_setopt(curl, CURLOPT_URL, ctx->url.c_str());
    switch (ctx->method) {
    case rest_method::PUT:
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
        curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)0);
        break;
    case rest_method::GET:
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        break;
    case rest_method::HEAD:
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
        break;
    }
    if (ctx->headers) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, ctx->headers);
    }
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, captureBody);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx->responseBody);
    curl_easy_setopt(curl, CURLOPT_PRIVATE, ctx);
    // No signals from libcurl: this process is multi-threaded.
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 1000L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(requestTimeoutMs_));
}

// The posted closure captures only the callback and the result, so the pool never
// touches curl state.
void
restClient::finishRequest(requestCtx *ctx, CURLcode res, long http_code) {
    if (ctx->method == rest_method::HEAD) {
        std::optional<bool> result;
        if (res != CURLE_OK) {
            NIXL_ERROR << absl::StrFormat("checkObjectExistsAsync: curl_code=%d (%s) for HEAD %s",
                                          static_cast<int>(res),
                                          curlErrorText(res, ctx->errorBuf),
                                          ctx->url);
            result = std::nullopt;
        } else if (http_code >= 200 && http_code < 300) {
            result = true;
        } else if (http_code == 404) {
            result = false;
        } else {
            NIXL_ERROR << absl::StrFormat(
                "checkObjectExistsAsync: HTTP %ld for HEAD %s", http_code, ctx->url);
            result = std::nullopt;
        }
        auto cb = std::move(ctx->checkCb);
        asio::post(pool_, [cb = std::move(cb), result]() {
            try {
                if (cb) {
                    cb(result);
                }
            }
            catch (...) {
            }
        });
    } else {
        const bool success = (res == CURLE_OK) && (http_code >= 200 && http_code < 300);
        if (!success) {
            NIXL_ERROR << absl::StrFormat(
                "%s: failed url=%s curl_code=%d (%s) http_code=%ld body=%s",
                ctx->opName,
                ctx->url,
                static_cast<int>(res),
                curlErrorText(res, ctx->errorBuf),
                http_code,
                ctx->responseBody.empty() ? "<empty>" : ctx->responseBody);
        } else {
            NIXL_DEBUG << absl::StrFormat(
                "%s: success url=%s http_code=%ld", ctx->opName, ctx->url, http_code);
        }
        auto cb = std::move(ctx->boolCb);
        asio::post(pool_, [cb = std::move(cb), success]() {
            try {
                if (cb) {
                    cb(success);
                }
            }
            catch (...) {
            }
        });
    }
    delete ctx;
}

restClient::restClient(nixl_b_params_t *custom_params)
    : numThreads_(parseNumThreads(custom_params)),
      pool_(numThreads_),
      requestTimeoutMs_(parseRequestTimeoutMs(custom_params)) {
    std::call_once(curl_init_flag, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
    if (!custom_params) {
        throw std::invalid_argument("restClient: custom_params is null");
    }

    auto ep_it = custom_params->find("endpoint_override");
    if (ep_it == custom_params->end() || ep_it->second.empty()) {
        throw std::invalid_argument("restClient: 'endpoint_override' parameter is required");
    }
    endpoint_ = ep_it->second;

    // Create the multi handle and start the poller only after validation, so a
    // throwing constructor leaves no thread or handle behind.
    multi_ = curl_multi_init();
    if (!multi_) {
        throw std::runtime_error("restClient: curl_multi_init failed");
    }
    poller_ = std::thread(&restClient::pollerLoop, this);

    NIXL_INFO << absl::StrFormat(
        "restClient initialized: endpoint=%s, callback_threads=%zu, request_timeout_ms=%zu "
        "(curl_multi poller)",
        endpoint_,
        numThreads_,
        requestTimeoutMs_);
}

restClient::~restClient() {
    stop_.store(true);
    if (multi_) {
        curl_multi_wakeup(multi_); // break the poller out of curl_multi_poll
    }
    if (poller_.joinable()) {
        poller_.join(); // poller fails any outstanding requests before returning
    }
    pool_.join(); // drain queued callbacks
    if (multi_) {
        curl_multi_cleanup(multi_);
    }
}

std::string
restClient::buildUrl(std::string_view key) const {
    return absl::StrFormat("%s/%s", endpoint_, key);
}

void
restClient::enqueue(std::unique_ptr<requestCtx> ctx) {
    {
        const std::lock_guard<std::mutex> lk(queueMtx_);
        incoming_.push(std::move(ctx));
    }
    curl_multi_wakeup(multi_); // thread-safe; nudges the poller to drain the queue
}

void
restClient::reapCompletions() {
    CURLMsg *msg = nullptr;
    int in_queue = 0;
    while ((msg = curl_multi_info_read(multi_, &in_queue)) != nullptr) {
        if (msg->msg != CURLMSG_DONE) {
            continue;
        }
        CURL *easy = msg->easy_handle;
        CURLcode res = msg->data.result;
        requestCtx *ctx = nullptr;
        curl_easy_getinfo(easy, CURLINFO_PRIVATE, &ctx);
        long http_code = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &http_code);

        curl_multi_remove_handle(multi_, easy);
        inflight_.erase(ctx);
        finishRequest(ctx, res, http_code);
    }
}

void
restClient::pollerLoop() {
    for (;;) {
        const bool stopping = stop_.load();

        // 1. Drain the producer queue into multi_. On shutdown, fail queued
        //    requests instead of starting them.
        std::queue<std::unique_ptr<requestCtx>> batch;
        {
            const std::lock_guard<std::mutex> lk(queueMtx_);
            std::swap(batch, incoming_);
        }
        while (!batch.empty()) {
            std::unique_ptr<requestCtx> ctx = std::move(batch.front());
            batch.pop();
            if (stopping) {
                finishRequest(ctx.release(), CURLE_ABORTED_BY_CALLBACK, 0);
                continue;
            }
            requestCtx *raw = ctx.get();
            CURLMcode mc = curl_multi_add_handle(multi_, raw->easy);
            if (mc != CURLM_OK) {
                NIXL_ERROR << absl::StrFormat(
                    "%s: curl_multi_add_handle failed: %s", raw->opName, curl_multi_strerror(mc));
                finishRequest(ctx.release(), CURLE_FAILED_INIT, 0);
                continue;
            }
            ctx.release(); // ownership tracked via CURLOPT_PRIVATE until completion
            inflight_.insert(raw);
        }

        // 2. Advance all in-flight transfers (non-blocking).
        int running = 0;
        curl_multi_perform(multi_, &running);

        // 3. Hand finished transfers' callbacks to the worker pool.
        reapCompletions();

        // 4. On shutdown, abort everything in flight so every callback fires
        //    exactly once.
        if (stopping) {
            for (requestCtx *ctx : inflight_) {
                curl_multi_remove_handle(multi_, ctx->easy);
                finishRequest(ctx, CURLE_ABORTED_BY_CALLBACK, 0);
            }
            inflight_.clear();
            break;
        }

        // 5. Block until socket activity, the 1s backstop, or curl_multi_wakeup().
        int numfds = 0;
        curl_multi_poll(multi_, nullptr, 0, 1000, &numfds);
    }
}

void
restClient::submitRdmaRequest(const char *op_name,
                              std::string_view key,
                              std::string_view rdma_desc,
                              bool is_upload,
                              std::function<void(bool)> callback,
                              size_t data_len,
                              size_t offset) {
    auto ctx = std::make_unique<requestCtx>();
    ctx->opName = op_name;
    ctx->method = is_upload ? rest_method::PUT : rest_method::GET;
    ctx->url = buildUrl(key);
    ctx->boolCb = std::move(callback);

    ctx->easy = curl_easy_init();
    if (!ctx->easy) {
        NIXL_ERROR << absl::StrFormat("%s: curl_easy_init failed", op_name);
        if (ctx->boolCb) {
            ctx->boolCb(false);
        }
        return;
    }

    std::string rdma_header = absl::StrFormat("x-scal-rdma: %s", rdma_desc);
    ctx->headers = curl_slist_append(ctx->headers, rdma_header.c_str());
    if (is_upload) {
        // Content-Length: 0; data is transferred via RDMA, not the HTTP body.
        ctx->headers = curl_slist_append(ctx->headers, "Content-Length: 0");
    } else if (data_len > 0) {
        // Always sent, offset 0 included: the caller registered exactly data_len
        // bytes, and an unranged GET would have the server send the whole object.
        std::string range_header =
            absl::StrFormat("Range: bytes=%zu-%zu", offset, offset + data_len - 1);
        ctx->headers = curl_slist_append(ctx->headers, range_header.c_str());
    }

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}

void
restClient::putObjectRdmaAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               std::string_view rdma_desc,
                               put_object_callback_t callback) {
    // The RDMA descriptor grants access to the buffer; log only its length.
    NIXL_DEBUG << absl::StrFormat(
        "putObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu, rdma_desc_len=%zu",
        key,
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset,
        rdma_desc.size());

    if (data_len == 0) {
        NIXL_ERROR << "putObjectRdmaAsync: data_len is 0, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    if (rdma_desc.empty()) {
        NIXL_ERROR << "putObjectRdmaAsync: rdma_desc is empty, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    submitRdmaRequest(
        "putObjectRdmaAsync", key, rdma_desc, /*is_upload=*/true, std::move(callback));
}

void
restClient::getObjectRdmaAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               std::string_view rdma_desc,
                               get_object_callback_t callback) {
    NIXL_DEBUG << absl::StrFormat(
        "getObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu, rdma_desc_len=%zu",
        key,
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset,
        rdma_desc.size());

    if (data_len == 0) {
        NIXL_ERROR << "getObjectRdmaAsync: data_len is 0, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    if (offset > (SIZE_MAX - (data_len - 1))) {
        NIXL_ERROR << "getObjectRdmaAsync: offset + data_len would overflow, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    if (rdma_desc.empty()) {
        NIXL_ERROR << "getObjectRdmaAsync: rdma_desc is empty, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    submitRdmaRequest("getObjectRdmaAsync",
                      key,
                      rdma_desc,
                      /*is_upload=*/false,
                      std::move(callback),
                      data_len,
                      offset);
}

void
restClient::checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) {
    auto ctx = std::make_unique<requestCtx>();
    ctx->opName = "checkObjectExistsAsync";
    ctx->method = rest_method::HEAD;
    ctx->url = buildUrl(key);
    ctx->checkCb = std::move(callback);

    ctx->easy = curl_easy_init();
    if (!ctx->easy) {
        NIXL_ERROR << "checkObjectExistsAsync: curl_easy_init failed";
        if (ctx->checkCb) {
            ctx->checkCb(std::nullopt);
        }
        return;
    }

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}
