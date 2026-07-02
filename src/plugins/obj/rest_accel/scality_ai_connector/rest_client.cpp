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

#include "rest_client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <asio/post.hpp>
#include <curl/curl.h>
#include <algorithm>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace {

size_t
captureBody(void *ptr, size_t size, size_t nmemb, void *userdata) {
    auto *body = static_cast<std::string *>(userdata);
    body->append(static_cast<char *>(ptr), size * nmemb);
    return size * nmemb;
}

std::once_flag curl_init_flag;

std::size_t
parseNumThreads(nixl_b_params_t *params) {
    if (!params || params->count("num_threads") == 0) {
        return std::max(2u, std::thread::hardware_concurrency() / 4);
    }
    // A zero pool would accept callbacks but never run them, hanging every
    // transfer; reject non-positive / malformed values instead.
    const std::string &value = params->at("num_threads");
    std::size_t consumed = 0;
    const std::size_t parsed = std::stoul(value, &consumed);
    if (consumed != value.size() || parsed == 0) {
        throw std::invalid_argument("RestClient: num_threads must be a positive integer");
    }
    return parsed;
}

} // namespace

enum class restMethod { PUT, GET, HEAD };

// Per-request state for an in-flight curl_multi transfer. Owns its easy handle,
// header list, and response buffer for the full transfer lifetime; the user
// callback (exactly one of the two, by method) is moved out on completion.
struct RestClient::RequestCtx {
    CURL *easy = nullptr;
    struct curl_slist *headers = nullptr;
    std::string url;
    std::string response_body;
    const char *op_name = "";
    restMethod method = restMethod::GET;
    std::function<void(bool)> bool_cb; // Put/Get
    std::function<void(std::optional<bool>)> check_cb; // Head

    ~RequestCtx() {
        if (headers) {
            curl_slist_free_all(headers);
        }
        if (easy) {
            // Must already be removed from the multi handle by the poller.
            curl_easy_cleanup(easy);
        }
    }
};

// Apply URL + method-specific options to a fresh easy handle. Wire format is
// kept byte-identical to the previous synchronous implementation.
void
RestClient::buildEasy(RequestCtx *ctx) {
    CURL *curl = ctx->easy;
    curl_easy_setopt(curl, CURLOPT_URL, ctx->url.c_str());
    switch (ctx->method) {
    case restMethod::PUT:
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
        curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)0);
        break;
    case restMethod::GET:
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        break;
    case restMethod::HEAD:
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L); // HTTP HEAD
        break;
    }
    if (ctx->headers) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, ctx->headers);
    }
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, captureBody);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx->response_body);
    curl_easy_setopt(curl, CURLOPT_PRIVATE, ctx);
    // Bound the control-plane request so a stalled endpoint can't leave a
    // transfer in-flight forever (the body is empty; bulk data is on RDMA).
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 5000L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 30000L);
}

// Dispatch a finished HEAD transfer's callback on the worker pool. Runs on the
// poller thread; the posted closure captures only the moved callback plus the
// plain result, so the pool never touches curl state. An empty callback is
// skipped rather than posted, and a throwing callback is contained so it cannot
// take down a pool worker.
void
RestClient::dispatchHeadResult(RequestCtx *ctx, CURLcode res, long http_code) {
    std::optional<bool> result;
    if (res != CURLE_OK) {
        NIXL_ERROR << absl::StrFormat(
            "checkObjectExistsAsync: curl_code=%d for HEAD %s", static_cast<int>(res), ctx->url);
    } else if (http_code >= 200 && http_code < 300) {
        result = true;
    } else if (http_code == 404) {
        result = false;
    } else {
        NIXL_ERROR << absl::StrFormat(
            "checkObjectExistsAsync: HTTP %ld for HEAD %s", http_code, ctx->url);
    }
    if (ctx->check_cb) {
        asio::post(pool_, [cb = std::move(ctx->check_cb), result]() {
            try {
                cb(result);
            }
            catch (...) {
                NIXL_WARN << "checkObjectExistsAsync: callback threw an exception";
            }
        });
    }
}

// Dispatch a finished PUT/GET transfer's callback on the worker pool. Same
// threading and safety contract as dispatchHeadResult. op_name is a string
// literal (static storage), so the lambda captures the pointer by value and it
// stays valid after ctx is freed.
void
RestClient::dispatchXferResult(RequestCtx *ctx, CURLcode res, long http_code) {
    const bool success = (res == CURLE_OK) && (http_code >= 200 && http_code < 300);
    if (!success) {
        NIXL_ERROR << absl::StrFormat("%s: failed url=%s curl_code=%d http_code=%ld body=%s",
                                      ctx->op_name,
                                      ctx->url,
                                      static_cast<int>(res),
                                      http_code,
                                      ctx->response_body.empty() ? "<empty>" : ctx->response_body);
    } else {
        NIXL_DEBUG << absl::StrFormat(
            "%s: success url=%s http_code=%ld", ctx->op_name, ctx->url, http_code);
    }
    if (ctx->bool_cb) {
        asio::post(pool_, [cb = std::move(ctx->bool_cb), success, op_name = ctx->op_name]() {
            try {
                cb(success);
            }
            catch (...) {
                NIXL_WARN << absl::StrFormat("%s: callback threw an exception", op_name);
            }
        });
    }
}

// Map a finished transfer to its callback dispatcher and free the context.
// Poller-thread only.
void
RestClient::finishRequest(RequestCtx *ctx, CURLcode res, long http_code) {
    if (ctx->method == restMethod::HEAD) {
        dispatchHeadResult(ctx, res, http_code);
    } else {
        dispatchXferResult(ctx, res, http_code);
    }
    delete ctx;
}

RestClient::RestClient(nixl_b_params_t *custom_params)
    : numThreads_(parseNumThreads(custom_params)),
      pool_(numThreads_) {
    std::call_once(curl_init_flag, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
    if (!custom_params) {
        throw std::invalid_argument("RestClient: custom_params is null");
    }

    auto ep_it = custom_params->find("endpoint_override");
    if (ep_it == custom_params->end() || ep_it->second.empty()) {
        throw std::invalid_argument("RestClient: 'endpoint_override' parameter is required");
    }
    endpoint_ = ep_it->second;

    // Create the multi handle and start the poller only after validation, so a
    // throwing constructor leaves no thread or handle behind.
    multi_ = curl_multi_init();
    if (!multi_) {
        throw std::runtime_error("RestClient: curl_multi_init failed");
    }
    poller_ = std::thread(&RestClient::pollerLoop, this);

    NIXL_INFO << absl::StrFormat(
        "RestClient initialized: endpoint=%s, callback_threads=%zu (curl_multi poller)",
        endpoint_,
        numThreads_);
}

RestClient::~RestClient() {
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
RestClient::buildUrl(std::string_view key) const {
    return absl::StrFormat("%s/%s", endpoint_, key);
}

void
RestClient::enqueue(std::unique_ptr<RequestCtx> ctx) {
    {
        const std::lock_guard<std::mutex> lk(queueMtx_);
        incoming_.push(std::move(ctx));
    }
    curl_multi_wakeup(multi_); // thread-safe; nudges the poller to drain the queue
}

void
RestClient::reapCompletions() {
    CURLMsg *msg = nullptr;
    int in_queue = 0;
    while ((msg = curl_multi_info_read(multi_, &in_queue)) != nullptr) {
        if (msg->msg != CURLMSG_DONE) {
            continue;
        }
        CURL *easy = msg->easy_handle;
        CURLcode res = msg->data.result;
        RequestCtx *ctx = nullptr;
        curl_easy_getinfo(easy, CURLINFO_PRIVATE, &ctx);
        long http_code = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &http_code);

        curl_multi_remove_handle(multi_, easy);
        inflight_.erase(ctx);
        finishRequest(ctx, res, http_code);
    }
}

void
RestClient::pollerLoop() {
    for (;;) {
        const bool stopping = stop_.load();

        // 1. Drain the producer queue. On shutdown, fail queued requests instead
        //    of starting them.
        std::queue<std::unique_ptr<RequestCtx>> batch;
        {
            const std::lock_guard<std::mutex> lk(queueMtx_);
            std::swap(batch, incoming_);
        }
        while (!batch.empty()) {
            std::unique_ptr<RequestCtx> ctx = std::move(batch.front());
            batch.pop();
            if (stopping) {
                finishRequest(ctx.release(), CURLE_ABORTED_BY_CALLBACK, 0);
                continue;
            }
            RequestCtx *raw = ctx.get();
            CURLMcode mc = curl_multi_add_handle(multi_, raw->easy);
            if (mc != CURLM_OK) {
                NIXL_ERROR << absl::StrFormat(
                    "%s: curl_multi_add_handle failed: %s", raw->op_name, curl_multi_strerror(mc));
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

        // 4. On shutdown, abort anything still in flight so every callback fires
        //    and every request is freed (finishRequest deletes the context).
        if (stopping) {
            for (RequestCtx *ctx : inflight_) {
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
RestClient::submitRdmaRequest(const char *op_name,
                              std::string_view key,
                              std::string_view rdma_desc,
                              bool is_upload,
                              std::function<void(bool)> callback) {
    auto ctx = std::make_unique<RequestCtx>();
    ctx->op_name = op_name;
    ctx->method = is_upload ? restMethod::PUT : restMethod::GET;
    ctx->url = buildUrl(key);
    ctx->bool_cb = std::move(callback);

    ctx->easy = curl_easy_init();
    if (!ctx->easy) {
        NIXL_ERROR << absl::StrFormat("%s: curl_easy_init failed", op_name);
        if (ctx->bool_cb) {
            ctx->bool_cb(false);
        }
        return;
    }

    const std::string rdma_header = absl::StrFormat("x-scal-rdma: %s", rdma_desc);
    ctx->headers = curl_slist_append(ctx->headers, rdma_header.c_str());
    if (is_upload) {
        // Content-Length: 0; data is transferred via RDMA, not the HTTP body.
        ctx->headers = curl_slist_append(ctx->headers, "Content-Length: 0");
    }

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}

void
RestClient::putObjectRdmaAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               std::string_view rdma_desc,
                               put_object_callback_t callback) {
    // The RDMA descriptor is sensitive transfer-capability metadata; log only
    // its length, never the raw token.
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
RestClient::getObjectRdmaAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               std::string_view rdma_desc,
                               get_object_callback_t callback) {
    // Log only the descriptor length, never the raw RDMA token (sensitive).
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

    if ((data_len > 0) && (offset > (SIZE_MAX - (data_len - 1)))) {
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

    submitRdmaRequest(
        "getObjectRdmaAsync", key, rdma_desc, /*is_upload=*/false, std::move(callback));
}

void
RestClient::checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) {
    auto ctx = std::make_unique<RequestCtx>();
    ctx->op_name = "checkObjectExistsAsync";
    ctx->method = restMethod::HEAD;
    ctx->url = buildUrl(key);
    ctx->check_cb = std::move(callback);

    ctx->easy = curl_easy_init();
    if (!ctx->easy) {
        NIXL_ERROR << "checkObjectExistsAsync: curl_easy_init failed";
        if (ctx->check_cb) {
            ctx->check_cb(std::nullopt);
        }
        return;
    }

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}
