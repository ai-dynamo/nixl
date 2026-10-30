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
#include "nixl_tcpstore_client_async.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <span>

#include "asio/error_code.hpp"
#include "byte_buffer.h"
#include "common/nixl_log.h"

// Implementation notes on threading model:
//
// Assumptions:
// - The given I/O context is driven by a single thread. That is, its run()
//   function is called from a single thread. Therefore, all functions
//   scheduled on the I/O context are executed sequentially.
//
// Facts:
// - The socket object is not thread-safe.
// - Communications with the TCPStore server are message based.
//
// Consequences:
// - We must NOT write multiple requests concurrently to the same socket.
//   That is we must NOT start writing a request until after the previous
//   request has been sent completely, even in case of short writes.
// - We must NOT read multiple responses concurrently from the same socket.
//   That is we must NOT start reading a response until after the previous
//   response has been read completely, even in case of short reads.
// - However, we can issue a read and a write concurrently.
//
// Design:
// - Accesses to the socket object must be scheduled on the I/O context. This
//   ensures that the socket object is not accessed concurrently, and thus
//   avoids the need to synchronize explicitly.
// - If any message cannot be written or read in its entirety, we must close
//   the socket, fail all pending requests and reconnect. Request-level
//   retry is handled by the upper layer.
// - Each TCPStore request is executed in two phases: a send phase and a
//   receive phase.
// - The send and receive phases are each serialized on its own queue. Each
//   phase uses a hasInflight flag to block the next request from being initiated
//   until the previous request has been completed.
// - The send queue is accessed from both the caller thread and the I/O thread,
//   and thus must be protected by a mutex. The reason for this is two fold:
//   - The I/O thread may have exited, so the caller must determine if the
//     request can be queued in its own thread.
//   - Inserting the request must be synchronized with the SHUTDOWN sequence.
//     We must make sure that if the request is queued, the SHUTDOWN sequence
//     must see the request.
// - On the other hand, the receive queue is accessed exclusively from the I/O
//   thread, and thus does not need to be protected by a mutex.
// - The send queue and the receive queue can be processed concurrently.

constexpr std::int32_t max_connection_retry_count = 10;
constexpr std::chrono::milliseconds connection_retry_interval = std::chrono::seconds(1);

namespace tcpstore {
// c10d TCPStore wire constants (torch 2.x). Keys carry the same "/" prefix the
// reference client prepends so we share the keyspace with torch clients.
// Integers go on the wire in native byte order (matching c10d/torch), so this
// assumes the store server runs on the same endianness as the client.
constexpr std::uint32_t validation_magic = 0x3C85F7CE;
constexpr char key_prefix[] = "/";

// Upper bound on a single value read from the store; metadata blobs are far
// smaller, so anything larger means a corrupt or desynced response.
constexpr std::uint64_t max_blob_bytes = 1ULL << 30; // 1 GiB

// Subset of c10d::detail::QueryType we use; values are the enum ordinals.
enum class query_type_t : std::uint8_t {
    VALIDATE = 0,
    SET = 1,
    GET = 3,
    CHECK = 5,
    DELETE_KEY = 8,
    PING = 13,
};

// c10d::detail::CheckResponseType.
enum class check_response_t : std::uint8_t { READY = 0, NOT_READY = 1 };
} // namespace tcpstore

// Base class for all TCPStore requests.
class nixlTcpStoreClientAsync::Request {
public:
    Request(nixlTcpStoreClientAsync &client, deadline_t deadline)
        : client_(client),
          deadline_(deadline) {}

    virtual ~Request() = default;

    virtual std::string
    name() const = 0;

    virtual nixl::dynamicSizedBuffer
    requestMessage() = 0;

    // Return std::dynamic_extent if the response is variable-length.
    virtual std::size_t
    responseMessageSize() const = 0;

    virtual void
    onComplete(asio::error_code error, nixl::fixedCapacityBuffer response) = 0;


    nixlTcpStoreClientAsync &client_;
    deadline_t deadline_;
    std::int32_t connectionGeneration_ = -1;
};

template<typename Fn, typename... Args>
auto
nixlTcpStoreClientAsync::cancelAndErrorAdapter(std::string what,
                                               deadline_t deadline,
                                               Fn &&on_complete,
                                               void (Fn::*)(asio::error_code, Args...)) {
    return asio::cancel_at(
        deadline,
        asio::cancellation_type::terminal,
        [this,
         what = std::move(what),
         deadline,
         on_complete = std::move(on_complete),
         generation = generation_](asio::error_code error, Args... args) mutable {
            if (error) {
                onError(std::move(what),
                        error,
                        std::chrono::steady_clock::now() >= deadline,
                        generation);
            }
            on_complete(error, std::forward<Args>(args)...);
        });
}

template<typename Buffer, typename Fn>
    requires std::is_same_v<Buffer, nixl::fixedCapacityBuffer> ||
    std::is_same_v<Buffer, asio::mutable_buffer>
void
nixlTcpStoreClientAsync::initiateAsyncRead(Buffer buf,
                                           std::string what,
                                           deadline_t deadline,
                                           Fn &&on_complete) {
    asio::mutable_buffer b;
    if constexpr (std::is_same_v<Buffer, nixl::fixedCapacityBuffer>) {
        b = asio::buffer(buf.data(), buf.capacity());
    } else if constexpr (std::is_same_v<Buffer, asio::mutable_buffer>) {
        b = buf;
    }

    auto on_complete_with_buf = [buf = std::move(buf), on_complete = std::move(on_complete)](
                                    asio::error_code error, std::size_t bytes_transferred) mutable {
        on_complete(std::move(buf), error, bytes_transferred);
    };

    asio::async_read(
        socket_, b, cancelAndErrorAdapter(what, deadline, std::move(on_complete_with_buf)));
}

template<typename Fn>
void
nixlTcpStoreClientAsync::initiateAsyncWrite(nixl::dynamicSizedBuffer buf,
                                            std::string what,
                                            deadline_t deadline,
                                            Fn &&on_complete) {
    asio::const_buffer b = asio::buffer(buf.data(), buf.size());

    auto on_complete_with_buf = [buf = std::move(buf), on_complete = std::move(on_complete)](
                                    asio::error_code error, std::size_t bytes_transferred) mutable {
        on_complete(std::move(buf), error, bytes_transferred);
    };

    asio::async_write(
        socket_, b, cancelAndErrorAdapter(what, deadline, std::move(on_complete_with_buf)));
}

nixlTcpStoreClientAsync::nixlTcpStoreClientAsync(std::string host,
                                                 std::uint16_t port,
                                                 std::chrono::milliseconds connect_timeout,
                                                 std::chrono::milliseconds op_timeout,
                                                 asio::io_context &io_context) noexcept
    : host_(host),
      port_(port),
      portStr_(std::to_string(port)),
      connectTimeout_(connect_timeout),
      opTimeout_(op_timeout),
      ioContext_(io_context),
      socket_(ioContext_),
      connectionRetryTimer_(ioContext_, std::chrono::steady_clock::now()),
      shutdownTimer_(ioContext_) {}

nixlTcpStoreClientAsync::~nixlTcpStoreClientAsync() = default;

void
nixlTcpStoreClientAsync::connect() {
    asio::post(ioContext_, [this]() {
        // Reset the connection retry timer. This call also cancels any pending
        // asynchronous wait operation.
        connectionRetryTimer_.expires_at(std::chrono::steady_clock::now());

        doResolve(connectTimeout_);
    });
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::doResolve(std::chrono::milliseconds timeout) {
    if (connectionState_ != connection_state_t::CLOSED) {
        NIXL_INFO << "[TCPStore client] Client is already connected or shutdown.";
        return;
    }
    if (connectionRetryTimer_.expiry() > std::chrono::steady_clock::now()) {
        NIXL_DEBUG << "[TCPStore client] Connection is broken. Waiting for reconnection.";
        return;
    }

    NIXL_INFO << "[TCPStore client] connecting to " << host_ << ":" << portStr_;
    connectionState_ = connection_state_t::CONNECTING;
    generation_++;
    connectionRetryCount_++;

    const auto deadline = std::chrono::steady_clock::now() + timeout;

    if (resolvedEndpoints_.empty()) {
        auto resolver = std::make_unique<asio::ip::tcp::resolver>(ioContext_);
        resolver->async_resolve(
            host_,
            portStr_,
            cancelAndErrorAdapter(
                "resolve",
                deadline,
                [this, deadline, resolver = std::move(resolver)](
                    asio::error_code error, asio::ip::tcp::resolver::results_type results) mutable {
                    if (!error && connectionState_ == connection_state_t::CONNECTING) {
                        resolvedEndpoints_ = results;
                        doConnect(deadline);
                    }
                }));
    } else {
        doConnect(deadline);
    }
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::doConnect(deadline_t deadline) {
    asio::async_connect(
        socket_,
        resolvedEndpoints_,
        cancelAndErrorAdapter(
            "connect",
            deadline,
            [this, deadline](asio::error_code error, const asio::ip::tcp::endpoint &) mutable {
                if (!error && connectionState_ == connection_state_t::CONNECTING) {
                    // Attempt to set keep-alive and ignore the error.
                    asio::error_code ec;
                    socket_.set_option(asio::socket_base::keep_alive(true), ec);
                    doValidate(deadline);
                }
            }));
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::doValidate(deadline_t deadline) {
    nixl::dynamicSizedBuffer validate_msg;
    validate_msg.putValues(static_cast<std::uint8_t>(tcpstore::query_type_t::VALIDATE),
                           tcpstore::validation_magic);

    initiateAsyncWrite(
        std::move(validate_msg),
        "validate",
        deadline,
        [this, deadline](nixl::dynamicSizedBuffer, asio::error_code error, std::size_t) {
            if (!error && connectionState_ == connection_state_t::CONNECTING) {
                doPing(deadline);
            }
        });
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::doPing(deadline_t deadline) {
    const auto nonce = static_cast<std::uint32_t>(::getpid());
    nixl::dynamicSizedBuffer ping_msg;
    ping_msg.putValues(static_cast<std::uint8_t>(tcpstore::query_type_t::PING), nonce);

    initiateAsyncWrite(
        std::move(ping_msg),
        "ping",
        deadline,
        [this, deadline, nonce](nixl::dynamicSizedBuffer, asio::error_code error, std::size_t) {
            if (!error && connectionState_ == connection_state_t::CONNECTING) {
                doVerifyNonce(nonce, deadline);
            }
        });
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::doVerifyNonce(std::uint32_t nonce, deadline_t deadline) {
    // Using a fixed buffer here is safe because there is at most one read
    // into the nonce buffer.
    asio::mutable_buffer buf = asio::buffer(&nonce_, sizeof(nonce_));

    initiateAsyncRead(
        buf,
        "verify nonce",
        deadline,
        [this, nonce](asio::mutable_buffer, asio::error_code error, std::size_t) {
            if (!error && connectionState_ == connection_state_t::CONNECTING) {
                if (nonce_ == nonce) {
                    connectionState_ = connection_state_t::CONNECTED;
                    connectionRetryCount_ = 0;
                    NIXL_INFO << "[TCPStore client][Server: " << host_ << ":" << port_ << "]: "
                              << "connected. Connection generation: " << generation_;

                    // Process the send queue after reconnecting.
                    processSendQueue();
                } else {
                    onError("verify nonce", asio::error::invalid_argument, false, generation_);
                }
            }
        });
}

void
nixlTcpStoreClientAsync::close() {
    asio::post(ioContext_, [this]() { initiateShutdown(opTimeout_ * 5); });
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::initiateShutdown(std::chrono::milliseconds grace_period) {
    switch (connectionState_) {
    case connection_state_t::CLOSED:
    case connection_state_t::CONNECTING:
        connectionState_ = connection_state_t::SHUTTING_DOWN;
        doShutdown();
        return;
    case connection_state_t::CONNECTED:
        connectionState_ = connection_state_t::SHUTTING_DOWN;
        connectionRetryTimer_.cancel();
        processSendQueue();

        if (connectionState_ != connection_state_t::SHUTDOWN) {
            shutdownTimer_.expires_after(grace_period);
            shutdownTimer_.async_wait([this](asio::error_code error) {
                if (connectionState_ == connection_state_t::SHUTTING_DOWN) {
                    doShutdown();
                }
            });
        }
        break;
    case connection_state_t::SHUTTING_DOWN:
    case connection_state_t::SHUTDOWN:
        break;
    }
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::doShutdown() {
    assert(connectionState_ == connection_state_t::SHUTTING_DOWN);

    if (socket_.is_open()) {
        asio::error_code ec;
        socket_.shutdown(asio::ip::tcp::socket::shutdown_both, ec);
        socket_.close(ec);
    }
    connectionRetryTimer_.cancel();
    shutdownTimer_.cancel();

    drainSendQueue();
    drainRecvQueue();

    connectionState_ = connection_state_t::SHUTDOWN;
    NIXL_INFO << "[TCPStore client][Server: " << host_ << ":" << port_ << "]: "
              << "shutdown completed.";
}

void
nixlTcpStoreClientAsync::set(const std::string &key,
                             const std::string &value,
                             std::function<void(asio::error_code)> on_complete) {
    // TCPStore SET request implementation.
    class setRequest : public Request {
    public:
        setRequest(nixlTcpStoreClientAsync &client,
                   deadline_t deadline,
                   const std::string &key,
                   const std::string &value,
                   std::function<void(asio::error_code)> on_complete)
            : Request(client, deadline),
              key_(key),
              value_(value),
              onComplete_(on_complete) {}

        std::string
        name() const override {
            return "set";
        }

        nixl::dynamicSizedBuffer
        requestMessage() override {
            nixl::dynamicSizedBuffer buf;
            buf.putValues(static_cast<std::uint8_t>(tcpstore::query_type_t::SET),
                          tcpstore::key_prefix + key_,
                          value_);
            return buf;
        }

        std::size_t
        responseMessageSize() const override {
            // SET request has no response.
            return 0;
        }

        void
        onComplete(asio::error_code error, nixl::fixedCapacityBuffer) override {
            onComplete_(error);
        }

    private:
        std::string key_;
        std::string value_;
        std::function<void(asio::error_code)> onComplete_;
    };

    scheduleRequest(std::make_unique<setRequest>(
        *this, std::chrono::steady_clock::now() + opTimeout_, key, value, on_complete));
}

void
nixlTcpStoreClientAsync::get(
    const std::string &key,
    std::function<void(asio::error_code, std::optional<std::string>)> on_complete) {
    // TCPStore GET request implementation.
    class getRequest : public Request {
    public:
        getRequest(nixlTcpStoreClientAsync &client,
                   deadline_t deadline,
                   const std::string &key,
                   std::function<void(asio::error_code, std::optional<std::string>)> on_complete)
            : Request(client, deadline),
              key_(key),
              onComplete_(on_complete) {}

        std::string
        name() const override {
            return "get";
        }

        nixl::dynamicSizedBuffer
        requestMessage() override {
            nixl::dynamicSizedBuffer buf;
            buf.putValues(static_cast<std::uint8_t>(tcpstore::query_type_t::GET),
                          tcpstore::key_prefix + key_);
            return buf;
        }

        std::size_t
        responseMessageSize() const override {
            return std::dynamic_extent;
        }

        void
        onComplete(asio::error_code error, nixl::fixedCapacityBuffer response) override {
            std::optional<std::string> value;
            if (error || response.size() == 0) {
                value = std::nullopt;
            } else {
                value = std::string(reinterpret_cast<char *>(response.data()), response.size());
            }
            onComplete_(error, value);
        }

    private:
        std::string key_;
        std::function<void(asio::error_code, std::optional<std::string>)> onComplete_;
    };

    // TCPStore CHECK request implementation.
    class checkRequest : public Request {
    public:
        checkRequest(nixlTcpStoreClientAsync &client,
                     deadline_t deadline,
                     const std::string &key,
                     std::function<void(asio::error_code, std::optional<std::string>)> on_complete)
            : Request(client, deadline),
              key_(key),
              onComplete_(on_complete) {}

        std::string
        name() const override {
            return "check";
        }

        nixl::dynamicSizedBuffer
        requestMessage() override {
            nixl::dynamicSizedBuffer buf;
            buf.putValues(static_cast<std::uint8_t>(tcpstore::query_type_t::CHECK),
                          static_cast<std::uint64_t>(1),
                          tcpstore::key_prefix + key_);
            return buf;
        }

        std::size_t
        responseMessageSize() const override {
            return sizeof(tcpstore::check_response_t);
        }

        void
        onComplete(asio::error_code error, nixl::fixedCapacityBuffer buf) override {
            if (error) {
                onComplete_(error, std::nullopt);
                return;
            }

            tcpstore::check_response_t response;
            memcpy(&response, buf.data(), sizeof(tcpstore::check_response_t));
            if (response == tcpstore::check_response_t::READY) {
                client_.scheduleRequest(
                    std::make_unique<getRequest>(client_, deadline_, key_, onComplete_));
            } else {
                onComplete_(asio::error_code(), std::nullopt);
            }
        }

    private:
        std::string key_;
        std::function<void(asio::error_code, std::optional<std::string>)> onComplete_;
    };

    scheduleRequest(std::make_unique<checkRequest>(
        *this, std::chrono::steady_clock::now() + opTimeout_, key, on_complete));
}

void
nixlTcpStoreClientAsync::deleteKey(const std::string &key,
                                   std::function<void(asio::error_code, bool)> on_complete) {
    // TCPStore DELETE_KEY request implementation.
    class deleteKeyRequest : public Request {
    public:
        deleteKeyRequest(nixlTcpStoreClientAsync &client,
                         deadline_t deadline,
                         const std::string &key,
                         std::function<void(asio::error_code, bool)> on_complete)
            : Request(client, deadline),
              key_(key),
              onComplete_(on_complete) {}

        std::string
        name() const override {
            return "delete_key";
        }

        nixl::dynamicSizedBuffer
        requestMessage() override {
            nixl::dynamicSizedBuffer buf;
            buf.putValues(static_cast<std::uint8_t>(tcpstore::query_type_t::DELETE_KEY),
                          tcpstore::key_prefix + key_);
            return buf;
        }

        std::size_t
        responseMessageSize() const override {
            return sizeof(std::int64_t);
        }

        void
        onComplete(asio::error_code error, nixl::fixedCapacityBuffer buf) override {
            if (error) {
                onComplete_(error, false);
            } else {
                std::int64_t num_deleted;
                memcpy(&num_deleted, buf.data(), sizeof(std::int64_t));
                onComplete_(asio::error_code(), (num_deleted == 1));
            }
        }

    private:
        std::string key_;
        std::function<void(asio::error_code, bool)> onComplete_;
    };

    scheduleRequest(std::make_unique<deleteKeyRequest>(
        *this, std::chrono::steady_clock::now() + opTimeout_, key, on_complete));
}

// This function is thread-safe and can be called from any thread.
void
nixlTcpStoreClientAsync::scheduleRequest(std::unique_ptr<Request> request) {
    bool is_queued = false;
    {
        // Concurrent correctness:
        // We must check the connection state under the lock, so that:
        //   - either we see the SHUTDOWN/SHUTTING_DOWN state and fail the
        //     request in-place; or,
        //   - the processSendQueue() called after setting the SHUTTING_DOWN
        //     inside initiateShutdown() sees our inserted request.
        const std::lock_guard lock(sendQueueMutex_);
        if (connectionState_ != connection_state_t::SHUTTING_DOWN &&
            connectionState_ != connection_state_t::SHUTDOWN) {
            sendQueue_.push_back(std::move(request));
            is_queued = true;
        }
    }

    if (is_queued) {
        asio::post(ioContext_, [this]() { processSendQueue(); });
    } else {
        // After shutdown, the worker thread may have terminated.
        // So, we have no choice but to fail the request in-place.
        request->onComplete(asio::error::shut_down, nixl::fixedCapacityBuffer(0));
    }
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::processSendQueue() {
    switch (connectionState_) {
    case connection_state_t::CLOSED:
        doResolve(opTimeout_);
        return;
    case connection_state_t::CONNECTING:
        // Wait for the connection to be established. When the connection
        // is established, processSendQueue() will be called again.
        return;
    case connection_state_t::CONNECTED:
    case connection_state_t::SHUTTING_DOWN:
        if (hasInflightSend_) {
            // We must NOT send a new request while the previous one is
            // in flight. Wait for the send completion handler of the
            // in-flight request to trigger the next send.
            return;
        } else {
            std::unique_ptr<Request> request;
            {
                const std::lock_guard lock(sendQueueMutex_);
                if (!sendQueue_.empty()) {
                    request = std::move(sendQueue_.front());
                    sendQueue_.pop_front();
                }
            }

            if (!request) {
                if (connectionState_ == connection_state_t::SHUTTING_DOWN && !hasInflightRecv_ &&
                    recvQueue_.empty()) {
                    doShutdown();
                }
                return;
            }

            if (std::chrono::steady_clock::now() >= request->deadline_) {
                // Fail expired requests without initiating any I/O operations.
                // Otherwise, the I/O operation will timeout immediately and
                // trigger error handling which close the connection unnecessarily.
                request->onComplete(asio::error::timed_out, nixl::fixedCapacityBuffer(0));
                asio::post(ioContext_, [this]() { processSendQueue(); });
                return;
            }

            hasInflightSend_ = true;

            request->connectionGeneration_ = generation_;
            Request *request_ptr = request.get();

            // Transfer ownership of the request to the completion handler.
            auto send_complete = [this, request = std::move(request)](nixl::dynamicSizedBuffer,
                                                                      asio::error_code error,
                                                                      std::size_t) mutable {
                hasInflightSend_ = false;

                if (!error) {
                    recvQueue_.push_back(std::move(request));
                    processRecvQueue();
                } else {
                    request->onComplete(error, nixl::fixedCapacityBuffer(0));
                }

                // Even on error, we want to call processSendQueue() because
                // it triggers reconnection if needed.
                processSendQueue();
            };

            initiateAsyncWrite(request_ptr->requestMessage(),
                               request_ptr->name(),
                               request_ptr->deadline_,
                               send_complete);
        }
        break;
    case connection_state_t::SHUTDOWN: {
        const std::lock_guard lock(sendQueueMutex_);
        assert(sendQueue_.empty());
        break;
    }
    }
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::processRecvQueue() {
    if (hasInflightRecv_) {
        // We must NOT receive a new response while the previous one is
        // in flight. Wait for the receive completion handler of the
        // in-flight request to trigger the next receive.
        return;
    } else if (!recvQueue_.empty()) {
        auto request = std::move(recvQueue_.front());
        recvQueue_.pop_front();
        Request *request_ptr = request.get();

        switch (connectionState_) {
        case connection_state_t::CLOSED:
        case connection_state_t::CONNECTING:
        case connection_state_t::SHUTDOWN:
            request->onComplete(asio::error::connection_reset, nixl::fixedCapacityBuffer(0));
            break;
        case connection_state_t::CONNECTED:
        case connection_state_t::SHUTTING_DOWN:
            if (generation_ != request->connectionGeneration_) {
                request->onComplete(asio::error::connection_reset, nixl::fixedCapacityBuffer(0));
                break;
            }

            std::size_t response_size = request_ptr->responseMessageSize();
            if (response_size == 0) {
                request->onComplete(asio::error_code(), nixl::fixedCapacityBuffer(0));
                break;
            }

            hasInflightRecv_ = true;

            auto recv_complete =
                [this, request = std::move(request)](nixl::fixedCapacityBuffer response,
                                                     asio::error_code error,
                                                     std::size_t bytes_transferred) mutable {
                    hasInflightRecv_ = false;
                    response.setSize(bytes_transferred);
                    request->onComplete(error, std::move(response));
                    processRecvQueue();
                };

            if (response_size != std::dynamic_extent) {
                nixl::fixedCapacityBuffer response(response_size);
                initiateAsyncRead(std::move(response),
                                  request_ptr->name(),
                                  request_ptr->deadline_,
                                  recv_complete);
            } else {
                nixl::fixedCapacityBuffer msg_size_buf(sizeof(std::uint64_t));
                initiateAsyncRead(
                    std::move(msg_size_buf),
                    request_ptr->name(),
                    request_ptr->deadline_,
                    [this, request_ptr, recv_complete = std::move(recv_complete)](
                        nixl::fixedCapacityBuffer msg_size_buf,
                        asio::error_code error,
                        std::size_t) mutable {
                        if (!error && generation_ == request_ptr->connectionGeneration_ &&
                            (connectionState_ == connection_state_t::CONNECTED ||
                             connectionState_ == connection_state_t::SHUTTING_DOWN)) {
                            std::uint64_t msg_size;
                            memcpy(&msg_size, msg_size_buf.data(), sizeof(std::uint64_t));

                            if (msg_size == 0) {
                                recv_complete(nixl::fixedCapacityBuffer(0), asio::error_code(), 0);
                            } else if (msg_size > tcpstore::max_blob_bytes) {
                                onError(request_ptr->name(),
                                        asio::error::message_size,
                                        false,
                                        generation_);
                                recv_complete(
                                    nixl::fixedCapacityBuffer(0), asio::error::message_size, 0);
                            } else {
                                nixl::fixedCapacityBuffer response(msg_size);
                                initiateAsyncRead(std::move(response),
                                                  request_ptr->name(),
                                                  request_ptr->deadline_,
                                                  recv_complete);
                            }
                        } else {
                            recv_complete(nixl::fixedCapacityBuffer(0),
                                          error ? error : asio::error::connection_reset,
                                          0);
                        }
                    });
            }
            break;
        }

        processRecvQueue();
    } else {
        // No receive task.
        if (connectionState_ == connection_state_t::SHUTTING_DOWN) {
            bool has_work_left = false;
            {
                const std::lock_guard lock(sendQueueMutex_);
                has_work_left = (hasInflightSend_ || !sendQueue_.empty());
            }
            if (!has_work_left) {
                doShutdown();
            }
        }
    }
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::drainSendQueue() {
    std::deque<std::unique_ptr<Request>> requests;
    {
        const std::lock_guard lock(sendQueueMutex_);
        requests.swap(sendQueue_);
    }

    for (auto &r : requests) {
        r->onComplete(asio::error::connection_reset, nixl::fixedCapacityBuffer(0));
    }
    requests.clear();
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::drainRecvQueue() {
    for (auto &request : recvQueue_) {
        request->onComplete(asio::error::connection_reset, nixl::fixedCapacityBuffer(0));
    }
    recvQueue_.clear();
}

// This function must be called from the I/O context.
void
nixlTcpStoreClientAsync::onError(std::string what,
                                 asio::error_code error,
                                 bool is_timeout,
                                 std::int32_t op_generation) {
    if (is_timeout) {
        NIXL_ERROR << "[TCPStore client][Server: " << host_ << ":" << port_ << "]: " << what
                   << " timed out. Close the connection.";
    } else {
        NIXL_ERROR << "[TCPStore client][Server: " << host_ << ":" << port_ << "]: " << what
                   << " failed: " << error.message() << ". Close the connection.";
    }

    // There may be multiple async operations in flight. We can close the socket
    // only if the socket is still in the same generation. We do not want to
    // close the socket opened for reconnection, which has a different generation.
    if (op_generation == generation_) {
        // If there is an error, we lose the track of message boundaries.
        // Therefore, we must close the socket to reset the stream.
        socket_.close();

        // Fail all pending receive requests. These are requests that have sent
        // their requests to the server but have not initiated the receive
        // operation yet. Since the socket is closed, there is no way to receive
        // the response.
        //
        // For requests that have initiated the receive operations, we rely on
        // their I/O completion handlers to complete the requests.
        //
        // There is no need to drain the send queue because those are requests
        // that have not been sent to the server yet. They will be executed
        // when the connection is established again.
        drainRecvQueue();

        if (connectionState_ == connection_state_t::CONNECTING ||
            connectionState_ == connection_state_t::CONNECTED) {
            connectionState_ = connection_state_t::CLOSED;
            if (connectionRetryCount_ < max_connection_retry_count) {
                connectionRetryTimer_.expires_after(connection_retry_interval);
                connectionRetryTimer_.async_wait([this](asio::error_code error) {
                    if (!error) {
                        doResolve(opTimeout_);
                    }
                });
            } else {
                NIXL_ERROR << "[TCPStore client][Server: " << host_ << ":" << port_ << "]: "
                           << "Max connection retry count reached. Close the connection.";
                drainSendQueue();
            }
        } else if (connectionState_ == connection_state_t::SHUTTING_DOWN) {
            doShutdown();
        }
    }
}
