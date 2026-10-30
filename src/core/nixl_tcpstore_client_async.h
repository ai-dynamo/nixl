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
/**
 * @file nixl_tcpstore_client_async.h
 * @brief Asynchronous in-house client for the PyTorch c10d TCPStore wire protocol.
 *
 * Core-internal: speaks to the same server torch.distributed.TCPStore connects
 * to (no libtorch dependency). Only the subset nixlTcpStoreMetadataBackend
 * needs is implemented. Values are opaque byte blobs; the framing matches c10d
 * (uint64 length prefixes in host byte order), so it interoperates on
 * same-endian hosts.
 */
#ifndef NIXL_SRC_CORE_NIXL_TCPSTORE_CLIENT_ASYNC_H
#define NIXL_SRC_CORE_NIXL_TCPSTORE_CLIENT_ASYNC_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <asio.hpp>

#include "common/byte_buffer.h"

// All requests are asynchronous. Request completion handlers are invoked on
// the given execution context. If a request is started after stop() is called,
// its completion handler is invoked immediately with an error before the request
// call is finished. Request completion handlers must NOT block indefinitely,
// and should return fast. Otherwise, it may block the execution of other requests.
class nixlTcpStoreClientAsync {
public:
    // Construct a client with the given configuration.
    // Called at Agent construction time. Must NOT perform any I/O.
    //
    // @param host The TCPStore host. Both IPv4 and IPv6 are supported.
    // @param port The TCPStore port.
    // @param connect_timeout The timeout for the initial connection.
    // @param op_timeout The timeout for each operation.
    // @param io_context The execution context for request completion handlers.
    nixlTcpStoreClientAsync(std::string host,
                            std::uint16_t port,
                            std::chrono::milliseconds connect_timeout,
                            std::chrono::milliseconds op_timeout,
                            asio::io_context &io_context) noexcept;

    ~nixlTcpStoreClientAsync();

    nixlTcpStoreClientAsync(const nixlTcpStoreClientAsync &) = delete;
    nixlTcpStoreClientAsync &
    operator=(const nixlTcpStoreClientAsync &) = delete;

    // Initiate the connection to the TCPStore server if not already connected.
    // The TCPStore client always try to auto connect if no connection is
    // established to handle requests. This function is used by the upper layer
    // to explicitly initiate the connection.
    //
    // This function is thread-safe.
    void
    connect();

    // Close the connection to the TCPStore server.
    // Once close() is called, auto connect is disabled.
    //
    // This function is thread-safe.
    void
    close();

    // Initiate a SET (upsert) request with the given key and value.
    // Upon completion, the handler is invoked with the error code.
    //
    // This function is thread-safe.
    void
    set(const std::string &key,
        const std::string &value,
        std::function<void(asio::error_code)> on_complete);

    // Initiate a CHECK followed by a GET request with the given key.
    // Upon completion, the handler is invoked with the error code and the value
    // for the key (or nullopt if the key is not found). The presence of the key
    // is resolved internally with the CHECK query, whose answer for a missing
    // key is defined (a bare GET is not).
    //
    // This function is thread-safe.
    void
    get(const std::string &key,
        std::function<void(asio::error_code, std::optional<std::string>)> on_complete);

    // Initiate a DELETE_KEY request with the given key.
    // Upon completion, the handler is invoked with the error code and true if
    // exactly one key was deleted, false otherwise.
    //
    // This function is thread-safe.
    void
    deleteKey(const std::string &key, std::function<void(asio::error_code, bool)> on_complete);

private:
    using deadline_t = std::chrono::steady_clock::time_point;
    // Forward declaration. The base class for all TCPStore requests.
    class Request;

    // Resolve the hostname to the TCPStore server.
    void
    doResolve(std::chrono::milliseconds timeout);

    // Connect to the TCPStore server.
    void
    doConnect(deadline_t deadline);

    // VALIDATE with the TCPStore server.
    void
    doValidate(deadline_t deadline);

    // PING the TCPStore server.
    void
    doPing(deadline_t deadline);

    // Verify the nonce from the PING response.
    void
    doVerifyNonce(std::uint32_t nonce, deadline_t deadline);

    // Shutdown the connection to the TCPStore server.
    void
    initiateShutdown(std::chrono::milliseconds grace_period);

    void
    doShutdown();

    // Schedule a request to be sent to the TCPStore server.
    void
    scheduleRequest(std::unique_ptr<Request> request);

    // Process the send queue.
    void
    processSendQueue();

    // Process the receive queue.
    void
    processRecvQueue();

    // Drain the send queue.
    void
    drainSendQueue();

    // Drain the receive queue.
    void
    drainRecvQueue();

    // Handle I/O error.
    void
    onError(std::string what, asio::error_code error, bool is_timeout, std::int32_t op_generation);


    // Helper function to initiate an asynchronous read operation with
    // cancellation and error handling. If the buffer owns the memory,
    // the ownership is transferred to the completion handler.
    //
    // The completion handler's signature must be:
    // void (Buffer, asio::error_code, std::size_t)
    template<typename Buffer, typename Fn>
        requires std::is_same_v<Buffer, nixl::fixedCapacityBuffer> ||
        std::is_same_v<Buffer, asio::mutable_buffer>
    void
    initiateAsyncRead(Buffer buf, std::string what, deadline_t deadline, Fn &&on_complete);

    // Helper function to initiate an asynchronous write operation with
    // cancellation and error handling. The ownership of the buffer is transferred
    // to the completion handler.
    //
    // The completion handler's signature must be:
    // void (nixl::inputByteBuffer,asio::error_code, std::size_t)
    template<typename Fn>
    void
    initiateAsyncWrite(nixl::dynamicSizedBuffer buf,
                       std::string what,
                       deadline_t deadline,
                       Fn &&on_complete);

    // Helper function that adds timeout cancellation and error handling to
    // the given completion handler.
    template<typename Fn>
    auto
    cancelAndErrorAdapter(std::string what, deadline_t deadline, Fn &&on_complete) {
        return cancelAndErrorAdapter(
            std::move(what), deadline, std::forward<Fn>(on_complete), &Fn::operator());
    }

    // Overload function that unpacks the arguments from the completion handler.
    template<typename Fn, typename... Args>
    auto
    cancelAndErrorAdapter(std::string what,
                          deadline_t deadline,
                          Fn &&on_complete,
                          void (Fn::*)(asio::error_code, Args...));

    // Client configuration.
    const std::string host_;
    const std::uint16_t port_;
    const std::string portStr_;
    const std::chrono::milliseconds connectTimeout_;
    const std::chrono::milliseconds opTimeout_;

    // The execution context for I/O operations and completion handlers.
    asio::io_context &ioContext_;

    //////////////////////////////////////////////////////////////
    // See the implementation notes in nixl_tcpstore_client_async.cpp for
    // more details on the threading model on the following variables.

    //// The following variables manage the connection state. ////

    // No need to use mutex to protect these variables because they are only
    // accessed from the I/O context.
    enum class connection_state_t {
        CLOSED,
        CONNECTING,
        CONNECTED,
        SHUTTING_DOWN,
        SHUTDOWN,
    };
    // The resolved endpoints of the TCPStore server.
    asio::ip::tcp::resolver::results_type resolvedEndpoints_;
    // The socket to the TCPStore server.
    asio::ip::tcp::socket socket_;
    // The state of the connection.
    std::atomic<connection_state_t> connectionState_ = connection_state_t::CLOSED;
    // The nonce from the PING response.
    std::uint32_t nonce_;
    // The generation of the current connection.
    std::int32_t generation_ = 0;
    // The number of times the connection has been retried.
    std::int32_t connectionRetryCount_ = 0;
    // The timer for the connection retry.
    asio::steady_timer connectionRetryTimer_;
    // The timer for the shutdown grace period.
    asio::steady_timer shutdownTimer_;

    //// The following variables manage the request state. ////

    // Queued requests to be sent to the TCPStore server and the mutex to protect it.
    // The caller thread produces requests and the I/O context consumes them.
    std::mutex sendQueueMutex_;
    std::deque<std::unique_ptr<Request>> sendQueue_;
    bool hasInflightSend_ = false;

    // Queued requests waiting for responses from the TCPStore server.
    // The receive task queue is accessed exclusively from the I/O context.
    std::deque<std::unique_ptr<Request>> recvQueue_;
    bool hasInflightRecv_ = false;
};

#endif // NIXL_SRC_CORE_NIXL_TCPSTORE_CLIENT_ASYNC_H
