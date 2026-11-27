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
#include "nixl_p2p_metadata_backend.h"
#include "asio/error.hpp"
#include "asio/error_code.hpp"
#include "nixl_metadata_context.h"

#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/byte_buffer.h"
#include "common/nixl_log.h"
#include "nixl_types.h"

namespace {
// A peer supplies the frame length, so it is bounded before it drives an
// allocation. Metadata blobs are orders of magnitude smaller than this.
constexpr std::size_t max_frame_bytes = 1UL << 30; // 1 GiB

// A peer that sends a partial message and then stalls must be detected
// and disconnected.
constexpr auto receive_timeout = std::chrono::seconds(5);

constexpr auto connection_retry_interval = std::chrono::seconds(1);
constexpr auto max_connection_retry_count = 10;
constexpr auto min_accept_retry_delay = std::chrono::microseconds(50000); // 50ms

constexpr auto module_prefix = "[P2P Metadata Backend]";
} // namespace

// Functions from this class must be called from the I/O context.
class nixlP2PMetadataBackend::nixlP2PMetadataPeer
    : public std::enable_shared_from_this<nixlP2PMetadataPeer> {

public:
    // Constructor for a new peer.
    nixlP2PMetadataPeer(nixlP2PMetadataBackend &backend, const peerAddress &address)
        : backend_(backend),
          peerAddress_(address),
          socket_(backend.ioContext_),
          autoConnect_(true),
          connectionState_(connection_state_t::CLOSED),
          connectionRetryTimer_(backend.ioContext_),
          shutdownTimer_(backend.ioContext_) {}

    // Set the socket to a newly accepted connection.
    void
    setAcceptedSocket(asio::ip::tcp::socket socket) {
        if (stopCalled_) {
            NIXL_WARN << getLogPrefix() << "setAcceptedSocket() called after stop(). Ignoring.";
            asio::error_code ec;
            socket.close(ec);
            return;
        }

        // The old socket must have died. Close and ignore the error just in case.
        if (socket_.is_open()) {
            asio::error_code ec;
            socket_.close(ec);
        }
        socket_ = std::move(socket);
        autoConnect_ = false;
        connectionState_ = connection_state_t::CONNECTED;
        generation_++;
        hasInflightSend_ = false;
        connectionRetryCount_ = 0;
        connectionRetryTimer_.expires_at(std::chrono::steady_clock::time_point::min());

        receiveMessageSize();
        sendNextMessage();
    }

    // Send a message to the peer.
    // Calling this function also enables auto-connect if the connection is closed.
    void
    sendMessage(std::string message) {
        if (!stopCalled_) {
            autoConnect_ = true;
            sendQueue_.push_back(std::move(message));
            sendNextMessage();
        } else {
            NIXL_WARN << getLogPrefix() << "Message dropped after stop: " << message;
        }
    }

    // Gracefully stop the async I/O operations.
    void
    stop() {
        if (stopCalled_) {
            return;
        }

        stopCalled_ = true;
        autoConnect_ = false;

        if (checkCleanup()) {
            return;
        }

        // Stop the receive loop.
        asio::error_code ec;
        socket_.shutdown(asio::ip::tcp::socket::shutdown_receive, ec);

        sendNextMessage();

        shutdownTimer_.expires_after(receive_timeout);
        shutdownTimer_.async_wait(
            [shared_this = shared_from_this()](const asio::error_code &error) {
                if (error != asio::error::operation_aborted) {
                    shared_this->cleanup();
                }
            });
    }

private:
    enum class on_stop_action_t {
        CONTINUE_ON_STOP,
        STOP,
    };

    template<on_stop_action_t StopAction,
             typename Rep,
             typename Period,
             typename Fn,
             typename... Args>
    auto
    cancelAndErrorAdapter(std::string what,
                          std::chrono::duration<Rep, Period> timeout,
                          Fn &&on_success,
                          void (Fn::*)(Args...)) {
        return asio::cancel_after(
            timeout,
            asio::cancellation_type::terminal,
            [shared_this = shared_from_this(),
             generation = this->generation_,
             what = std::move(what),
             on_success = std::move(on_success)](asio::error_code error, Args... args) mutable {
                if (error) {
                    shared_this->onError(generation, error, what);
                } else if (StopAction == on_stop_action_t::STOP && shared_this->stopCalled_) {
                    shared_this->checkCleanup();
                } else if (generation != shared_this->generation_) {
                    NIXL_WARN << shared_this->getLogPrefix()
                              << "Connection has been reset. Drop stale operation.";
                } else {
                    on_success(std::forward<Args>(args)...);
                }
            });
    }

    template<on_stop_action_t StopAction = on_stop_action_t::STOP,
             typename Rep,
             typename Period,
             typename Fn>
    auto
    cancelAndErrorAdapter(std::string what,
                          std::chrono::duration<Rep, Period> timeout,
                          Fn &&on_success) {
        return cancelAndErrorAdapter<StopAction>(
            std::move(what), timeout, std::forward<Fn>(on_success), &Fn::operator());
    }

    std::string
    getLogPrefix() const {
        return absl::StrFormat(
            "%s[Peer: %s:%d] ", module_prefix, peerAddress_.ip_, peerAddress_.port_);
    }

    void
    resolveEndpoint() {
        if (connectionState_ != connection_state_t::CLOSED) {
            return;
        }
        if (stopCalled_) {
            NIXL_INFO << getLogPrefix() << "Stop called, cancelling connection.";
            cleanup();
            return;
        }
        if (connectionRetryTimer_.expiry() > std::chrono::steady_clock::now()) {
            NIXL_DEBUG << getLogPrefix() << "Connection retry in progress.";
            return;
        }

        connectionState_ = connection_state_t::CONNECTING;
        generation_++;
        connectionRetryCount_++;
        NIXL_INFO << getLogPrefix() << " connecting to peer.";


        auto resolver = std::make_unique<asio::ip::tcp::resolver>(backend_.ioContext_);
        auto resolver_ptr = resolver.get();

        auto resolve_complete = [shared_this = shared_from_this(), r = std::move(resolver)](
                                    asio::ip::tcp::resolver::results_type results) mutable {
            shared_this->connectPeer(std::move(results));
        };

        resolver_ptr->async_resolve(
            peerAddress_.ip_,
            std::to_string(peerAddress_.port_),
            asio::ip::tcp::resolver::numeric_host | asio::ip::tcp::resolver::numeric_service,
            cancelAndErrorAdapter("resolve", receive_timeout, std::move(resolve_complete)));
    }

    void
    connectPeer(asio::ip::tcp::resolver::results_type results) {
        auto connect_complete = [shared_this =
                                     shared_from_this()](const asio::ip::tcp::endpoint &) mutable {
            shared_this->connectionState_ = connection_state_t::CONNECTED;
            shared_this->connectionRetryCount_ = 0;
            shared_this->hasInflightSend_ = false;
            shared_this->receiveMessageSize();
            shared_this->sendNextMessage();
        };

        asio::async_connect(
            socket_,
            results,
            cancelAndErrorAdapter("connect", receive_timeout, std::move(connect_complete)));
    }

    void
    receiveMessageSize() {
        if (stopCalled_) {
            checkCleanup();
            return;
        }

        messageSizeBuf_.messageSize_ = 0;

        auto receive_complete = [shared_this = shared_from_this(), generation = this->generation_](
                                    const asio::error_code &error, std::size_t bytes_transferred) {
            if (generation != shared_this->generation_) {
                NIXL_INFO << shared_this->getLogPrefix()
                          << "Connection has been reset. Drop stale message.";
                return;
            }
            if (shared_this->stopCalled_) {
                return;
            }

            constexpr std::size_t buf_size = sizeof(shared_this->messageSizeBuf_.buf_);
            if (bytes_transferred > 0) {
                if (bytes_transferred == buf_size) {
                    // We have received the message size.
                    shared_this->receiveMessageBody();
                } else {
                    // Read the remaining bytes.
                    auto buffer =
                        asio::buffer(shared_this->messageSizeBuf_.buf_ + bytes_transferred,
                                     buf_size - bytes_transferred);
                    auto read_complete = [shared_this](std::size_t) mutable {
                        shared_this->receiveMessageBody();
                    };
                    asio::async_read(
                        shared_this->socket_,
                        buffer,
                        shared_this->cancelAndErrorAdapter("receive message size failed",
                                                           receive_timeout,
                                                           std::move(read_complete)));
                }
                return;
            }

            // Nothing has been received yet.
            if (error == asio::error::interrupted || error == asio::error::try_again ||
                error == asio::error::would_block || !error) {
                // Retryable error (or no error).
                shared_this->receiveMessageSize();
                return;
            }

            shared_this->onError(generation, error, "receive message size failed");
        };

        // There is no guarantee that a complete message will ever come, so we
        // cannot use async_read() to handle short reads. Instead, we use
        // async_receive() with no timeout and we must handle short reads.
        // Once we receive at least one byte, we know a message is coming and
        // we can switch to async_read() with a timeout.
        socket_.async_receive(asio::buffer(messageSizeBuf_.buf_), receive_complete);
    }

    void
    receiveMessageBody() {
        if (messageSizeBuf_.messageSize_ > max_frame_bytes) {
            onError(generation_,
                    asio::error::make_error_code(asio::error::message_size),
                    "message corrupted");
            return;
        }

        try {
            nixl::fixedCapacityBuffer buffer(messageSizeBuf_.messageSize_);
            auto read_buf = asio::buffer(buffer.data(), buffer.capacity());

            auto read_complete = [shared_this = shared_from_this(), buffer = std::move(buffer)](
                                     std::size_t bytes_transferred) mutable {
                buffer.setSize(bytes_transferred);
                shared_this->processMessage(std::move(buffer));
            };

            asio::async_read(socket_,
                             read_buf,
                             cancelAndErrorAdapter("receive message body failed",
                                                   receive_timeout,
                                                   std::move(read_complete)));
        }
        catch (const std::exception &e) {
            onError(generation_,
                    asio::error::make_error_code(asio::error::no_memory),
                    "allocate buffer for message body");
        }
    }

    void
    processMessage(nixl::fixedCapacityBuffer buffer) {
        std::string_view message(reinterpret_cast<char *>(buffer.data()), buffer.size());
        for (const auto &command : absl::StrSplit(message, "NIXLCOMM:")) {
            if (command.size() < 4) {
                continue;
            }
            auto header = command.substr(0, 4);

            if (header == "LOAD") {
                std::string remote_agent;
                const nixl_status_t ret =
                    backend_.ctx_.loadRemoteMD(std::string(command.substr(4)), remote_agent);
                if (ret != NIXL_SUCCESS) {
                    NIXL_ERROR << getLogPrefix() << "loadRemoteMD in P2P backend failed with error "
                               << ret;
                }
            } else if (header == "SEND") {
                nixl_blob_t blob;
                (void)backend_.ctx_.getLocalMD(blob);
                sendQueue_.push_back("NIXLCOMM:LOAD" + blob);
                sendNextMessage();
            } else if (header == "INVL") {
                // No break: the rest of this batch is still ours to dispatch.
                (void)backend_.ctx_.invalidateRemoteMD(std::string(command.substr(4)));
            } else {
                NIXL_ERROR << getLogPrefix() << "Received socket message with bad header " << header
                           << ".";
            }
        }

        // Post the next receive operation.
        receiveMessageSize();
    }

    void
    sendNextMessage() {
        switch (connectionState_) {
        case connection_state_t::CLOSED:
            resolveEndpoint();
            break;
        case connection_state_t::CONNECTING:
            break;
        case connection_state_t::CONNECTED:
            if (hasInflightSend_) {
                return;
            }

            if (sendQueue_.empty()) {
                if (stopCalled_) {
                    cleanup();
                }
                return;
            }

            hasInflightSend_ = true;

            nixl::dynamicSizedBuffer buf;
            buf.put(sendQueue_.front());
            sendQueue_.pop_front();
            auto buffer = asio::buffer(buf.data(), buf.size());

            auto send_complete = [shared_this = shared_from_this(),
                                  buf = std::move(buf)](std::size_t) mutable {
                shared_this->hasInflightSend_ = false;
                shared_this->sendNextMessage();
            };

            asio::async_write(
                socket_,
                buffer,
                cancelAndErrorAdapter<on_stop_action_t::CONTINUE_ON_STOP>(
                    "send message failed", receive_timeout, std::move(send_complete)));

            break;
        }
    }

    bool
    checkCleanup() {
        if (connectionState_ != connection_state_t::CONNECTED ||
            (!hasInflightSend_ && sendQueue_.empty())) {
            cleanup();
            return true;
        }
        return false;
    }

    void
    cleanup() {
        autoConnect_ = false;

        if (socket_.is_open()) {
            asio::error_code ec;
            socket_.shutdown(asio::ip::tcp::socket::shutdown_both, ec);
            socket_.close(ec);
            NIXL_INFO << getLogPrefix() << "Closed socket.";
        }

        connectionState_ = connection_state_t::CLOSED;
        connectionRetryCount_ = 0;
        connectionRetryTimer_.cancel();
        shutdownTimer_.cancel();
        hasInflightSend_ = false;
        sendQueue_.clear();

        auto it = backend_.peerSessions_.find(peerAddress_);
        if (it != backend_.peerSessions_.end() && it->second.get() == this) {
            backend_.peerSessions_.erase(it);
        }
    }

    void
    onError(std::uint32_t generation, const asio::error_code &error, const std::string_view &msg) {
        if (error == asio::error::eof) {
            if (stopCalled_ && connectionState_ == connection_state_t::CONNECTED) {
                // We are in the graceful shutdown period, and the read operation
                // is shutdown by us.
                return;
            }
            NIXL_INFO << getLogPrefix() << "Peer closed.";
        } else {
            NIXL_ERROR << getLogPrefix() << msg << ": " << error.message();
        }

        if (generation != this->generation_) {
            return;
        }

        asio::error_code ec;
        socket_.close(ec);
        connectionState_ = connection_state_t::CLOSED;

        if (autoConnect_ && connectionRetryCount_ < max_connection_retry_count) {
            connectionRetryTimer_.expires_after(connection_retry_interval);
            connectionRetryTimer_.async_wait(
                [shared_this = shared_from_this(),
                 generation = this->generation_](const asio::error_code &error) {
                    if (error != asio::error::operation_aborted &&
                        generation == shared_this->generation_) {
                        shared_this->resolveEndpoint();
                    }
                });
        } else {
            // Destroy this session. If a future message needs to be sent to
            // the same peer, a new session will be created.
            NIXL_INFO << getLogPrefix() << "Peer session died. Cleaning up.";
            cleanup();
        }
    }

private:
    // The backend.
    nixlP2PMetadataBackend &backend_;

    // The address of the peer.
    const peerAddress peerAddress_;
    // The socket to the peer.
    asio::ip::tcp::socket socket_;
    // Whether to automatically connect to the peer.
    bool autoConnect_ = false;

    enum class connection_state_t {
        CLOSED,
        CONNECTING,
        CONNECTED,
    };
    // The state of the connection.
    connection_state_t connectionState_;
    // The generation of the connection.
    std::uint32_t generation_ = 0;
    // The number of times the connection has been retried.
    std::int32_t connectionRetryCount_ = 0;
    // The timer for the connection retry.
    asio::steady_timer connectionRetryTimer_;
    // The timer for the shutdown grace period.
    asio::steady_timer shutdownTimer_;

    // The per-peer buffer used to receive the message size.
    // This is an optimization to avoid allocating a small buffer
    // on the heap for each message.
    union {
        std::uint64_t messageSize_;
        std::uint8_t buf_[sizeof(messageSize_)];
    } messageSizeBuf_;

    // The queue of messages to send to the peer.
    std::deque<std::string> sendQueue_;
    // Whether there is an inflight send.
    bool hasInflightSend_ = false;

    // Whether the stop() method has been called.
    bool stopCalled_ = false;
};

nixlP2PMetadataBackend::nixlP2PMetadataBackend(nixlMetadataContext &ctx,
                                               const nixlMDConfig &config,
                                               asio::io_context &io_context)
    : ctx_(ctx),
      ioContext_(io_context),
      acceptor_(io_context),
      acceptRetryTimer_(io_context),
      acceptRetryInterval_(std::max(config.workerDelay, min_accept_retry_delay)) {
    if (config.useListenThread) {
        try {
            bindListener(asio::ip::tcp::endpoint(asio::ip::tcp::v6(), config.listenPort));
        }
        catch (const asio::system_error &e) {
            NIXL_INFO << getLogPrefix() << "Failed to bind IPv6 listener socket: " << e.what()
                      << ". Trying IPv4.";
            asio::error_code ec;
            acceptor_.close(ec);
            bindListener(asio::ip::tcp::endpoint(asio::ip::tcp::v4(), config.listenPort));
        }
        acceptor_.listen();

        NIXL_INFO << getLogPrefix() << "MD listener is listening on port "
                  << acceptor_.local_endpoint().port();
    }
}

nixlP2PMetadataBackend::~nixlP2PMetadataBackend() {
    stop();
}

void
nixlP2PMetadataBackend::bindListener(const asio::ip::tcp::endpoint &endpoint) {
    // Throws asio::system_error on failure.
    // This is OK because this function is only called in the constructor.
    acceptor_.open(endpoint.protocol());
    if (endpoint.address().is_v6()) {
        acceptor_.set_option(asio::ip::v6_only(false));
    }
    acceptor_.set_option(asio::socket_base::reuse_address(true));
    acceptor_.bind(endpoint);
}

void
nixlP2PMetadataBackend::start() {
    if (acceptor_.is_open()) {
        acceptPeers();
    }
}

void
nixlP2PMetadataBackend::acceptPeers() {
    acceptor_.async_accept([this](const asio::error_code &error, asio::ip::tcp::socket socket) {
        if (!acceptor_.is_open()) {
            NIXL_INFO << getLogPrefix() << "Listener closed.";
            return;
        }
        if (error) {
            NIXL_ERROR << getLogPrefix() << "Failed to accept peer: " << error.message();
            acceptRetryTimer_.expires_after(acceptRetryInterval_);
            acceptRetryTimer_.async_wait([this](const asio::error_code &error) {
                if (acceptor_.is_open()) {
                    acceptPeers();
                }
            });
            return;
        }

        try {
            auto peer_endpoint = socket.remote_endpoint();
            auto peer_address = peer_endpoint.address();
            if (peer_address.is_v6()) {
                auto peer_address_v6 = peer_address.to_v6();
                if (peer_address_v6.is_v4_mapped()) {
                    peer_address = asio::ip::make_address_v4(asio::ip::v4_mapped, peer_address_v6);
                }
            }

            peerAddress address{.ip_ = peer_address.to_string(), .port_ = peer_endpoint.port()};
            getOrCreateSession(address)->setAcceptedSocket(std::move(socket));
        }
        catch (const std::exception &e) {
            NIXL_ERROR << getLogPrefix() << "Failed to get peer address: " << e.what();
            asio::error_code ec;
            socket.close(ec);
        }

        acceptPeers();
    });
}

void
nixlP2PMetadataBackend::stop() {
    asio::post(ioContext_, [this]() {
        if (stopped_) {
            return;
        }
        stopped_ = true;

        asio::error_code ec;
        acceptor_.close(ec);
        acceptRetryTimer_.cancel();

        std::vector<std::shared_ptr<nixlP2PMetadataPeer>> sessions;
        sessions.reserve(peerSessions_.size());
        for (auto &[_, session] : peerSessions_) {
            sessions.push_back(session);
        }

        for (auto &session : sessions) {
            session->stop();
        }
    });
}

nixl_status_t
nixlP2PMetadataBackend::sendLocal(const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    nixl_blob_t blob;
    const nixl_status_t ret = ctx_.getLocalMD(blob);
    if (ret < 0) {
        return ret;
    }
    const std::string ip = extra_params->ipAddr;
    const int port = extra_params->port;

    asio::post(ioContext_, [this, ip, port, blob = std::move(blob)]() {
        sendMessage(ip, port, "NIXLCOMM:LOAD" + blob);
    });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlP2PMetadataBackend::sendLocalPartial(const nixl_reg_dlist_t &descs,
                                         const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    nixl_blob_t blob;
    const nixl_status_t ret = ctx_.getLocalPartialMD(descs, blob, extra_params);
    if (ret < 0) {
        return ret;
    }
    const std::string ip = extra_params->ipAddr;
    const int port = extra_params->port;

    asio::post(ioContext_, [this, ip, port, blob = std::move(blob)]() {
        sendMessage(ip, port, "NIXLCOMM:LOAD" + blob);
    });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlP2PMetadataBackend::fetchRemote(const std::string &remote_name,
                                    const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // Socket fetch is keyed by address, not name; the reply is loaded into the
    // remote-section cache by serviceEvents() when the peer answers.
    const std::string ip = extra_params->ipAddr;
    const int port = extra_params->port;

    asio::post(ioContext_, [this, ip, port]() { sendMessage(ip, port, "NIXLCOMM:SEND"); });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlP2PMetadataBackend::invalidateLocal(const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const std::string ip = extra_params->ipAddr;
    const int port = extra_params->port;

    asio::post(ioContext_,
               [this, ip, port]() { sendMessage(ip, port, "NIXLCOMM:INVL" + ctx_.agentName()); });
    return NIXL_SUCCESS;
}

// Must be called from the I/O context.
nixlP2PMetadataBackend::nixlP2PMetadataPeer *
nixlP2PMetadataBackend::getOrCreateSession(const peerAddress &address) {
    auto it = peerSessions_.find(address);
    if (it == peerSessions_.end()) {
        auto session = std::make_shared<nixlP2PMetadataPeer>(*this, address);
        auto session_ptr = session.get();
        peerSessions_.emplace(address, std::move(session));
        return session_ptr;
    } else {
        return it->second.get();
    }
}

void
nixlP2PMetadataBackend::sendMessage(const std::string &ip, int port, std::string message) {
    if (stopped_) {
        NIXL_WARN << getLogPrefix() << "Stopped. Skipping message: " << message;
        return;
    }
    peerAddress address{.ip_ = ip, .port_ = port};
    getOrCreateSession(address)->sendMessage(std::move(message));
}

std::string
nixlP2PMetadataBackend::getLogPrefix() const {
    return absl::StrFormat("%s[%s] ", module_prefix, ctx_.agentName());
}
