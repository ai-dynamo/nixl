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

#include "nixl_metadata_context.h"
#include "stream/metadata_stream.h"
#include "common/nixl_log.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

// Socket helpers, moved from the former nixl_listener.cpp. They are the wire
// mechanics of the P2P transport and belong with this backend.

// A peer supplies the frame length, so it is bounded before it drives an
// allocation. Metadata blobs are orders of magnitude smaller than this.
constexpr size_t max_frame_bytes = 1UL << 30; // 1 GiB

// A peer that sends a length prefix and then stalls would otherwise hold the
// worker in the forced body read indefinitely.
constexpr auto frame_body_timeout = std::chrono::seconds(5);

int
connectToIP(const std::string &ip_addr, int port) {
    struct sockaddr_in listenerAddr{};
    listenerAddr.sin_port = htons(port);
    listenerAddr.sin_family = AF_INET;

    if (inet_pton(AF_INET, ip_addr.c_str(), &listenerAddr.sin_addr) <= 0) {
        NIXL_ERROR << "inet_pton failed for ip_addr: " << ip_addr;
        return -1;
    }

    int ret_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (ret_fd == -1) {
        NIXL_ERROR << "socket creation failed for ip_addr: " << ip_addr << " and port: " << port;
        return -1;
    }

    int ret = connect(ret_fd, (struct sockaddr *)&listenerAddr, sizeof(listenerAddr));
    if (ret < 0 && errno != EINPROGRESS) {
        close(ret_fd);
        return -1;
    }

    struct pollfd pfd;
    pfd.fd = ret_fd;
    pfd.events = POLLOUT;
    pfd.revents = 0;

    ret = poll(&pfd, 1, 1000); // 1000ms timeout
    if (ret <= 0) {
        if (ret < 0) {
            NIXL_PERROR << "poll failed for ip_addr: " << ip_addr << " and port: " << port;
        } else {
            NIXL_ERROR << "poll timed out for ip_addr: " << ip_addr << " and port: " << port;
        }
        close(ret_fd);
        return -1;
    }

    if (!(pfd.revents & POLLOUT)) {
        NIXL_ERROR << "poll returned but socket not ready for write for ip_addr: " << ip_addr
                   << " and port: " << port;
        close(ret_fd);
        return -1;
    }

    int error = 0;
    socklen_t len = sizeof(error);
    if (getsockopt(ret_fd, SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
        NIXL_PERROR << "getsockopt failed for ip_addr: " << ip_addr << " and port: " << port;
        close(ret_fd);
        return -1;
    }

    if (error != 0) {
        errno = error; // For the 'PERROR'.
        NIXL_PERROR << "getsockopt gave error for ip_addr: " << ip_addr << " and port: " << port;
        close(ret_fd);
        return -1;
    }

    return ret_fd;
}

void
sendCommMessage(int fd, const std::string &msg) {
    size_t size = msg.size();
    constexpr size_t iov_size = 2;
    struct iovec iov[iov_size] = {{&size, sizeof(size)},
                                  {const_cast<char *>(msg.data()), msg.size()}};

    for (size_t i = 0, offset = 0, sent = 0; i < iov_size;) {
        auto bytes = send(fd,
                          static_cast<char *>(iov[i].iov_base) + offset,
                          iov[i].iov_len - offset,
                          MSG_NOSIGNAL);
        if (bytes < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            throw std::runtime_error(
                absl::StrFormat("sendCommMessage(fd=%d, msg=%s) %zu/%zu bytes failed, errno=%d",
                                fd,
                                msg.c_str(),
                                sent,
                                size + sizeof(size),
                                errno));
        }
        offset += bytes;
        sent += bytes;
        if (offset == iov[i].iov_len) {
            offset = 0;
            ++i;
        }
    }
}

bool
recvCommMessageType(int fd, void *data, size_t size, bool force = false) {
    const auto deadline = std::chrono::steady_clock::now() + frame_body_timeout;
    for (size_t received = 0; received < size;) {
        auto bytes = recv(fd, static_cast<char *>(data) + received, size - received, 0);
        if (bytes > 0) {
            received += bytes;
            continue;
        }
        if (bytes == 0 && received == 0 && !force) {
            return false;
        }
        if (bytes < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (!force && received == 0) {
                    return false; // nothing to read yet
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error(
                        absl::StrFormat("recvCommMessage(fd=%d) timed out after %zu/%zu bytes",
                                        fd,
                                        received,
                                        size));
                }
                continue;
            }
        }
        throw std::runtime_error(
            absl::StrFormat("recvCommMessage(fd=%d) %zu/%zu bytes failed ret=%d errno=%d",
                            fd,
                            received,
                            size,
                            bytes,
                            errno));
    }
    return true;
}

bool
recvCommMessage(int fd, std::string &msg) {
    size_t size = 0;
    if (!recvCommMessageType(fd, &size, sizeof(size))) {
        return false;
    }
    if (size > max_frame_bytes) {
        throw std::runtime_error(
            absl::StrFormat("recvCommMessage(fd=%d) declared %zu bytes, over the %zu cap",
                            fd,
                            size,
                            max_frame_bytes));
    }
    msg.resize(size);
    return recvCommMessageType(fd, msg.data(), size, true);
}

} // namespace

nixlP2PMetadataBackend::nixlP2PMetadataBackend(nixlMetadataContext &ctx, const nixlMDConfig &config)
    : ctx_(ctx),
      config_(config) {
    if (config_.useListenThread) {
        listener_ = std::make_unique<nixlMDStreamListener>(config_.listenPort);
        listener_->setupListener(); // throws on bind/listen failure
    }
}

nixlP2PMetadataBackend::~nixlP2PMetadataBackend() {
    // Before closing the sockets, not just when worker_ is destroyed after this
    // body: its poll reads from remoteSockets_.
    worker_.stop();
    for (auto &[peer, fd] : remoteSockets_) {
        close(fd);
    }
}

std::string_view
nixlP2PMetadataBackend::name() const {
    return "P2P";
}

bool
nixlP2PMetadataBackend::usesThread() const {
    return config_.useListenThread;
}

void
nixlP2PMetadataBackend::start() {
    if (config_.useListenThread) {
        worker_.start([this] { serviceEvents(); }, config_.workerDelay);
    }
}

void
nixlP2PMetadataBackend::stop() {
    worker_.stop();
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
    worker_.submit([this, ip, port, blob = std::move(blob)]() {
        sendToPeer(ip, port, "NIXLCOMM:LOAD" + blob);
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
    worker_.submit([this, ip, port, blob = std::move(blob)]() {
        sendToPeer(ip, port, "NIXLCOMM:LOAD" + blob);
    });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlP2PMetadataBackend::fetchRemote(const std::string & /*remote_name*/,
                                    const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // Socket fetch is keyed by address, not name; the reply is loaded into the
    // remote-section cache by serviceEvents() when the peer answers.
    const std::string ip = extra_params->ipAddr;
    const int port = extra_params->port;
    worker_.submit([this, ip, port]() { sendToPeer(ip, port, "NIXLCOMM:SEND"); });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlP2PMetadataBackend::invalidateLocal(const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const std::string ip = extra_params->ipAddr;
    const int port = extra_params->port;
    worker_.submit(
        [this, ip, port]() { sendToPeer(ip, port, "NIXLCOMM:INVL" + ctx_.agentName()); });
    return NIXL_SUCCESS;
}

void
nixlP2PMetadataBackend::serviceEvents() {
    acceptPeers();
    readIncoming();
}

void
nixlP2PMetadataBackend::sendToPeer(const std::string &ip, int port, const std::string &msg) {
    const auto key = std::make_pair(ip, port);
    auto client = remoteSockets_.find(key);
    if (client == remoteSockets_.end()) {
        const int new_client = connectToIP(ip, port);
        if (new_client == -1) {
            NIXL_ERROR << "P2P backend could not connect to IP " << ip << " and port " << port;
            return;
        }
        client = remoteSockets_.emplace(key, new_client).first;
    }
    try {
        sendCommMessage(client->second, msg);
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
        close(client->second);
        remoteSockets_.erase(client);
    }
}

void
nixlP2PMetadataBackend::acceptPeers() {
    int new_fd = 0;
    while (new_fd != -1) {
        new_fd = listener_->acceptClient();
        if (new_fd == -1) {
            break;
        }
        sockaddr_in client_address;
        socklen_t client_addrlen = sizeof(client_address);
        if (getpeername(new_fd, (sockaddr *)&client_address, &client_addrlen) != 0) {
            NIXL_PERROR << "getpeername failed for accepted client";
            close(new_fd);
            continue;
        }
        char client_ip[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &client_address.sin_addr, client_ip, INET_ADDRSTRLEN) == nullptr) {
            NIXL_PERROR << "inet_ntop failed for client address";
            close(new_fd);
            continue;
        }
        remoteSockets_[std::make_pair(std::string(client_ip),
                                      (int)ntohs(client_address.sin_port))] = new_fd;
        const int flags = fcntl(new_fd, F_GETFL, 0);
        if (flags == -1 || fcntl(new_fd, F_SETFL, flags | O_NONBLOCK) == -1) {
            NIXL_PERROR << "fcntl failed for accepted client";
        }
    }
}

void
nixlP2PMetadataBackend::readIncoming() {
    auto socket_iter = remoteSockets_.begin();
    while (socket_iter != remoteSockets_.end()) {
        std::string commands;
        bool disconnected = false;

        try {
            if (!recvCommMessage(socket_iter->second, commands)) {
                ++socket_iter;
                continue;
            }
        }
        // std::exception, not std::runtime_error: a frame that fails to allocate
        // throws std::bad_alloc, which must disconnect the peer rather than
        // escape serviceEvents() and take down the worker.
        catch (const std::exception &e) {
            NIXL_ERROR << "Failed to receive message from peer, disconnecting: " << e.what();
            close(socket_iter->second);
            socket_iter = remoteSockets_.erase(socket_iter);
            continue;
        }

        for (const auto &command : absl::StrSplit(commands, "NIXLCOMM:")) {
            if (command.size() < 4) {
                continue;
            }
            const std::string header = std::string(command.substr(0, 4));

            if (header == "LOAD") {
                std::string remote_agent;
                const nixl_status_t ret =
                    ctx_.loadRemoteMD(std::string(command.substr(4)), remote_agent);
                if (ret != NIXL_SUCCESS) {
                    NIXL_ERROR << "loadRemoteMD in P2P backend failed from peer "
                               << socket_iter->first.first << ":" << socket_iter->first.second
                               << " with error " << ret;
                }
            } else if (header == "SEND") {
                nixl_blob_t blob;
                (void)ctx_.getLocalMD(blob);
                try {
                    sendCommMessage(socket_iter->second, "NIXLCOMM:LOAD" + blob);
                }
                catch (const std::runtime_error &e) {
                    NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
                    disconnected = true;
                    break;
                }
            } else if (header == "INVL") {
                // No break: the rest of this batch is still ours to dispatch.
                (void)ctx_.invalidateRemoteMD(std::string(command.substr(4)));
            } else {
                NIXL_ERROR << "Received socket message with bad header " << header << " from peer "
                           << socket_iter->first.first << ":" << socket_iter->first.second;
            }
        }

        if (disconnected) {
            close(socket_iter->second);
            socket_iter = remoteSockets_.erase(socket_iter);
        } else {
            ++socket_iter;
        }
    }
}
