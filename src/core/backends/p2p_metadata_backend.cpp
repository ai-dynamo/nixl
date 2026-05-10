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
#include "p2p_metadata_backend.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include <absl/strings/str_format.h>

#include "common/nixl_log.h"
#include "nixl.h"

namespace {

constexpr int kSocketRetryPollMs = 5;

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

    struct pollfd pfd{};
    pfd.fd = ret_fd;
    pfd.events = POLLOUT;

    ret = poll(&pfd, 1, 1000);
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
        errno = error;
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
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd{};
                pfd.fd = fd;
                pfd.events = POLLOUT;
                (void)poll(&pfd, 1, kSocketRetryPollMs);
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
                    return false;
                }
                struct pollfd pfd{};
                pfd.fd = fd;
                pfd.events = POLLIN;
                (void)poll(&pfd, 1, kSocketRetryPollMs);
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
    msg.resize(size);
    return recvCommMessageType(fd, msg.data(), size, true);
}

bool
parsePeerKey(const std::string &key, std::string &ip, int &port) {
    const auto colon = key.rfind(':');
    if (colon == std::string::npos) {
        return false;
    }

    try {
        ip = key.substr(0, colon);
        port = std::stoi(key.substr(colon + 1));
    }
    catch (const std::exception &) {
        return false;
    }
    return true;
}

} // namespace

nixlP2PMetadataBackend::nixlP2PMetadataBackend(int listen_port, std::string my_agent_name)
    : listen_port_(listen_port == 0 ? default_comm_port : listen_port),
      my_agent_name_(std::move(my_agent_name)) {}

nixlP2PMetadataBackend::~nixlP2PMetadataBackend() {
    for (auto &[peer, fd] : sockets_) {
        shutdown(fd, SHUT_RDWR);
        close(fd);
    }
    sockets_.clear();
}

nixl_status_t
nixlP2PMetadataBackend::publish(const std::string &key, const nixl_blob_t &value) {
    std::string ip;
    int port = 0;
    if (!parsePeerKey(key, ip, port)) {
        NIXL_ERROR << "P2P publish: malformed key (expected ip:port): " << key;
        return NIXL_ERR_INVALID_PARAM;
    }
    return sendToPeer(ip, port, value);
}

nixl_status_t
nixlP2PMetadataBackend::fetch(const std::string & /*key*/, nixl_blob_t & /*value*/) {
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlP2PMetadataBackend::remove(const std::string &key) {
    std::string ip;
    int port = 0;
    if (!parsePeerKey(key, ip, port)) {
        NIXL_ERROR << "P2P remove: malformed key (expected ip:port): " << key;
        return NIXL_ERR_INVALID_PARAM;
    }
    return invalidatePeerMetadata(ip, port, my_agent_name_);
}

bool
nixlP2PMetadataBackend::isHealthy() const noexcept {
    return listener_ != nullptr;
}

nixl_status_t
nixlP2PMetadataBackend::watch(const std::string &prefix, nixl_md_watch_cb_t cb) {
    (void)prefix;
    (void)cb;
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlP2PMetadataBackend::fetchBatch(const std::vector<std::string> &keys,
                                   std::vector<nixl_blob_t> &out,
                                   std::vector<nixl_status_t> &per_key_status) {
    out.assign(keys.size(), nixl_blob_t{});
    per_key_status.assign(keys.size(), NIXL_ERR_NOT_SUPPORTED);
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlP2PMetadataBackend::ensureSocket(const std::string &ip, int port) {
    if (socketFor(ip, port) != -1) {
        return NIXL_SUCCESS;
    }
    const int fd = connectToIP(ip, port);
    if (fd == -1) {
        NIXL_ERROR << "Listener thread could not connect to IP " << ip << " and port " << port;
        return NIXL_ERR_BACKEND;
    }
    sockets_[{ip, port}] = fd;
    return NIXL_SUCCESS;
}

int
nixlP2PMetadataBackend::socketFor(const std::string &ip, int port) const {
    const auto it = sockets_.find({ip, port});
    if (it == sockets_.end()) {
        return -1;
    }
    return it->second;
}

void
nixlP2PMetadataBackend::forgetSocket(const std::string &ip, int port) {
    const auto it = sockets_.find({ip, port});
    if (it == sockets_.end()) {
        return;
    }
    forgetSocket(it);
}

void
nixlP2PMetadataBackend::forgetSocket(nixl::md::socket_map_t::iterator it) {
    if (it == sockets_.end()) {
        return;
    }
    close(it->second);
    sockets_.erase(it);
}

nixl_status_t
nixlP2PMetadataBackend::sendToPeer(const std::string &ip, int port, const nixl_blob_t &blob) {
    const nixl_status_t ret = ensureSocket(ip, port);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }
    const int fd = socketFor(ip, port);
    if (fd == -1) {
        return NIXL_ERR_BACKEND;
    }

    try {
        sendCommMessage(fd, "NIXLCOMM:LOAD" + blob);
        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
        forgetSocket(ip, port);
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlP2PMetadataBackend::requestMetadataFromPeer(const std::string &ip, int port) {
    const nixl_status_t ret = ensureSocket(ip, port);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }
    const int fd = socketFor(ip, port);
    if (fd == -1) {
        return NIXL_ERR_BACKEND;
    }

    try {
        sendCommMessage(fd, "NIXLCOMM:SEND");
        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
        forgetSocket(ip, port);
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlP2PMetadataBackend::invalidatePeerMetadata(const std::string &ip,
                                               int port,
                                               const std::string &my_agent_name) {
    const nixl_status_t ret = ensureSocket(ip, port);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }
    const int fd = socketFor(ip, port);
    if (fd == -1) {
        return NIXL_ERR_BACKEND;
    }

    try {
        sendCommMessage(fd, "NIXLCOMM:INVL" + my_agent_name);
        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
        forgetSocket(ip, port);
        return NIXL_ERR_BACKEND;
    }
}

void
nixlP2PMetadataBackend::processOnce(nixlAgent &agent) {
    // Accept new client connections.
    if (listener_) {
        for (;;) {
            const int new_fd = listener_->acceptClient();
            if (new_fd == -1) {
                break;
            }

            sockaddr_in client_address{};
            socklen_t client_addrlen = sizeof(client_address);
            if (getpeername(new_fd, (sockaddr *)&client_address, &client_addrlen) != 0) {
                NIXL_PERROR << "getpeername failed for accepted client";
                close(new_fd);
                continue;
            }

            char client_ip[INET_ADDRSTRLEN]{};
            if (inet_ntop(AF_INET, &client_address.sin_addr, client_ip, INET_ADDRSTRLEN) ==
                nullptr) {
                NIXL_PERROR << "inet_ntop failed for client address";
                close(new_fd);
                continue;
            }

            const int cur_flags = fcntl(new_fd, F_GETFL, 0);
            if (cur_flags == -1 || fcntl(new_fd, F_SETFL, cur_flags | O_NONBLOCK) == -1) {
                NIXL_PERROR << "fcntl on accepted client failed";
                close(new_fd);
                continue;
            }

            sockets_[{std::string(client_ip), ntohs(client_address.sin_port)}] = new_fd;
        }
    }

    // Process inbound messages from already-connected peers.
    auto it = sockets_.begin();
    while (it != sockets_.end()) {
        std::string commands;
        bool disconnected = false;

        try {
            const bool received = recvCommMessage(it->second, commands);
            if (!received) {
                ++it;
                continue;
            }
        }
        catch (const std::runtime_error &e) {
            NIXL_ERROR << "Failed to receive message from peer, disconnecting: " << e.what();
            close(it->second);
            it = sockets_.erase(it);
            continue;
        }

        const std::string prefix = "NIXLCOMM:";
        if (commands.rfind(prefix, 0) != 0 || commands.size() < prefix.size() + 4) {
            NIXL_ERROR << "Received socket message with bad prefix from peer " << it->first.first
                       << ":" << it->first.second << "; disconnecting";
            disconnected = true;
        } else {
            const std::string header(commands.substr(prefix.size(), 4));
            const std::string payload(commands.substr(prefix.size() + 4));

            if (header == "LOAD") {
                std::string remote_agent;
                const nixl_status_t ret = agent.loadRemoteMD(payload, remote_agent);
                if (ret != NIXL_SUCCESS) {
                    NIXL_ERROR << "loadRemoteMD in listener thread failed for md from peer "
                               << it->first.first << ":" << it->first.second << " with error "
                               << ret;
                }
            } else if (header == "SEND") {
                nixl_blob_t my_md;
                const nixl_status_t ret = agent.getLocalMD(my_md);
                if (ret != NIXL_SUCCESS) {
                    NIXL_ERROR << "getLocalMD in listener thread failed with error " << ret;
                } else {
                    try {
                        sendCommMessage(it->second, std::string("NIXLCOMM:LOAD" + my_md));
                    }
                    catch (const std::runtime_error &e) {
                        NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
                        disconnected = true;
                    }
                }
            } else if (header == "INVL") {
                agent.invalidateRemoteMD(payload);
            } else {
                NIXL_ERROR << "Received socket message with bad header " << header << " from peer "
                           << it->first.first << ":" << it->first.second << "; disconnecting";
                disconnected = true;
            }
        }

        if (disconnected) {
            close(it->second);
            it = sockets_.erase(it);
        } else {
            ++it;
        }
    }
}

nixl::md::socket_map_t
nixlP2PMetadataBackend::detachSockets() {
    nixl::md::socket_map_t out;
    out.swap(sockets_);
    return out;
}

void
nixlP2PMetadataBackend::setupListener() {
    listener_ = std::make_unique<nixlMDStreamListener>(listen_port_);
    listener_->setupListener();
}
