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
#include "metadata_stream.h"
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "common/nixl_log.h"

nixlMetadataStream::nixlMetadataStream(uint16_t port) noexcept : port(port) {}

nixlMDStreamListener::nixlMDStreamListener(uint16_t port) noexcept : nixlMetadataStream(port) {}

nixlMDStreamListener::~nixlMDStreamListener() {
    if (listenerThread.joinable()) {
        listenerThread.join();
    }
}

nixl::scopedFd
nixlMDStreamListener::setupStream(int family) {
    sockaddr_storage listener_addr{};
    socklen_t length;
    switch (family) {
    case AF_INET6: {
        auto *const addr = reinterpret_cast<sockaddr_in6 *>(&listener_addr);
        addr->sin6_family = AF_INET6;
        addr->sin6_addr = in6addr_any;
        addr->sin6_port = htons(port);
        length = sizeof(*addr);
        break;
    }
    case AF_INET: {
        auto *const addr = reinterpret_cast<sockaddr_in *>(&listener_addr);
        addr->sin_family = AF_INET;
        addr->sin_addr.s_addr = INADDR_ANY;
        addr->sin_port = htons(port);
        length = sizeof(*addr);
        break;
    }
    default:
        throw std::runtime_error("Unsupported metadata listener address family");
    }

    nixl::scopedFd fd(socket(family, SOCK_STREAM | SOCK_NONBLOCK, 0));
    if (!fd.valid()) {
        if (family == AF_INET6 &&
            (errno == EAFNOSUPPORT || errno == EPROTONOSUPPORT || errno == EACCES ||
             errno == EPERM)) {
            return {};
        }
        NIXL_PERROR << "failed to create stream socket for listener";
        throw std::runtime_error("Failed to create metadata listener socket");
    }
    if (family == AF_INET6) {
        // Explicitly allow IPv4 peers to connect through the IPv6 listener.
        const int v6only = 0;
        if (setsockopt(fd.get(), IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only)) < 0) {
            if (errno == ENOPROTOOPT) {
                return {};
            }
            NIXL_PERROR << "setsockopt(IPV6_V6ONLY) failed while setting up listener for MD";
            throw std::runtime_error("Failed to configure metadata listener socket");
        }
    }

    const int opt = 1;
    if (setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        NIXL_PERROR << "setsockopt(REUSEADDR) failed while setting up listener for MD";
        throw std::runtime_error("Failed to configure metadata listener socket");
    }

    if (bind(fd.get(), reinterpret_cast<const sockaddr *>(&listener_addr), length) < 0) {
        if (family == AF_INET6 && errno == EADDRNOTAVAIL) {
            return {};
        }
        NIXL_PERROR << "Socket Bind failed while setting up listener for MD";
        throw std::runtime_error("Failed to bind metadata listener socket");
    }
    return fd;
}

void
nixlMDStreamListener::setupListener() {
    auto fd = setupStream(AF_INET6);
    if (!fd.valid()) {
        fd = setupStream(AF_INET);
    }

    sockaddr_storage bound_addr;
    socklen_t addr_len = sizeof(bound_addr);
    if (getsockname(fd.get(), reinterpret_cast<sockaddr *>(&bound_addr), &addr_len) != 0) {
        throw std::runtime_error(
            std::string("getsockname() failed to retrieve bound port for metadata listener: ") +
            std::strerror(errno));
    }
    uint16_t bound_port;
    switch (bound_addr.ss_family) {
    case AF_INET6:
        bound_port = ntohs(reinterpret_cast<const sockaddr_in6 *>(&bound_addr)->sin6_port);
        break;
    case AF_INET:
        bound_port = ntohs(reinterpret_cast<const sockaddr_in *>(&bound_addr)->sin_port);
        break;
    default:
        throw std::runtime_error("Unsupported metadata listener address family");
    }

    if (listen(fd.get(), 128) < 0) {
        NIXL_PERROR << "Listening failed for stream Socket: " << fd.get();
        throw std::runtime_error("Failed to listen on metadata listener socket");
    }

    const bool os_assigned_port = (port == 0);
    port = bound_port;
    socketFd = std::move(fd);
    const std::string log_msg = "MD listener is listening on port ";
    if (os_assigned_port) {
        NIXL_INFO << log_msg << port;
    } else {
        NIXL_DEBUG << log_msg << port;
    }
}

nixl::scopedFd
nixlMDStreamListener::acceptClient() {
    if (!socketFd.valid()) {
        return {};
    }
    nixl::scopedFd client(accept(socketFd.get(), nullptr, nullptr));
    if (!client.valid() && errno != EAGAIN) {
        NIXL_PERROR << "Cannot accept client connection";
    }
    return client;
}

void nixlMDStreamListener::acceptClientsAsync() {
    while(true) {
        nixl::scopedFd clientSocket(accept(socketFd.get(), nullptr, nullptr));
        if (!clientSocket.valid()) {
            NIXL_PERROR << "Cannot accept client connection";
            continue;
        }
        NIXL_DEBUG << "Client connected.";
        std::thread clientThread(
            &nixlMDStreamListener::recvFromClients, this, std::move(clientSocket));
        clientThread.detach();
    }
}

std::string nixlMDStreamListener::recvFromClient() {
        char            buffer[RECV_BUFFER_SIZE];
        int             bytes_read;
        std::string     recvData;

        bytes_read = recv(csock.get(), buffer, sizeof(buffer), 0);

        if (bytes_read > 0) {
                recvData = std::string(buffer, bytes_read);
        } else if (bytes_read == 0) {
                NIXL_DEBUG << "Client Disconnected";
        } else {
                NIXL_ERROR << "Error receiving data";
        }
        return recvData;
}

void
nixlMDStreamListener::recvFromClients(nixl::scopedFd clientSocket) {
    char buffer[RECV_BUFFER_SIZE];
    int bytes_read;

    while ((bytes_read = recv(clientSocket.get(), buffer, sizeof(buffer), 0)) > 0) {
        buffer[bytes_read] = '\0';
        // Return ack
        std::string ack = "Message received";
        send(clientSocket.get(), ack.c_str(), ack.size(), 0);
        std::string recv_message(buffer);
        NIXL_DEBUG << "Message Received" << recv_message;
    }
    NIXL_DEBUG << "Client Disconnected";
}

void nixlMDStreamListener::startListenerForClient() {
    setupListener();
    csock = acceptClient();
}


void nixlMDStreamListener::startListenerForClients() {
    setupListener();
    listenerThread = std::thread(&nixlMDStreamListener::acceptClientsAsync,
                                 this);
}

nixlMDStreamClient::nixlMDStreamClient(const std::string &listenerAddress, uint16_t port)
    : nixlMetadataStream(port),
      listenerAddress(listenerAddress) {}

bool nixlMDStreamClient::setupClient() {
    nixl::scopedFd fd(socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0));
    if (!fd.valid()) {
        NIXL_PERROR << "Failed to create metadata client socket";
        return false;
    }

    sockaddr_in listenerAddr{};
    listenerAddr.sin_family = AF_INET;
    listenerAddr.sin_port   = htons(port);

    if (inet_pton(AF_INET, listenerAddress.c_str(), &listenerAddr.sin_addr) <= 0) {
        NIXL_PERROR << "Invalid address/ Address not supported";
        return false;
    }

    if (connect(fd.get(), (struct sockaddr *)&listenerAddr, sizeof(listenerAddr)) < 0) {
        NIXL_PERROR << "Connection Failed";
        return false;
    }
    socketFd = std::move(fd);
    NIXL_DEBUG << "Connected to listener at " << listenerAddress << ":" << port;
    return true;
}

bool nixlMDStreamClient::connectListener() {
   return setupClient();
}


void nixlMDStreamClient::sendData(const std::string &data) {
    if (send(socketFd.get(), data.c_str(), data.size(), 0) < 0) {
        NIXL_ERROR << "Send failed";
    }
}

std::string nixlMDStreamClient::recvData() {
    char buffer[RECV_BUFFER_SIZE];
    int bytes_read = recv(socketFd.get(), buffer, sizeof(buffer), 0);
    if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            return std::string(buffer);
    }
    return "";
}
