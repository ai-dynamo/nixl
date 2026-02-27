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
#include <cstring>
#include <stdexcept>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "common/nixl_log.h"

nixlMetadataStream::nixlMetadataStream(int port): port(port), socketFd(-1) {
    memset(&listenerAddr, 0, sizeof(listenerAddr));
}

nixlMetadataStream::~nixlMetadataStream() {
    closeStream();
}

bool nixlMetadataStream::setupStream() {

    socketFd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (socketFd == -1) {
        NIXL_PERROR << "failed to create stream socket for listener";
        return false;
    }

    listenerAddr.sin_family = AF_INET;
    listenerAddr.sin_addr.s_addr = INADDR_ANY;
    listenerAddr.sin_port = htons(port);

    return true;
}

void nixlMetadataStream::closeStream() {
   if (socketFd != -1) {
        close(socketFd);
        socketFd = -1;
   }
}


nixlMDStreamListener::nixlMDStreamListener(int port) :
        nixlMetadataStream(port) {}

nixlMDStreamListener::~nixlMDStreamListener() {
    if (listenerThread.joinable()) {
        listenerThread.join();
    }
    if (csock >= 0) {
        close(csock);
    }
}

void nixlMDStreamListener::setupListener() {
    if (!setupStream()) {
        throw std::runtime_error("Failed to create metadata listener socket");
    }

    int opt = 1;
    if (setsockopt(socketFd, SOL_SOCKET, SO_REUSEADDR,
                   &opt, sizeof(opt)) < 0) {
        NIXL_PERROR << "setsockopt(REUSEADDR) failed while setting up listener for MD";
        closeStream();
        throw std::runtime_error("Failed to configure metadata listener socket");
    }

    if (bind(socketFd, (struct sockaddr*)&listenerAddr,
             sizeof(listenerAddr)) < 0) {
        NIXL_PERROR << "Socket Bind failed while setting up listener for MD";
        closeStream();
        throw std::runtime_error("Failed to bind metadata listener socket");
    }

    if (listen(socketFd, 128) < 0) {
        NIXL_PERROR << "Listening failed for stream Socket: "
                    << socketFd;
        closeStream();
        throw std::runtime_error("Failed to listen on metadata listener socket");
    }
    NIXL_DEBUG << "MD listener is listening on port "
               << port << "...";
}

int nixlMDStreamListener::acceptClient() {
    if (socketFd < 0) {
        return -1;
    }
    csock = accept(socketFd, nullptr, nullptr);
    if (csock < 0 && errno != EAGAIN) {
        NIXL_PERROR << "Cannot accept client connection";
    }
    return csock;
}

void nixlMDStreamListener::acceptClientsAsync() {
    while(true) {
        int clientSocket = accept(socketFd, NULL, NULL);
        if (clientSocket < 0) {
            NIXL_PERROR << "Cannot accept client connection";
            continue;
        }
        NIXL_DEBUG << "Client connected.";
        std::thread clientThread(&nixlMDStreamListener::recvFromClients,
                                 this, clientSocket);
        clientThread.detach();
    }
}

std::string nixlMDStreamListener::recvFromClient() {
        char            buffer[RECV_BUFFER_SIZE];
        int             bytes_read;
        std::string     recvData;

        bytes_read = recv(csock, buffer, sizeof(buffer), 0);

        if (bytes_read > 0) {
                recvData = std::string(buffer, bytes_read);
        } else if (bytes_read == 0) {
                NIXL_DEBUG << "Client Disconnected";
        } else {
                NIXL_ERROR << "Error receiving data";
        }
        return recvData;
}

void nixlMDStreamListener::recvFromClients(int clientSocket) {
        char    buffer[RECV_BUFFER_SIZE];
        int     bytes_read;

        while ((bytes_read = recv(clientSocket, buffer,
                                  sizeof(buffer), 0)) > 0) {
              buffer[bytes_read] = '\0';
              // Return ack
              std::string ack = "Message received";
              send(clientSocket, ack.c_str(), ack.size(), 0);
              std::string recv_message(buffer);
              NIXL_DEBUG << "Message Received" << recv_message;
        }
        close(clientSocket);
        NIXL_DEBUG << "Client Disconnected";
}

void nixlMDStreamListener::startListenerForClient() {
    setupListener();
    acceptClient();
}


void nixlMDStreamListener::startListenerForClients() {
    setupListener();
    listenerThread = std::thread(&nixlMDStreamListener::acceptClientsAsync,
                                 this);
}

nixlMDStreamClient::nixlMDStreamClient(const std::string &listenerAddress,
                                       int port) : nixlMetadataStream(port),
                                       listenerAddress(listenerAddress) {}

nixlMDStreamClient::~nixlMDStreamClient() {
    closeStream();
}

bool nixlMDStreamClient::setupClient() {
    if (!setupStream()) {
        NIXL_PERROR << "Failed to create metadata client socket";
        return false;
    }

    struct sockaddr_in listenerAddr;
    listenerAddr.sin_family = AF_INET;
    listenerAddr.sin_port   = htons(port);

    if (inet_pton(AF_INET, listenerAddress.c_str(),
                  &listenerAddr.sin_addr) <= 0) {
        NIXL_PERROR << "Invalid address/ Address not supported";
        closeStream();
        return false;
    }

    if (connect(socketFd, (struct sockaddr*)&listenerAddr,
                sizeof(listenerAddr)) < 0) {
        NIXL_PERROR << "Connection Failed";
        closeStream();
        return false;
    }
    NIXL_DEBUG << "Connected to listener at "
               << listenerAddress << ":" << port;
    return true;
}

bool nixlMDStreamClient::connectListener() {
   return setupClient();
}


void nixlMDStreamClient::sendData(const std::string &data) {
    if (send(socketFd, data.c_str(), data.size(), 0) < 0) {
        NIXL_ERROR << "Send failed";
    }
}

std::string nixlMDStreamClient::recvData() {
    char buffer[RECV_BUFFER_SIZE];
    int  bytes_read = recv(socketFd, buffer, sizeof(buffer), 0);
    if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            return std::string(buffer);
    }
    return "";
}
