/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>
#include <unistd.h>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

nixlMetadataStream::nixlMetadataStream(int port): port(port), socketFd(-1) {
    memset(&listenerAddr, 0, sizeof(listenerAddr));
}

nixlMetadataStream::~nixlMetadataStream() {
    closeStream();
}

bool nixlMetadataStream::setupStream() {

    socketFd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (socketFd == -1) {
        std::cerr << "failed to create stream socket for listener";
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
    setupStream();

    if (bind(socketFd, (struct sockaddr*)&listenerAddr,
             sizeof(listenerAddr)) < 0) {
        std::cerr << "Socket Bind failed while setting up listener for MD\n";
        closeStream();
        return;
    }

    if (listen(socketFd, 128) < 0) {
        std::cerr << "Listening failed for stream Socket: "
                  << socketFd  << "\n";
        closeStream();
        return;
    }
    std::cout << "MD listener is listening on port "
              << port << "...\n";
}

int nixlMDStreamListener::acceptClient() {
        csock = accept(socketFd, NULL, NULL);
        if (csock < 0 && errno != EAGAIN) {
            std::cerr << "Cannot accept client connection\n"
                      << strerror(errno) << std::endl;
        }
        return csock;
}


void nixlMDStreamListener::acceptClientsAsync() {
    while(true) {
        int clientSocket = accept(socketFd, NULL, NULL);
        if (clientSocket < 0) {
            std::cerr << "Cannot accept client connection\n"
                      << strerror(errno) << std::endl;
            continue;
        }
        std::cout << "Client connected.\n";
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
                std::cout << "Client Disconnectd" <<std::endl;
        } else {
                std::cerr << "Error receiving data" << std::endl;
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
              std::cout << "Message Received" << recv_message <<"\n";
        }
        close(clientSocket);
        std::cout << "Client Disconnected\n";
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
    setupStream();

    struct sockaddr_in listenerAddr;
    listenerAddr.sin_family = AF_INET;
    listenerAddr.sin_port   = htons(port);

    if (inet_pton(AF_INET, listenerAddress.c_str(),
                  &listenerAddr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        return false;
    }

    if (connect(socketFd, (struct sockaddr*)&listenerAddr,
                sizeof(listenerAddr)) < 0) {
        std::cerr << "Connection Failed: "<< strerror(errno) << std::endl;
        closeStream();
        return false;
    }
    std::cout << "Connected to listener at "
              << listenerAddress << ":" << port << "\n";
    return true;
}

bool nixlMDStreamClient::connectListener() {
   return setupClient();
}


void nixlMDStreamClient::sendData(const std::string &data) {
    if (send(socketFd, data.c_str(), data.size(), 0) < 0) {
        std::cerr << "Send failed\n";
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
