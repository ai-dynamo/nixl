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
#ifndef __METADATA_STREAM_H
#define __METADATA_STREAM_H

#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/un.h>
#include <thread>
#include <mutex>
#include <string>
#include <queue>
#include <vector>
#include <netinet/in.h>
#include "nixl_types.h"

#define RECV_BUFFER_SIZE 16384

class nixlMetadataStream {
    protected:
        int                 port;
        int                 socketFd;
        std::string         listenerAddress;
        struct sockaddr_in  listenerAddr;

        bool setupStream();
        void closeStream();

    public:
        nixlMetadataStream(int port);
        ~nixlMetadataStream();
};


class nixlMDStreamListener: public nixlMetadataStream {
    private:
        std::thread listenerThread;
        int csock = -1;

        void            acceptClientsAsync();
        void            recvFromClients(int clientSocket);

    public:
        nixlMDStreamListener(int port);
        ~nixlMDStreamListener();

        int         acceptClient();
        void        setupListener();
        void        startListenerForClients();
        void        startListenerForClient();
        std::string recvFromClient();
};

class nixlMDStreamClient: public nixlMetadataStream {
    private:
        int         csock;
        std::string listenerAddress;
        bool setupClient();

    public:
        nixlMDStreamClient(const std::string& listenerAddress, int port);
        ~nixlMDStreamClient();

        bool connectListener();
        void sendData(const std::string& data);
        std::string recvData();
};

// This class talks to the metadata server.
class nixlMetadataH {
    private:
        // Maybe the connection information should go to Agent,
        // to add p2p support
        std::string   ipAddress;
        uint16_t      port;

    public:
        // Creates the connection to the metadata server
        nixlMetadataH() {}
        nixlMetadataH(const std::string &ip_address, uint16_t port);
        ~nixlMetadataH();

        /** Sync the local section with the metadata server */
        nixl_status_t sendLocalMetadata(const std::string &local_md);

        // Get a remote section from the metadata server
        std::string getRemoteMd(const std::string &remote_agent);

        // Invalidating the information in the metadata server
        nixl_status_t removeLocalMetadata(const std::string &local_agent);
};

#endif
