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
#ifndef __AGENT_DATA_H_
#define __AGENT_DATA_H_

#include "common/str_tools.h"
#include "mem_section.h"
#include "stream/metadata_stream.h"

#define DEFAULT_COMM_PORT 8888

typedef std::vector<nixlBackendEngine*> backend_list_t;

//Internal typedef to define metadata communication request types
//To be extended with ETCD operations
typedef enum { SOCK_SEND, SOCK_FETCH, SOCK_INVAL } nixl_comm_t;
typedef std::tuple<nixl_comm_t, std::string, int> nixl_comm_req_t;

class nixlAgentData {
    private:
        std::string     name;
        nixlAgentConfig config;

        // some handle that can be used to instantiate an object from the lib
        std::map<std::string, void*> backendLibs;

        // Bookkeeping from backend type and memory type to backend engine
        backend_list_t                         notifEngines;
        backend_map_t                          backendEngines;
        std::array<backend_list_t, FILE_SEG+1> memToBackend;

        // Bookkeping for local connection metadata and user handles per backend
        std::unordered_map<nixl_backend_t, nixlBackendH*> backendHandles;
        std::unordered_map<nixl_backend_t, std::string>   connMD;

        // Local section, and Remote sections and their available common backends
        nixlLocalSection*                                        memorySection;

        std::unordered_map<std::string, std::set<nixl_backend_t>,
                           std::hash<std::string>, strEqual>     remoteBackends;
        std::unordered_map<std::string, nixlRemoteSection*,
                           std::hash<std::string>, strEqual>     remoteSections;

        // State/methods for listener thread
        nixlMDStreamListener                  *listener;
        std::unordered_map<std::string, int>  remoteSockets;
        std::thread                           commThread;
        std::vector<nixl_comm_req_t>          commQueue;
        std::mutex                            commLock;
        bool                                  commThreadStop;

        void enqueueCommWork(nixl_comm_req_t request);
        void getCommWork(std::vector<nixl_comm_req_t> &req_list);

        nixlAgentData(nixlAgent* my_agent, const std::string &name, const nixlAgentConfig &cfg);
        ~nixlAgentData();

    friend class nixlAgent;
};

class nixlBackendEngine;
// This class hides away the nixlBackendEngine from user of the Agent API
class nixlBackendH {
    private:
        nixlBackendEngine* engine;

        nixlBackendH(nixlBackendEngine* &engine) { this->engine = engine; }
        ~nixlBackendH () {}

    public:
        nixl_backend_t getType () const { return engine->getType(); }

        bool supportsRemote () const { return engine->supportsRemote(); }
        bool supportsLocal  () const { return engine->supportsLocal (); }
        bool supportsNotif  () const { return engine->supportsNotif (); }
        bool supportsProgTh () const { return engine->supportsProgTh(); }

    friend class nixlAgentData;
    friend class nixlAgent;
};

#endif
