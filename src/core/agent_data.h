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
#ifndef __AGENT_DATA_H_
#define __AGENT_DATA_H_

#include "common/str_tools.h"
#include "mem_section.h"
#include "telemetry.h"
#include "stream/metadata_stream.h"
#include "sync.h"


#if HAVE_ETCD
#include <etcd/SyncClient.hpp>

namespace etcd {
class SyncClient;
}

#define NIXL_ETCD_NAMESPACE_DEFAULT "/nixl/agents/"
#endif // HAVE_ETCD

using backend_list_t = std::vector<nixlBackendEngine*>;

//Internal typedef to define metadata communication request types
//To be extended with ETCD operations
enum nixl_comm_t {
    SOCK_SEND,
    SOCK_FETCH,
    SOCK_INVAL,
    SOCK_MAX,
#if HAVE_ETCD
    ETCD_SEND,
    ETCD_FETCH,
    ETCD_INVAL
#endif // HAVE_ETCD
};

//Command to be sent to listener thread from NIXL API
// 1) Command type
// 2) IP Address
// 3) Port
// 4) Metadata to send (for sendLocalMD calls)
using nixl_comm_req_t = std::tuple<nixl_comm_t, std::string, int, nixl_blob_t>;

using nixl_socket_peer_t = std::pair<std::string, int>;

using nixl_socket_map_t = std::map<nixl_socket_peer_t, int>;

class nixlAgentData {
    private:
        std::string     name;
        nixlAgentConfig config;
        nixlLock        lock;
        bool telemetryEnabled = false;

        // some handle that can be used to instantiate an object from the lib
        std::map<std::string, void*> backendLibs;

        // Bookkeeping from backend type and memory type to backend engine
        backend_list_t                         notifEngines;
        backend_map_t                          backendEngines;
        std::array<backend_list_t, FILE_SEG+1> memToBackend;

        // Bookkeping for local connection metadata and user handles per backend
        std::unordered_map<nixl_backend_t, nixlBackendH*> backendHandles;
        std::unordered_map<nixl_backend_t, nixl_blob_t>   connMD;

        // Bookkeeping from GPU request handles to backend engines
        std::unordered_map<nixlGpuXferReqH, nixlBackendEngine *> gpuReqToEngine;

        // Bookkeeping from memory view handles to backend engines
        std::unordered_map<nixlMemoryViewH, nixlBackendEngine &> mvhToEngine;

        // Local section, and Remote sections and their available common backends
        nixlLocalSection*                                        memorySection;

        std::unordered_map<std::string,
                           std::unordered_map<nixl_backend_t, nixl_blob_t>,
                           std::hash<std::string>, strEqual>     remoteBackends;
        std::unordered_map<std::string, nixlRemoteSection*,
                           std::hash<std::string>, strEqual>     remoteSections;

        // State/methods for listener thread
        nixlMDStreamListener *listener;
        nixl_socket_map_t remoteSockets;
        std::thread commThread;
        std::vector<nixl_comm_req_t> commQueue;
        std::mutex commLock;
        std::atomic<bool> commThreadStop;
        std::atomic<bool> agentShutdown;
        bool useEtcd;
        std::unique_ptr<nixlTelemetry> telemetry_;
        std::exception_ptr commThreadException_;

        void
        commWorker(nixlAgent &myAgent) noexcept;
        void
        commWorkerInternal(nixlAgent *myAgent);
        void enqueueCommWork(nixl_comm_req_t request);
        void getCommWork(std::vector<nixl_comm_req_t> &req_list);
        nixl_status_t
        loadConnInfo(const std::string &remote_name,
                     const nixl_backend_t &backend,
                     const nixl_blob_t &conn_info);
        nixl_status_t
        loadRemoteSections(const std::string &remote_name, nixlSerDes &sd);
        nixl_status_t
        invalidateRemoteData(const std::string &remote_name);
        [[nodiscard]] static backend_set_t
        getBackends(const nixl_opt_args_t *opt_args, nixlMemSection *section, nixl_mem_t mem_type);

    public:
        nixlAgentData(const std::string &name, const nixlAgentConfig &cfg);
        ~nixlAgentData();

        inline void
        addErrorTelemetry(nixl_status_t err_status) {
            if (telemetry_) telemetry_->updateErrorCount(err_status);
        }

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

    friend class nixlAgentData;
    friend class nixlAgent;
};

#endif
