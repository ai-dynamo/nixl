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

#include "str_tools.h"
#include "mem_section.h"

class nixlAgentData {
    private:
        std::string     name;
        nixlAgentConfig config;

        // some handle that can be used to instantiate and object from the lib
        std::map<std::string, void*>                           backendLibs;

        std::unordered_map<nixl_backend_t, nixlBackendH*>      backendHandles;
        std::unordered_map<nixl_backend_t, nixlBackendEngine*> backendEngines;
        std::unordered_map<nixl_backend_t, std::string>        connMD; // Local info

        nixlLocalSection                                       memorySection;

        std::unordered_map<std::string, nixlRemoteSection*,
                           std::hash<std::string>, strEqual>   remoteSections;
        std::unordered_map<std::string, backend_set_t,
                           std::hash<std::string>, strEqual>   remoteBackends;

        nixlAgentData(const std::string &name, const nixlAgentConfig &cfg);
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
