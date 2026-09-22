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
#ifndef NIXL_SRC_CORE_AGENT_DATA_H
#define NIXL_SRC_CORE_AGENT_DATA_H

#include "mem_section.h"
#include "nixl_md_manager.h"
#include "nixl_metadata_context.h"
#include "telemetry.h"
#include "tracing/trace.h"
#include "sync.h"

#include <atomic>
#include <functional>
#include <memory>

using backend_list_t = std::vector<nixlBackendEngine*>;

// Implements nixlMetadataContext, which is the whole surface a metadata backend
// sees of the agent: serialization, cache load and invalidation, nothing else.
// Preserve the grandfathered 8-space class layout below.
// clang-format off
class nixlAgentData final : public nixlMetadataContext {
    private:
        const std::string name_;
        const nixlAgentConfig config_;
        // Agent-owned metadata manager; always built (single metadata path).
        // It owns the pluggable backends, which own their own transport state
        // (sockets/listener for P2P, client for ETCD) and their own threads.
        // Declared before `lock` because whether any backend runs a thread is
        // what decides the effective sync mode.
        nixlMDManager md_;
        nixlLock        lock;
        std::atomic<bool> efaWarningChecked = false;

        // some handle that can be used to instantiate an object from the lib
        std::map<std::string, void*> backendLibs;

        // Bookkeeping from backend type and memory type to backend engine
        backend_list_t                         notifEngines;
        std::array<backend_list_t, FILE_SEG+1> memToBackend;

        // Bookkeeping from memory view handles to backend engines
        std::unordered_map<nixlMemViewH, nixlBackendEngine &> mvhToEngine;

        std::unordered_map<std::string, std::unordered_map<nixl_backend_t, nixl_blob_t>>
            remoteBackends_;

        // The order of the following data members is crucial for destruction.
        // Bookkeeping for local connection metadata and user handles per backend
        std::unordered_map<nixl_backend_t, std::unique_ptr<nixlBackendH>> backendHandles_;
        std::unordered_map<nixl_backend_t, nixl_blob_t> connMd_;
        backend_map_t backendEngines_;
        // Owning shared_ptr per registration generation; weak refs in handles expire on
        // invalidation or re-registration.
        std::unordered_map<std::string, std::shared_ptr<nixlRemoteSection>> remoteSections_;
        std::unique_ptr<nixlTelemetry> telemetry_;
        // Composite tracer (fans out to every enabled backend); null when no
        // backend is active.
        const std::unique_ptr<nixl::trace::Tracer> tracer_;
        nixlLocalSection localSection_;

        // nixlMetadataContext impl; private as before (backends call via the interface).
        [[nodiscard]] nixl_status_t
        getLocalMD(nixl_blob_t &blob) override;
        [[nodiscard]] nixl_status_t
        getLocalPartialMD(const nixl_reg_dlist_t &descs,
                          nixl_blob_t &blob,
                          const nixl_opt_args_t *extra_params) override;
        [[nodiscard]] const std::string &
        agentName() const noexcept override {
            return name_;
        }
        [[nodiscard]] nixl_status_t
        loadRemoteMD(const nixl_blob_t &blob, std::string &out_name) override;
        nixl_status_t
        invalidateRemoteMD(const std::string &remote_name) override;
        nixl_status_t
        loadConnInfo(const std::string &remote_name,
                     const nixl_backend_t &backend,
                     const nixl_blob_t &conn_info);
        nixl_status_t
        loadRemoteSections(const std::string &remote_name, nixlSerDes &sd);
        [[nodiscard]] static backend_set_t
        getBackends(const nixl_opt_args_t *opt_args,
                    const nixlMemSection &section,
                    nixl_mem_t mem_type);
        void
        warnAboutEfaHardwareMismatch();

        template<class dlist_t>
        nixl_status_t
        prepXferDlist(const std::string &agent_name,
                      const dlist_t &descs,
                      nixlDlistH *&dlist_hndl,
                      const nixl_opt_args_t *extra_params);

    public:
        nixlAgentData(const std::string &name, const nixlAgentConfig &config);

        // Stops and joins the metadata backends' worker threads before any
        // member is destroyed, so no backend work can touch the caches
        // (remoteSections_, backendEngines_) torn down after this body runs.
        ~nixlAgentData();

        void
        addErrorTelemetry(nixl_status_t err_status) {
            if (telemetry_) {
                telemetry_->updateErrorCount(err_status);
            }
        }

    friend class nixlAgent;
};

// clang-format on

class nixlBackendEngine;

// This class hides away the nixlBackendEngine from user of the Agent API
class nixlBackendH {
    private:
        nixlBackendEngine* engine;

        explicit nixlBackendH(nixlBackendEngine *engine) noexcept : engine(engine) {}

    public:
        ~nixlBackendH() = default;

        // TODO? engine->getType() returns a const nixl_backend_t&
        nixl_backend_t
        getType() const noexcept {
            return engine->getType();
        }

        bool
        supportsRemote() const {
            return engine->supportsRemote();
        }

        bool
        supportsLocal() const {
            return engine->supportsLocal();
        }

        bool
        supportsNotif() const {
            return engine->supportsNotif();
        }

    friend class nixlAgentData;
    friend class nixlAgent;
};

#endif
