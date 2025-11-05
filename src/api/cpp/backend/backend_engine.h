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
#ifndef __BACKEND_ENGINE_H
#define __BACKEND_ENGINE_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <mutex>

#include "nixl_types.h"
#include "backend_aux.h"
#include "telemetry_event.h"

constexpr size_t MAX_TELEMETRY_QUEUE_SIZE = 1000;

// Base backend engine class for different backend implementations
class nixlBackendEngine {
    private:
        // Members that cannot be modified by a child backend and parent bookkeep
        nixl_backend_t  backendType;
        nixl_b_params_t customParams;
        std::vector<nixlTelemetryEvent> telemetryEvents_;
        std::mutex telemetryEventsMutex_;

    protected:
        // Members that can be accessed by the child (localAgent cannot be modified)
        bool              initErr = false;
        const std::string localAgent;
        const bool enableTelemetry_;

        [[nodiscard]] nixl_status_t
        setInitParam(const std::string &key, const std::string &value) {
            if (customParams.emplace(key, value).second) {
                return NIXL_SUCCESS;
            }
            return NIXL_ERR_NOT_ALLOWED;
        }

        [[nodiscard]] nixl_status_t getInitParam(const std::string &key, std::string &value) const {
            const auto iter = customParams.find(key);
            if (iter != customParams.end()) {
                value = iter->second;
                return NIXL_SUCCESS;
            }
            return NIXL_ERR_INVALID_PARAM;
        }

        void
        addTelemetryEvent(const std::string &event_name, uint64_t value) {
            if (!enableTelemetry_) return;
            if (telemetryEvents_.size() >= MAX_TELEMETRY_QUEUE_SIZE) return;
            std::lock_guard<std::mutex> lock(telemetryEventsMutex_);
            telemetryEvents_.emplace_back(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::system_clock::now().time_since_epoch())
                                              .count(),
                                          nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND,
                                          event_name,
                                          value);
        }

    public:
        explicit nixlBackendEngine(const nixlBackendInitParams *init_params)
            : backendType(init_params->type),
              customParams(*init_params->customParams),
              localAgent(init_params->localAgent),
              enableTelemetry_(init_params->enableTelemetry_) {}

        nixlBackendEngine(nixlBackendEngine&&) = delete;
        nixlBackendEngine(const nixlBackendEngine&) = delete;

        void operator=(nixlBackendEngine&&) = delete;
        void operator=(const nixlBackendEngine&) = delete;

        virtual ~nixlBackendEngine() = default;

        std::vector<nixlTelemetryEvent>
        getTelemetryEvents() {
            std::lock_guard<std::mutex> lock(telemetryEventsMutex_);
            return std::move(telemetryEvents_);
        }

        bool getInitErr() const noexcept { return initErr; }
        const nixl_backend_t& getType() const noexcept { return backendType; }
        const nixl_b_params_t& getCustomParams() const noexcept { return customParams; }

        // The support function determine which methods are necessary by the child backend, and
        // if they're called by mistake, they will return error if not implemented by backend.

        // Determines if a backend supports remote operations
        virtual bool supportsRemote() const = 0;

        // Determines if a backend supports local operations
        virtual bool supportsLocal() const = 0;

        // Determines if a backend supports sending notifications. Related methods are not
        // pure virtual, and return errors, as parent shouldn't call if supportsNotif is false.
        virtual bool supportsNotif() const = 0;

        virtual nixl_mem_list_t getSupportedMems() const = 0;  // TODO: Return by const-reference and mark noexcept?


        // *** Pure virtual methods that need to be implemented by any backend *** //

        // Register and deregister local memory
        virtual nixl_status_t registerMem (const nixlBlobDesc &mem,
                                           const nixl_mem_t &nixl_mem,
                                           nixlBackendMD* &out) = 0;
        virtual nixl_status_t deregisterMem (nixlBackendMD* meta) = 0;

        // Make connection to a remote node identified by the name into loaded conn infos
        // Child might just return 0, if making proactive connections are not necessary.
        // An agent might need to connect to itself for local operations.
        virtual nixl_status_t connect(const std::string &remote_agent) = 0;
        virtual nixl_status_t disconnect(const std::string &remote_agent) = 0;

        // Remove loaded local or remtoe metadata for target
        virtual nixl_status_t unloadMD (nixlBackendMD* input) = 0;

        // Preparing a request, which populates the async handle as desired
        virtual nixl_status_t prepXfer (const nixl_xfer_op_t &operation,
                                        const nixl_meta_dlist_t &local,
                                        const nixl_meta_dlist_t &remote,
                                        const std::string &remote_agent,
                                        nixlBackendReqH* &handle,
                                        const nixl_opt_b_args_t* opt_args=nullptr
                                       ) const = 0;

        // Posting a request, which completes the async handle creation and posts it
        virtual nixl_status_t postXfer (const nixl_xfer_op_t &operation,
                                        const nixl_meta_dlist_t &local,
                                        const nixl_meta_dlist_t &remote,
                                        const std::string &remote_agent,
                                        nixlBackendReqH* &handle,
                                        const nixl_opt_b_args_t* opt_args=nullptr
                                       ) const = 0;

        // Use a handle to progress backend engine and see if a transfer is completed or not
        virtual nixl_status_t checkXfer(nixlBackendReqH* handle) const = 0;

        //Backend aborts the transfer if necessary, and destructs the relevant objects
        virtual nixl_status_t releaseReqH(nixlBackendReqH* handle) const = 0;

        // Create a GPU transfer request to GPU memory for GPU transfer.
        virtual nixl_status_t
        createGpuXferReq(const nixlBackendReqH &req_hndl,
                         const nixl_meta_dlist_t &local_descs,
                         const nixl_meta_dlist_t &remote_descs,
                         nixlGpuXferReqH &gpu_req_hndl) const {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        // Release a GPU transfer request from GPU memory
        virtual void
        releaseGpuXferReq(nixlGpuXferReqH gpu_req_hndl) const {}

        // Get the size required for a GPU signal
        virtual nixl_status_t
        getGpuSignalSize(size_t &signal_size) const {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        // Initialize a signal for GPU transfer using memory handle from descriptor
        virtual nixl_status_t
        prepGpuSignal(const nixlBackendMD &meta,
                      void *signal,
                      const nixl_opt_b_args_t *opt_args = nullptr) const {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        // *** Needs to be implemented if supportsRemote() is true *** //

        // Gets serialized form of public metadata
        virtual nixl_status_t getPublicData (const nixlBackendMD* meta,
                                             std::string &str) const {
            return NIXL_ERR_BACKEND;
        };

        // Provide the required connection info for remote nodes, should be non-empty
        virtual nixl_status_t getConnInfo(std::string &str) const {
            return NIXL_ERR_BACKEND;
        }

        // Deserialize from string the connection info for a remote node, if supported
        // The generated data should be deleted in nixlBackendEngine destructor
        virtual nixl_status_t loadRemoteConnInfo (const std::string &remote_agent,
                                                  const std::string &remote_conn_info) {
            return NIXL_ERR_BACKEND;
        }

        // Load remtoe metadata, if supported.
        virtual nixl_status_t loadRemoteMD (const nixlBlobDesc &input,
                                            const nixl_mem_t &nixl_mem,
                                            const std::string &remote_agent,
                                            nixlBackendMD* &output) {
            return NIXL_ERR_BACKEND;
        }


        // *** Needs to be implemented if supportsLocal() is true *** //

        // Provide the target metadata necessary for local operations, if supported
        virtual nixl_status_t loadLocalMD (nixlBackendMD* input,
                                           nixlBackendMD* &output) {
            return NIXL_ERR_BACKEND;
        }


        // *** Needs to be implemented if supportsNotif() is true *** //

        // Populate an empty received notif list. Elements are released within backend then.
        virtual nixl_status_t getNotifs(notif_list_t &notif_list) { return NIXL_ERR_BACKEND; }

        // Generates a standalone notification, not bound to a transfer.
        virtual nixl_status_t genNotif(const std::string &remote_agent, const std::string &msg) const {
            return NIXL_ERR_BACKEND;
        }


        // *** Optional virtual methods that are good to be implemented in any backend *** //

        // Query information about a list of memory/storage
        virtual nixl_status_t
        queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const {
            // Default implementation for file backends
            // File backends can override this to provide custom implementation
            // For now, return not supported - for object backends
            return NIXL_ERR_NOT_SUPPORTED;
        }

        // Estimate the cost (duration) of a transfer operation.
        virtual nixl_status_t
        estimateXferCost(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *const &handle,
                         std::chrono::microseconds &duration,
                         std::chrono::microseconds &err_margin,
                         nixl_cost_t &method,
                         const nixl_opt_args_t *extra_params = nullptr) const {
            return NIXL_ERR_NOT_SUPPORTED;
        }
};
#endif
