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
/** NVIDIA Inference Xfer Library */
#ifndef _NIXL_H
#define _NIXL_H

#include "nixl_types.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"

// Main transfer object
class nixlAgent {
    private:
        nixlAgentData* data;

    public:

        /*** Initialization and Registering Methods ***/

        // Populates agent name and device metadata
        nixlAgent (const std::string &name, const nixlAgentConfig &cfg);
        ~nixlAgent ();

        // Returns the available plugins found in the paths.
        nixl_status_t getAvailPlugins (std::vector<nixl_backend_t> &plugins);

        // Returns the supported configs with their default values
        nixl_status_t getPluginOptions (const nixl_backend_t &type,
                                        nixl_b_params_t &params);

        // returns the backend parameters after instantiation
        nixl_status_t getBackendOptions (const nixlBackendH* backend,
                                         nixl_b_params_t &params);

        // Instantiate BackendEngine objects, based on corresponding params
        nixl_status_t createBackend (const nixl_backend_t &type,
                                     const nixl_b_params_t &params,
                                     nixlBackendH* &backend);

        // Register with the backend and populate memory_section
        nixl_status_t registerMem (const nixl_reg_dlist_t &descs,
                                   nixlBackendH* backend);
        // Deregister and remove from memory section
        nixl_status_t deregisterMem (const nixl_reg_dlist_t &descs,
                                     nixlBackendH* backend);

        // Make connection proactively, instead of at transfer time
        nixl_status_t makeConnection (const std::string &remote_agent);


        /*** Transfer Request Handling ***/

        // Creates a transfer request, with automatic backend selection if null.
        nixl_status_t createXferReq (const nixl_xfer_dlist_t &local_descs,
                                     const nixl_xfer_dlist_t &remote_descs,
                                     const std::string &remote_agent,
                                     const nixl_blob_t &notif_msg,
                                     const nixl_xfer_op_t &operation,
                                     nixlXferReqH* &req_handle,
                                     const nixlBackendH* backend = nullptr) const;

        // Submit a transfer request, which populates the req async handler.
        nixl_status_t postXferReq (nixlXferReqH* req);

        // Check the status of transfer requests
        nixl_status_t getXferStatus (nixlXferReqH* req);

        // Invalidate transfer request if we no longer need it.
        // Will also abort a running transfer.
        nixl_status_t invalidateXferReq (nixlXferReqH* req);


        /*** Alternative method to create transfer handle manually ***/

        // User can ask for backend chosen for a XferReq to use it for prepXferSide.
        nixl_status_t getXferBackend(const nixlXferReqH* req_handle,
                                     nixlBackendH* &backend) const;

        // Prepares descriptors for one side of a transfer with given backend.
        // Empty string for remote_agent means it's local side.
        nixl_status_t prepXferSide (const nixl_xfer_dlist_t &descs,
                                    const std::string &remote_agent,
                                    const nixlBackendH* backend,
                                    nixlXferSideH* &side_handle) const;

        // Makes a transfer request from already prepared side transfer handles.
        nixl_status_t makeXferReq (const nixlXferSideH* local_side,
                                   const std::vector<int> &local_indices,
                                   const nixlXferSideH* remote_side,
                                   const std::vector<int> &remote_indices,
                                   const nixl_blob_t &notif_msg,
                                   const nixl_xfer_op_t &operation,
                                   nixlXferReqH* &req_handle) const;

        nixl_status_t invalidateXferSide (nixlXferSideH* side_handle) const;

        /*** Notification Handling ***/

        // Add entries to the passed received notifications list (can be
        // non-empty), and return number of added entries, or -1 if there was
        // an error. Elements are released within the Agent after this call.
        nixl_status_t getNotifs (nixl_notifs_t &notif_map,
                                 int &new_notifs);

        // Generate a notification, not bound to a transfer, e.g., for control.
        // Can be used after the remote metadata is exchanged. Will be received
        // in notif list. Nixl will choose a backend if null is passed.
        nixl_status_t genNotif (const std::string &remote_agent,
                                const nixl_blob_t &msg,
                                nixlBackendH* backend = nullptr);

        /*** Metadata handling through side channel ***/

        // Get nixl_metadata for this agent. Empty string means error.
        // The std::string used for serialized MD can have \0 values.
        nixl_status_t getLocalMD (nixl_blob_t &str) const;

        // Load other agent's metadata and unpack it internally.
        // Returns the found agent name in metadata, or "" in case of error.
        nixl_status_t loadRemoteMD (const nixl_blob_t &remote_metadata,
                                    std::string &agent_name);

        // Invalidate the remote section information cached locally
        nixl_status_t invalidateRemoteMD (const std::string &remote_agent);
};

#endif
