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
                                        nixl_b_params_t &params) const;

        // returns the backend parameters after instantiation
        nixl_status_t getBackendOptions (const nixlBackendH* &backend,
                                         nixl_b_params_t &params) const;

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


        /*** Transfer Request Prepration ***/

        // Method 1, for when memory addresses of the transfer is not known
        // beforehand, and the transfer request is prepared with information
        // from both sides. The backend is selected automatically by NIXL,
        // while user can provide a list of backends as hints in extra_params.
        nixl_status_t prepXferFull (
                          const nixl_xfer_dlist_t &local_descs,
                          const nixl_xfer_dlist_t &remote_descs,
                          const std::string &remote_agent,
                          nixlXferReqH* &req_handle,
                          const nixl_xfer_params_t* extra_params = nullptr) const;

        // Method 2, for when memory blocks used in transfers are pre-known, but
        // selection of blocks for transfers are determined in run time.
        // There are two steps, initial preparations for each side, followed by a
        // selection by indices to prepare a full transfer request.

        // Prepares descriptors for one side of a transfer request. For local
        // initiator side, remote_agent should be passed as NIXL_INIT_AGENT.
        // For local target side in local transfers agent's own name is passed as
        // remote_agent. User can provide a list of backends as hints in
        // extra_params to limit preparations to those backends.
        nixl_status_t prepXferSide (
                          const nixl_xfer_dlist_t &descs,
                          const std::string &remote_agent,
                          nixlXferSideH* &side_handle,
                          const nixl_xfer_params_t* extra_params = nullptr) const;

        // Makes a full transfer request by selecting indices from already prepared sides.
        // NIXL automatically determines the backend that can perform the transfer.
        nixl_status_t selectXferSides (
                          const nixlXferSideH* local_side,
                          const std::vector<int> &local_indices,
                          const nixlXferSideH* remote_side,
                          const std::vector<int> &remote_indices,
                          nixlXferReqH* &req_handle,
                          const nixl_xfer_params_t* extra_params = nullptr) const;


        /*** Operations on prepared Transfer Request ***/

        // Submit a transfer request, which enables async checks on the transfer.
        // Notification message is optional, if operation does not demand for it.
        nixl_status_t postXferReq (const nixl_xfer_op_t &operation,
                                   nixlXferReqH* req,
                                   const nixl_blob_t &notif_msg=NIXL_NO_MSG);

        // Check the status of transfer requests
        nixl_status_t getXferStatus (nixlXferReqH* req);

        // User can ask for backend used in a nixlXferReqH. For example to use the
        // same backend for genNotif, or to know the decision after a prepXferFull.
        nixl_status_t getXferBackend (const nixlXferReqH* req_handle,
                                      nixlBackendH* &backend) const;

        // Invalidate (free) transfer request if we no longer need it.
        // Tries to abort a running transfer, or return error if couldn't
        nixl_status_t invalidateXferReq (nixlXferReqH* req);

        // Frees a side handle object
        nixl_status_t invalidateXferSide (nixlXferSideH* side_handle) const;


        /*** Notification Handling ***/

        // Add entries to the passed received notifications list (can be
        // non-empty). Number of new entries can be checked through new_notifs.
        // Elements are released within the Agent after this call.
        nixl_status_t getNotifs (nixl_notifs_t &notif_map, int &new_notifs);

        // Generate a notification, not bound to a transfer, e.g., for control.
        // Can be used after the remote metadata is exchanged. Will be received
        // in notif list. Providing a backend in extra_params is optional, as
        // nixl can automatically decide.
        nixl_status_t genNotif (const std::string &remote_agent,
                                const nixl_blob_t &msg,
                                const nixl_xfer_params_t* extra_params = nullptr
                               );

        /*** Metadata handling through side channel ***/

        // Get nixl metadata blob for this agent.
        nixl_status_t getLocalMD (nixl_blob_t &str) const;

        // Load other agent's metadata and unpack it internally.
        // Received agent name can be checked through agent_name.
        nixl_status_t loadRemoteMD (const nixl_blobl_t &remote_metadata,
                                    std::string &agent_name);

        // Invalidate the remote agent metadata cached locally, and disconnect from it.
        nixl_status_t invalidateRemoteMD (const std::string &remote_agent);
};

#endif
