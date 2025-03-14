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
        nixlAgent (const std::string &name,
                   const nixlAgentConfig &cfg);
        ~nixlAgent ();

        // Returns the available plugins found in the paths.
        nixl_status_t
        getAvailPlugins (std::vector<nixl_backend_t> &plugins);

        // Returns the supported configs with their default values
        nixl_status_t
        getPluginParams (const nixl_backend_t &type,
                               nixl_mem_list_t &mems,
                               nixl_b_params_t &params) const;

        // Returns the backend parameters after instantiation
        nixl_status_t
        getBackendParams (const nixlBackendH* backend,
                                nixl_mem_list_t &mems,
                                nixl_b_params_t &params) const;

        // Instantiate BackendEngine objects, based on corresponding params
        nixl_status_t
        createBackend (const nixl_backend_t &type,
                       const nixl_b_params_t &params,
                             nixlBackendH* &backend);

        // Register a memory with NIXL. User can provide a list of backends
        // to specify which backends are targeted for a memory, otherwise
        // NIXL will register with all backends that support the memory type.
        nixl_status_t
        registerMem (const nixl_reg_dlist_t &descs,
                     const nixl_opt_args_t* extra_params = nullptr);

        // Deregister a memory list from NIXL
        nixl_status_t
        deregisterMem (const nixl_reg_dlist_t &descs,
                       const nixl_opt_args_t* extra_params = nullptr);

        // Make connection proactively, instead of at transfer time
        nixl_status_t
        makeConnection (const std::string &remote_agent);


        /*** Transfer Request Preparation ***/

        // Prepares a list of descriptors for a transfer request, so later elements
        // from this list can be used to create a transfer request by index. It should
        // be done for descs on both sides of an xfer. There are 3 types of preps:
        // * local descs, on initiator side: remote_agent is set as NIXL_INIT_AGENT
        // * remote descs, on target side: remote_agent is set to the remote name
        // * local descs, on target side: for doing local transfers, remote_agent
        //   is set to agent's own name.
        // User can also provide a list of backends as hints in extra_params to
        // limit preparations to those backends, in order of preference.
        nixl_status_t
        prepXferDescs (const nixl_xfer_dlist_t &descs,
                       const std::string &remote_agent,
                             nixlPreppedH* &prep_hndl,
                       const nixl_opt_args_t* extra_params = nullptr) const;

        // Makes a transfer request `req_handl` by selecting indices from already
        // prepped handles. NIXL automatically determines the backend that can
        // perform the transfer. User can indicate their preference list over backends
        // through extra_params. Notification is optional at this stage, if any.
        nixl_status_t
        makeXferReq (const nixl_xfer_op_t &operation,
                     const nixlPreppedH* local_side,
                     const std::vector<int> &local_indices,
                     const nixlPreppedH* remote_side,
                     const std::vector<int> &remote_indices,
                           nixlXferReqH* &req_hndl,
                     const nixl_opt_args_t* extra_params = nullptr) const;

        // A combined API, where the user wants to create a transfer from two
        // descriptor lists. NIXL will prepare each side and create a transfer
        // handle `req_hndl`. The below set of operations are equivalent:
        // 1. A sequence of prepXferDescs & makeXferReq:
        //  * prepXferDescs(local_desc, NIXL_INIT_AGENT, local_desc_hndl)
        //  * prepXferDescs(remote_desc, "Agent-remote", remote_desc_hndl)
        //  * makeXferReq(NIXL_WRITE, local_desc_hndl, list of all local indices,
        //                remote_desc_hndl, list of all remote_indices, req_hndl)
        // 2. CreateXfer-based one
        //  * createXferReq(NIXL_WRITE, local_desc, remote_desc,
        //                  "Agent-remote", req_hndl)
        // User can also provide a list of backends in extra_params to limit which
        // backends are searched through, in order of preference.
        nixl_status_t
        createXferReq (const nixl_xfer_op_t &operation,
                       const nixl_xfer_dlist_t &local_descs,
                       const nixl_xfer_dlist_t &remote_descs,
                       const std::string &remote_agent,
                             nixlXferReqH* &req_hndl,
                       const nixl_opt_args_t* extra_params = nullptr) const;

        /*** Operations on prepared Transfer Request ***/

        // Submit a transfer request `req_hndl`, which enables async checks on
        // the transfer. Notification message can be preovided through the
        // extra_params, and can be changed per repost.
        nixl_status_t
        postXferReq (      nixlXferReqH* req_hndl,
                     const nixl_opt_args_t* extra_params = nullptr) const;

        // Check the status of transfer request `req_hndl`
        nixl_status_t
        getXferStatus (nixlXferReqH* req_hndl);

        // Query the backend associated with `req_hndl`. E.g., if for genNotif
        // the same backend as a xfer is desired, it can queried by this.
        nixl_status_t
        getXferBackend (const nixlXferReqH* req_hndl,
                              nixlBackendH* &backend) const;

        // Release the transfer request `req_hndl`. If the transfer is active,
        // it will be canceled, or return an error if the transfer cannot be aborted.
        nixl_status_t
        releaseXferReq (nixlXferReqH* req_hndl);

        // Release the preparred transfer descriptor handle `prep_hndl`
        nixl_status_t
        releasePrepped (nixlPreppedH* prep_hndl) const;


        /*** Notification Handling ***/

        // Add entries to the passed received notifications list (can be
        // non-empty). Elements are released within the Agent after this call.
        // Backends can be mentioned in extra_params to only get their notifs.
        nixl_status_t
        getNotifs (      nixl_notifs_t &notif_map,
                   const nixl_opt_args_t* extra_params = nullptr);

        // Generate a notification, not bound to a transfer, e.g., for control.
        // Can be used after the remote metadata is exchanged.
        // Will be received in notif list. Optionally, user can specify
        // which backend to use for the notification.
        nixl_status_t
        genNotif (const std::string &remote_agent,
                  const nixl_blob_t &msg,
                  const nixl_opt_args_t* extra_params = nullptr);

        /*** Metadata handling through side channel ***/

        // Get nixl metadata blob for this agent.
        nixl_status_t
        getLocalMD (nixl_blob_t &str) const;

        // Load other agent's metadata and unpack it internally.
        // Received agent name can be checked through agent_name.
        nixl_status_t
        loadRemoteMD (const nixl_blob_t &remote_metadata,
                            std::string &agent_name);

        // Invalidate the remote agent metadata cached locally, and disconnect from it.
        nixl_status_t
        invalidateRemoteMD (const std::string &remote_agent);
};

#endif
