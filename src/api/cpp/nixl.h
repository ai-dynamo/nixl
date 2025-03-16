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

        // Register a memory with NIXL. If a list of backends hints is provided
        // (via extra_params), the registration is limited to the specified backends.
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
        // be done for descriptors on the initiator agent, and for both sides of an
        // transfer. Considering loopback, there are 3 modes for remote_agent naming:
        //
        // * For local descriptors, remote_agent must be set NIXL_INIT_AGENT
        //   to indicate this is local preparation to be used as local_side handle.
        // * For remote descriptors: the remote_agent is set to the remote name to
        //   indicate this is remote side preparation to be used for remote_side handle.
        // * remote_agent can be set to local agent name for local (loopback) transfers.
        //
        // If a list of backends hints is provided (via extra_params), the preparation
        // is limited to the specified backends, in the order of preference.
        nixl_status_t
        prepXferDescs (const nixl_xfer_dlist_t &descs,
                       const std::string &remote_agent,
                       nixlDlistH* &dlist_hndl,
                       const nixl_opt_args_t* extra_params = nullptr) const;

        // Makes a transfer request `req_handl` by selecting indices from already
        // populated handles. NIXL automatically determines the backend that can
        // perform the transfer. Preference over the backends can be provided via
        // extra_params. Optionally, a notification message can also be provided.
        nixl_status_t
        makeXferReq (const nixl_xfer_op_t &operation,
                     const nixlDlistH* local_side,
                     const std::vector<int> &local_indices,
                     const nixlDlistH* remote_side,
                     const std::vector<int> &remote_indices,
                     nixlXferReqH* &req_hndl,
                     const nixl_opt_args_t* extra_params = nullptr) const;

        // A combined API, to create a transfer from two  descriptor lists.
        // NIXL will prepare each side and create a transfer handle `req_hndl`.
        // The below set of operations are equivalent:
        // 1. A sequence of prepXferDescs & makeXferReq:
        //  * prepXferDescs(local_desc, NIXL_INIT_AGENT, local_desc_hndl)
        //  * prepXferDescs(remote_desc, "Agent-remote/self", remote_desc_hndl)
        //  * makeXferReq(NIXL_WRITE, local_desc_hndl, list of all local indices,
        //                remote_desc_hndl, list of all remote_indices, req_hndl)
        // 2. A CreateXfer:
        //  * createXferReq(NIXL_WRITE, local_desc, remote_desc,
        //                  "Agent-remote/self", req_hndl)
        // Optionally, a list of backends in extra_params can be used to define a
        // subset of backends to be searched through, in the order of preference.
        nixl_status_t
        createXferReq (const nixl_xfer_op_t &operation,
                       const nixl_xfer_dlist_t &local_descs,
                       const nixl_xfer_dlist_t &remote_descs,
                       const std::string &remote_agent,
                       nixlXferReqH* &req_hndl,
                       const nixl_opt_args_t* extra_params = nullptr) const;

        /*** Operations on prepared Transfer Request ***/

        // Submits a transfer request `req_hndl` which initiates a transfer.
        // After this, the transfer state can be checked asynchronously till
        // completion. The output status will be NIXL_IN_PROG, or NIXL_SUCCESS
        // for small transfer that are completed within the call.
        // Notification  message  can be preovided through the extra_params,
        // and can be updated per re-post.
        nixl_status_t
        postXferReq (nixlXferReqH* req_hndl,
                     const nixl_opt_args_t* extra_params = nullptr) const;

        // Check the status of transfer request `req_hndl`
        nixl_status_t
        getXferStatus (nixlXferReqH* req_hndl);

        // Query the backend associated with `req_hndl`. E.g., if for genNotif
        // the same backend as a transfer is desired, it can queried by this.
        nixl_status_t
        queryXferBackend (const nixlXferReqH* req_hndl,
                          nixlBackendH* &backend) const;

        // Release the transfer request `req_hndl`. If the transfer is active,
        // it will be canceled, or return an error if the transfer cannot be aborted.
        nixl_status_t
        releaseXferReq (nixlXferReqH* req_hndl);

        // Release the preparred transfer descriptor handle `dlist_hndl`
        nixl_status_t
        releasePrepped (nixlDlistH* dlist_hndl) const;


        /*** Notification Handling ***/

        // Add entries to the passed received notifications list (can be
        // non-empty). Elements are released within the Agent after this call.
        // Backends can be mentioned in extra_params to only get their notifs.
        nixl_status_t
        getNotifs (nixl_notifs_t &notif_map,
                   const nixl_opt_args_t* extra_params = nullptr);

        // Generate a notification, not bound to a transfer, e.g., for control.
        // Can be used after the remote metadata is exchanged.
        // Will be received in notif list. A backend can be specified for the
        // notification through the extra_params.
        nixl_status_t
        genNotif (const std::string &remote_agent,
                  const nixl_blob_t &msg,
                  const nixl_opt_args_t* extra_params = nullptr);

        /*** Metadata handling through side channel ***/

        // Get nixl metadata blob for this agent. By loading this blob on a
        // remote agent (through a separate side channel transfer), that agent
        // can initiate transfers to this agent.
        nixl_status_t
        getLocalMD (nixl_blob_t &str) const;

        // Load other agent's metadata and unpack it internally.
        // Received agent name can be checked through agent_name.
        nixl_status_t
        loadRemoteMD (const nixl_blob_t &remote_metadata,
                      std::string &agent_name);

        // Invalidate the remote agent metadata cached locally, so transfers cannot
        // be initiated towards it. Also it will disconnect from that agent.
        nixl_status_t
        invalidateRemoteMD (const std::string &remote_agent);
};

#endif
