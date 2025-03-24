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
/**
 * @file nixl.h (NVIDIA Inference Xfer Library)
 * @brief These are NIXL Core APIs for applications
 */
#ifndef _NIXL_H
#define _NIXL_H

#include "nixl_types.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"

/**
 * @class nixlAgent
 * @brief nixlAgent forms the main transfer object class
 */
class nixlAgent {
    private:
        /** @var  data The stored private agent metadata for nixlAgent */
        nixlAgentData* data;

    public:
        /*** Initialization and Registering Methods ***/
        /**
         * @brief Constructor for nixlAgent which populates agent name and device metadata
         * @param name A String name assigned to the Agent to initialize the class
         * @param cfg  Agent configuration of class type nixlAgentConfig
         */
        nixlAgent (const std::string &name,
                   const nixlAgentConfig &cfg);
        /**
         * @brief Destructor for nixlAgent object
         */
        ~nixlAgent ();

        /**
         * @brief  Discover the available plugins supported found in the paths
         * @param  plugins  Vector of plugins returned to the user
         * @return nixl_status_t value is returned after the call
         */
        nixl_status_t
        getAvailPlugins (std::vector<nixl_backend_t> &plugins);

        /**
         * @brief This provides the supported configs with their default values
         * @param type   Provides the backend type
         * @param mems   List of memory types for nixl
         * @param params Plugin backend specific parameters for nixl Backend
         * @return nixl_status_t    Value is returned after the call
         */
        nixl_status_t
        getPluginParams (const nixl_backend_t &type,
                         nixl_mem_list_t &mems,
                         nixl_b_params_t &params) const;
        /**
         * @brief Returns the backend parameters after instantiation
         * @param backend   Provides the backend type
         * @param mems      List of memory types for nixl
         * @param params    Plugin backend specific parameters for nixl Backend
         * @return nixl_status_t    Value is returned after the call
         */
        nixl_status_t
        getBackendParams (const nixlBackendH* backend,
                          nixl_mem_list_t &mems,
                          nixl_b_params_t &params) const;

        /**
         * @brief Instantiates BackendEngine objects based on the corresponding parameters
         * @param type         Provides th backend type
         * @param params       Plugin backend specific parameters for nixl Backend
         * @param backend      Backend handle for NIXL
         * @return nixl_status_t    Value is returned after the call
         */
        nixl_status_t
        createBackend (const nixl_backend_t &type,
                       const nixl_b_params_t &params,
                       nixlBackendH* &backend);
        /**
         * @brief Register a memory with NIXL. If a list of backends hints is provided
         *        (via extra_params), the registration is limited to the specified backends.
         * @param descs        descriptor list for registering buffers
         * @param extra_params Optional additional parameters required for registering memory
         * @return nixl_status_t    Value is returned after the call
         */
        nixl_status_t
        registerMem (const nixl_reg_dlist_t &descs,
                     const nixl_opt_args_t* extra_params = nullptr);

        /**
         * @brief Deregister a memory list from NIXL. If a list of backends hints is provided
         *        (via extra_params), the registration is limited to the specified backends.
         * @param descs        descriptor list for registering buffers
         * @param extra_params Optional additional parameters required for registering memory
         * @return nixl_status_t    Value is returned after the call
         */
        nixl_status_t
        deregisterMem (const nixl_reg_dlist_t &descs,
                       const nixl_opt_args_t* extra_params = nullptr);

        /**
         * @brief Make connection proactively, instead of at transfer time
         * @param remote_agent  remote_agent
         * @return nixl_status_t value is returned after the call
         */
        nixl_status_t
        makeConnection (const std::string &remote_agent);


        /*** Transfer Request Preparation ***/
        /**
         * @brief Prepares a list of descriptors for a transfer request, so later elements
         *        from this list can be used to create a transfer request by index. It should
         *        be done for descriptors on the initiator agent, and for both sides of an
         *        transfer. Considering loopback, there are 3 modes for remote_agent naming:
         *             - For local descriptors, remote_agent must be set NIXL_INIT_AGENT
         *               to indicate this is local preparation to be used as local_side handle.
         *             - For remote descriptors: the remote_agent is set to the remote name to
         *               indicate this is remote side preparation to be used for remote_side handle.
         *             - remote_agent can be set to local agent name for local (loopback) transfers.
         *        If a list of backends hints is provided (via extra_params), the preparation
         *        is limited to the specified backends, in the order of preference.
         *        empty string for remote_agent means it's local side.
         * @param remote_agent   remote agent name as a string for preparing xfer handle
         * @param descs          provide the desc list for transfers requested
         * @param dlist_hndl     The descriptor list handle for this transfer request.
         * @param extra_params   Optional  parameters required for registering memory
         * @return nixl_status_t Value is returned after the call
         */
        nixl_status_t
        prepXferDlist (const std::string &remote_agent,
                       const nixl_xfer_dlist_t &descs,
                       nixlDlistH* &dlist_hndl,
                       const nixl_opt_args_t* extra_params = nullptr) const;
        /**
         * @brief Makes a transfer request `req_handl` by selecting indices from already
         *        populated handles. NIXL automatically determines the backend that can
         *        perform the transfer. Preference over the backends can be provided via
         *        extra_params. Optionally, a notification message can also be provided.
         * @param operation         Operation selection for NIXL Transfer operations
         * @param local_side        Local Descriptor list handle for create transfer request
         * @param local_indices     Local indices list for creating a transfer request
         * @param remote_side       Remote Descriptor list handle for creating transfer request
         * @param remote_indices    List of indices for the remote side
         * @param notif_msg         notification message as nixl blob
         * @param req_handle        request handle created for NIXL
         * @param extra_params      Optional extra parameters required for registering memory
         * @return nixl_status_t    Value is returned after the call
         */
        nixl_status_t
        makeXferReq (const nixl_xfer_op_t &operation,
                     const nixlDlistH* local_side,
                     const std::vector<int> &local_indices,
                     const nixlDlistH* remote_side,
                     const std::vector<int> &remote_indices,
                     nixlXferReqH* &req_hndl,
                     const nixl_opt_args_t* extra_params = nullptr) const;
        /**
         * @brief A combined API, to create a transfer from two  descriptor lists.
         *        NIXL will prepare each side and create a transfer handle `req_hndl`.
         *        The below set of operations are equivalent:
         *            1. A sequence of prepXferDlist & makeXferReq:
         * 		 prepXferDlist(local_desc, NIXL_INIT_AGENT, local_desc_hndl)
         * 		 prepXferDlist(remote_desc, "Agent-remote/self", remote_desc_hndl)
         *		 makeXferReq(NIXL_WRITE, local_desc_hndl, list of all local indices,
         *		 remote_desc_hndl, list of all remote_indices, req_hndl)
         *	      2. A CreateXfer:
         * 		 createXferReq(NIXL_WRITE, local_desc, remote_desc,
         *                             "Agent-remote/self", req_hndl)
         *
         *           For the createXfer case, if the user is reusing transfer entries there is repeated computation
         *           for validity checks and pre-processing, so makeXferReq is better in this case.
         * Optionally, a list of backends in extra_params can be used to define a
         * subset of backends to be searched through, in the order of preference.
         * @param  operation           Operation selection for NIXL Transfer operations
         * @param  local_descs        Provide the local descriptor list for creating transfer
         * @param  remote_descs       Provide the remote descriptor list for creating transfer
         * @param  remote_agent       Remote agent name for accessing the remote data
         * @param  req_hndl [out]     Pointer to the transfer request handle
         * @param  extra_params       Optional extra parameters required for registering memory
         * @return nixl_status_t      Value is returned after the call
         */
        nixl_status_t
        createXferReq (const nixl_xfer_op_t &operation,
                       const nixl_xfer_dlist_t &local_descs,
                       const nixl_xfer_dlist_t &remote_descs,
                       const std::string &remote_agent,
                       nixlXferReqH* &req_hndl,
                       const nixl_opt_args_t* extra_params = nullptr) const;

        /*** Operations on prepared Transfer Request ***/

        /**
         * @brief Submits a transfer request `req_hndl` which initiates a transfer.
         *        After this, the transfer state can be checked asynchronously till completion.
         *        The output status will be NIXL_IN_PROG, or NIXL_SUCCESS for small transfer
         *        that are completed within the call. Notification  message  can be preovided
         *        through the extra_params, and can be updated per re-post.
         * @param  req		 Provide the transfer request handle obtained from create
         * @param  extra_params  Optional extra parameters required for posting Xfer request
         * @return nixl_status_t Status value is returned after the call
         */
        nixl_status_t
        postXferReq (nixlXferReqH* req_hndl,
                     const nixl_opt_args_t* extra_params = nullptr) const;

        /**
         * @brief Check the status of transfer request `req_hndl`
         * @param  req_hndl      Provide the transfer request handle after postXferReq
         * @return nixl_status_t Status Value is returned after the call
         */
        nixl_status_t
        getXferStatus (nixlXferReqH* req_hndl);

        /**
         * @brief  Query the backend associated with `req_hndl`. E.g., if for genNotif
         *         the same backend as a transfer is desired, it can queried by this.
         * @param  req_hndl      Provide the transfer request handle after postXferReq
         * @param  backend       Backend handle for getNotif (same backend as transfer)
         * @return nixl_status_t Status Value is returned after the call
         */
        nixl_status_t
        queryXferBackend (const nixlXferReqH* req_hndl,
                          nixlBackendH* &backend) const;

        /**
         * @brief  Release the transfer request `req_hndl`. If the transfer is active,
         *         it will be canceled, or return an error if the transfer cannot be aborted.
         * @param  req_hndl      Provide the transfer request handle after postXferReq
         * @return nixl_status_t Status Value is returned after the call
         */
        nixl_status_t
        releaseXferReq (nixlXferReqH* req_hndl);

        /**
         * @brief  Release the prepared transfer descriptor handle `dlist_hndl`
         * @param  dlist_hndl    Provide the dlist_handle to be released
         * @return nixl_status_t Status Value is returned after the call
         */
        nixl_status_t
        releasedDlistH (nixlDlistH* dlist_hndl) const;


        /*** Notification Handling ***/
        /**
         * @brief  Add entries to the passed received notifications list
         *         (can be non-empty). Elements are released within the
         *         Agent after this call. Backends can be mentioned in
         *         extra_params to only get their notifs.
         * @param  notif_map     Pass received notifications list
         * @param  extra_params  Optional extra parameters required for registering memory
         * @return nixl_status_t Status Value is returned after the call
         */
        nixl_status_t
        getNotifs (nixl_notifs_t &notif_map,
                   const nixl_opt_args_t* extra_params = nullptr);

        /**
         * @brief  Generate a notification, not bound to a transfer, e.g., for control.
         *         Can be used after the remote metadata is exchanged. Will be received
         *         in notif list. A backend can be specified for the notification through
         *         the extra_params.
         * @param  remote_agent  Remote agent name as string
         * @param  msg           Notification message returned
         * @param  extra_params  Optional extra parameters required for registering memory
         * @return nixl_status_t Status Value is returned after the call
         */
        nixl_status_t
        genNotif (const std::string &remote_agent,
                  const nixl_blob_t &msg,
                  const nixl_opt_args_t* extra_params = nullptr);

        /*** Metadata handling through side channel ***/
        /**
         * @brief Get nixl_metadata for this agent. Empty string means error.
         *        The std::string used for serialized MD can have \0 values.
         * @param str  Get the serialized metadata blob
         * @return nixl_status_t value is returned after the call
         */
        nixl_status_t
        getLocalMD (nixl_blob_t &str) const;

        /**
         * @brief Load other agent's metadata and unpack it internally.
         *        Returns the found agent name in metadata, or "" in case of error.
         * @param remote_metadata  Serialized metadata blob to be loaded
         * @param agent_name       Agent name in metadata
         * @return nixl_status_t value is returned after the call
         */
        nixl_status_t
        loadRemoteMD (const nixl_blob_t &remote_metadata,
                      std::string &agent_name);

        /**
         * @brief Invalidate the remote section information cached locally
         * @param remote_agent  Agent name for which metadata to invalidate
         * @return nixl_status_t value is returned after the call
         */
        nixl_status_t
        invalidateRemoteMD (const std::string &remote_agent);
};

#endif
