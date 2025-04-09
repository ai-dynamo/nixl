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
        /** @var  data  The members in agent class wrapped into single nixlAgentData member. */
        nixlAgentData* data;

    public:
        /*** Initialization and Registering Methods ***/

        /**
         * @brief Constructor for nixlAgent which gets agent name and configurations.
         *
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
         * @brief  Discover the available supported plugins found in the plugin paths
         *
         * @param  plugins [out] Vector of available backend plugins
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        getAvailPlugins (std::vector<nixl_backend_t> &plugins);

        /**
         * @brief  Get the supported memory types, and init config parameters and their
         *         default values for a backend plugin.
         *
         * @param  type          Plugin backend type
         * @param  mems [out]    List of supported memory types for nixl by the plugin
         * @param  params [out]  List of init parameters and their values for the plugin
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        getPluginParams (const nixl_backend_t &type,
                         nixl_mem_list_t &mems,
                         nixl_b_params_t &params) const;
        /**
         * @brief  Get the backend parameters after instantiation. This will be a comprehensive
         *         list, for instance the default values used for parameters that were not
         *         specified during the instantiation.
         *
         * @param  backend       Backend type
         * @param  mems [out]    List of supported memory types for nixl by the backend
         * @param  params [out]  List of init parameters and their values for the backend
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        getBackendParams (const nixlBackendH* backend,
                          nixl_mem_list_t &mems,
                          nixl_b_params_t &params) const;

        /**
         * @brief  Instantiate a backend engine object based on the corresponding parameters
         *
         * @param  type          Backend type
         * @param  params        Backend specific parameters
         * @param  backend [out] Backend handle for NIXL
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        createBackend (const nixl_backend_t &type,
                       const nixl_b_params_t &params,
                       nixlBackendH* &backend);
        /**
         * @brief  Register a memory/storage with NIXL. If a list of backends hints is provided
         *         (via extra_params), the registration is limited to the specified backends.
         *
         * @param  descs         Descriptor list of the buffers to be registered
         * @param  extra_params  Optional additional parameters used in registering memory
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        registerMem (const nixl_reg_dlist_t &descs,
                     const nixl_opt_args_t* extra_params = nullptr);

        /**
         * @brief  Deregister a memory/storage from NIXL. If a list of backends hints is provided
         *         (via extra_params), the deregistration is limited to the specified backends.
         *         Each descriptor in the list should match a descriptor used during registration.
         *
         * @param  descs         Descriptor list of the buffers to be deregistered
         * @param  extra_params  Optional additional parameters used in deregistering memory
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        deregisterMem (const nixl_reg_dlist_t &descs,
                       const nixl_opt_args_t* extra_params = nullptr);

        /**
         * @brief  Make connection proactively, instead of at the time of the first transfer
         *         towards the target agent. If a list of backends hints is provided
         *         (via extra_params), the connection is made for the specified backends.
         *
         * @param  remote_agent  Name of the remote agent
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        makeConnection (const std::string &remote_agent,
                        const nixl_opt_args_t* extra_params = nullptr);

        /*** Transfer Request Preparation ***/
        /**
         * @brief  Prepare a list of descriptors for a transfer request, so later elements
         *         from this list can be used to create a transfer request by index. It should
         *         be done on the initiator agent, and for both sides of an transfer.
         *         Considering loopback, there are 3 modes for agent_name:
         *           - For local descriptors, it is set to NIXL_INIT_AGENT,
         *             indicating that this is a local preparation to be used as local_side handle.
         *           - For remote descriptors: it is set to the remote name, indicating
         *             that this is remote side preparation to be used for remote_side handle.
         *           - For loopback descriptors, it is set to local agent's name, indicating that
         *             this is for a loopback (local) transfer to be uued for remote_side handle
         *         If a list of backends hints is provided (via extra_params), the preparation
         *         is limited to the specified backends.
         *
         * @param  agent_name       Agent name as a string for preparing xfer handle
         * @param  descs            The descriptor list to be prepared for transfer requests
         * @param  dlist_hndl [out] The prepared descriptor list handle for this transfer request
         * @param  extra_params     Optional additional parameters used in preparing dlist handle
         * @return nixl_status_t    Error code if call was not successful
         */
        nixl_status_t
        prepXferDlist (const std::string &agent_name,
                       const nixl_xfer_dlist_t &descs,
                       nixlDlistH* &dlist_hndl,
                       const nixl_opt_args_t* extra_params = nullptr) const;
        /**
         * @brief  Make a transfer request `req_handl` by selecting indices from already
         *         prepared descriptor list handles. NIXL automatically determines the backend
         *         that can perform the transfer. If a list of backends hints is provided
         *         (via extra_params), the selection is limited to the specified backends.
         *         Optionally, a notification message can also be provided through extra_params.
         *
         * @param  operation        Operation for transfer (e.g., NIXL_WRITE)
         * @param  local_side       Local prepared descriptor list handle
         * @param  local_indices    Indices list to the local prepared descriptor list handle
         * @param  remote_side      Remote (or loopback) prepared descriptor list handle
         * @param  remote_indices   Indices list to the remote prepared descriptor list handle
         * @param  req_handle [out] Transfer request handle output
         * @param  extra_params     Optional additional parameters used in making a transfer request
         * @return nixl_status_t    Error code if call was not successful
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
         * @brief  A combined API, to create a transfer request from two descriptor lists.
         *         NIXL will prepare each side and create a transfer handle `req_hndl`.
         *         The below set of operations are equivalent:
         *           1. A sequence of prepXferDlist & makeXferReq:
         *              prepXferDlist(NIXL_INIT_AGENT, local_desc, local_desc_hndl)
         *              prepXferDlist("Agent-remote/self", remote_desc, remote_desc_hndl)
         *              makeXferReq(NIXL_WRITE, local_desc_hndl, list of all local indices,
         *                          remote_desc_hndl, list of all remote_indices, req_hndl)
         *           2. A CreateXfer:
         *              createXferReq(NIXL_WRITE, local_desc, remote_desc,
         *                            "Agent-remote/self", req_hndl)
         *
         *         If there are common descriptors across different transfer requests, using
         *         createXfer will result in repeated computation, such as validity checks and
         *         pre-processing done in the preparation step. If a list of backends hints is
         *         provided (via extra_params), the selection is limited to the specified backends.
         *         Optionally, a notification message can also be provided through extra_params.
         *
         * @param  operation      Operation for transfer (e.g., NIXL_WRITE)
         * @param  local_descs    Local descriptor list
         * @param  remote_descs   Remote (or loopback) descriptor list
         * @param  remote_agent   Remote (or self) agent name for accessing the remote (local) data
         * @param  req_hndl [out] Transfer request handle output
         * @param  extra_params   Optional extra parameters used in creating a transfer request
         * @return nixl_status_t  Error code if call was not successful
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
         * @brief  Submit a transfer request `req_hndl` which initiates a transfer.
         *         After this, the transfer state can be checked asynchronously till completion.
         *         In case of small transfers that are completed within the call, return value
         *         will be NIXL_SUCCESS. Otherwise, the output status will be NIXL_IN_PROG until
         *         completion. Notification  message  can be preovided through the extra_params,
         *         and can be updated per re-post.
         *
         * @param  req_hndl      Transfer request handle obtained from makeXferReq/createXferReq
         * @param  extra_params  Optional extra parameters used in posting a transfer request
         * @return nixl_status_t NIXL_IN_PROG or error code if call was not successful
         */
        nixl_status_t
        postXferReq (nixlXferReqH* req_hndl,
                     const nixl_opt_args_t* extra_params = nullptr) const;

        /**
         * @brief  Check the status of transfer request `req_hndl`
         *
         * @param  req_hndl      Transfer request handle after postXferReq
         * @return nixl_status_t NIXL_IN_PROG or error code if call was not successful
         */
        nixl_status_t
        getXferStatus (nixlXferReqH* req_hndl);

        /**
         * @brief  Query the backend associated with `req_hndl`. E.g., if for genNotif
         *         the same backend as a transfer is desired.
         *
         * @param  req_hndl      Transfer request handle obtained from makeXferReq/createXferReq
         * @param  backend [out] Output backend handle chosen for the transfer request
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        queryXferBackend (const nixlXferReqH* req_hndl,
                          nixlBackendH* &backend) const;

        /**
         * @brief  Release the transfer request `req_hndl`. If the transfer is active,
         *         it will be canceled, or return an error if the transfer cannot be aborted.
         *
         * @param  req_hndl      Transfer request handle to be released
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        releaseXferReq (nixlXferReqH* req_hndl);

        /**
         * @brief  Release the prepared descriptor list handle `dlist_hndl`
         *
         * @param  dlist_hndl    Prepared descriptor list handle to be released
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        releasedDlistH (nixlDlistH* dlist_hndl) const;


        /*** Notification Handling ***/

        /**
         * @brief  Add entries to the input notifications list (can be non-empty), which is a map
         *         from agent name to a list of notification received from that agent. Elements
         *         are released within the agent after this call. Optionally, a list of backends
         *         can be mentioned in extra_params to only get those backends notifications.
         *
         * @param  notif_map     Input notifications list
         * @param  extra_params  Optional extra parameters used in getting notifications
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        getNotifs (nixl_notifs_t &notif_map,
                   const nixl_opt_args_t* extra_params = nullptr);

        /**
         * @brief  Generate a notification, not bound to a transfer, e.g., for control.
         *         Metadata of remote agent should be available before this call. The
         *         generated notification will be received alongside other notifications
         *         by getNotifs. Optionally, a backend can be specified for the notification
         *         through the extra_params.
         *
         * @param  remote_agent  Remote agent name as string
         * @param  msg           Notification message to be sent
         * @param  extra_params  Optional extra parameters used in generating a standalone notif
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        genNotif (const std::string &remote_agent,
                  const nixl_blob_t &msg,
                  const nixl_opt_args_t* extra_params = nullptr);

        /*** Metadata handling through side channel ***/
        /**
         * @brief  Get metadata blob for this agent, to be given to other agents.
         *
         * @param  str [out]     The serialized metadata blob
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        getLocalMD (nixl_blob_t &str) const;

        /**
         * @brief  Load other agent's metadata and unpack it internally. Now the local
         *         agent can initiate transfers towards the remote agent.
         *
         * @param  remote_metadata  Serialized metadata blob to be loaded
         * @param  agent_name [out] Agent name extracted from the loaded metadata blob
         * @return nixl_status_t    Error code if call was not successful
         */
        nixl_status_t
        loadRemoteMD (const nixl_blob_t &remote_metadata,
                      std::string &agent_name);

        /**
         * @brief  Invalidate the remote agent metadata cached locally. This will
         *         disconnect from that agent if already connected, and no more
         *         transfers can be initiated towards that agent.
         *
         * @param  remote_agent  Remote agent name to invalidate its metadata blob
         * @return nixl_status_t Error code if call was not successful
         */
        nixl_status_t
        invalidateRemoteMD (const std::string &remote_agent);
};

#endif
