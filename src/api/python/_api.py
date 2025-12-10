# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
from typing import Optional, Union

import numpy as np
import torch

from . import _bindings as nixlBind  # type: ignore
from .logging import get_logger

# Get logger using centralized configuration
logger = get_logger(__name__)

DEFAULT_COMM_PORT = nixlBind.DEFAULT_COMM_PORT


"""
@brief Opaque handle wrapper for a prepared transfer descriptor list.
       Use release() to explicitly free resources; __del__ performs best-effort cleanup.
@param agent Owning nixl_agent used to perform release operations.
@param value Internal handle
"""


class nixl_prepped_dlist_handle:
    __slots__ = ("_handle", "_agent", "_released")

    def __init__(self, agent, value: int):
        self._handle = int(value)
        self._agent = agent
        self._released = False

    def __repr__(self) -> str:
        return (
            f"nixl_prepped_dlist_handle(0x{self._handle:x}, released={self._released})"
        )

    def release(self):
        if not self._released:
            self._agent.releasedDlistH(self._handle)
            self._released = True

    def __del__(self):
        if not self._released:
            try:
                self._agent.releasedDlistH(self._handle)
            except Exception:
                try:
                    logger.error(
                        "nixl_prepped_dlist_handle finalization failed for 0x%x",
                        self._handle,
                    )
                except Exception:
                    pass


"""
@brief Opaque handle wrapper for a transfer request.
       Use release() to explicitly free resources. If transfer was not complete, this will initiate
       the abort process (if available) and will raise an exception.
       __del__ calls release() and if it fails, it logs the failure and defers release by queuing
       the handle in leaked xfer handles list, which will be re-released during agent destruction
@param agent Owning nixl_agent used to perform release operations.
@param value Internal handle
"""


class nixl_xfer_handle:
    __slots__ = ("_handle", "_agent", "_released")

    def __init__(self, agent, value: int):
        self._handle = int(value)
        self._agent = agent
        self._released = False

    def __repr__(self) -> str:
        return f"nixl_xfer_handle(0x{self._handle:x}, released={self._released})"

    def release(self):
        if not self._released:
            self._agent.releaseXferReq(self._handle)
            self._released = True

    def __del__(self):
        if not self._released:
            try:
                self._agent.releaseXferReq(self._handle)
            except Exception:
                try:
                    logger.error(
                        "nixl_xfer_handle finalization failed for 0x%x; keeping handle alive in agent leak list",
                        self._handle,
                    )
                except Exception:
                    pass
                try:
                    self._agent._leaked_xfer_handles.append(self._handle)
                except Exception:
                    pass
                return


# Opaque handle for backend can be just int, as it's not passed to the user
nixl_backend_handle = int


"""
@brief Configuration class for NIXL agent.

@param enable_prog_thread Whether to enable the progress thread, if available.
@param enable_listen_thread Whether to enable the listener thread for metadata communication.
@param listen_port Specify the port for the listener thread to listen on.
@param capture_telemetry Whether to enable telemetry capture.
@param num_threads Specify number of threads for the supported multi-threaded backends.
@param backends List of backend names for agent to initialize.
        Default is UCX, other backends can be added to the list, or after
        agent creation, can be initialized with create_backend.
"""


class nixl_agent_config:
    def __init__(
        self,
        enable_prog_thread: bool = True,
        enable_listen_thread: bool = False,
        listen_port: int = 0,
        capture_telemetry: bool = False,
        num_threads: int = 0,
        backends: list[str] = ["UCX"],
    ):
        # TODO: add backend init parameters
        self.backends = backends
        self.enable_pthread = enable_prog_thread
        self.enable_listen = enable_listen_thread
        self.port = listen_port
        self.capture_telemetry = capture_telemetry
        self.num_threads = num_threads


"""
@brief Main class for creating a NIXL agent and performing transfers.
        This class provides methods for initializing backends, creating descriptor lists,
        registering memory, performing data transfers, and destroying NIXL objects.

@param agent_name Name of the agent, should be unique for clarity.
@param nixl_conf Optional configuration for the agent, described in nixl_agent_config.
@param instantiate_all Whether to instantiate all available backend plugins.
"""


class nixl_agent:
    def __init__(
        self,
        agent_name: str,
        nixl_conf: Optional[nixl_agent_config] = None,
        instantiate_all: bool = False,
    ):
        if nixl_conf and instantiate_all:
            instantiate_all = False
            logger.warning(
                "Ignoring instantiate_all based on the provided config in agent creation."
            )
        if not nixl_conf:
            nixl_conf = nixl_agent_config()  # Using defaults set in nixl_agent_config

        thread_config = (
            nixlBind.NIXL_THREAD_SYNC_STRICT
            if nixl_conf.enable_listen
            else nixlBind.NIXL_THREAD_SYNC_NONE
        )

        # Set agent config and instantiate an agent
        agent_config = nixlBind.nixlAgentConfig(
            nixl_conf.enable_pthread,
            nixl_conf.enable_listen,
            nixl_conf.port,
            thread_config,
            1,
            0,
            100000,
            nixl_conf.capture_telemetry,
        )
        self.agent = nixlBind.nixlAgent(agent_name, agent_config)

        self.name = agent_name
        self._leaked_xfer_handles: list[int] = []
        self.notifs: dict[str, list[bytes]] = {}
        self.backends: dict[str, nixl_backend_handle] = {}
        self.backend_mems: dict[str, list[str]] = {}
        self.backend_options: dict[str, dict[str, str]] = {}

        self.plugin_list = self.agent.getAvailPlugins()
        if len(self.plugin_list) == 0:
            logger.error("No plugins available, cannot start transfers!")
            raise RuntimeError("No plugins available for NIXL, cannot start transfers!")

        self.plugin_b_options: dict[str, dict[str, str]] = {}
        self.plugin_mem_types: dict[str, list[str]] = {}
        for plugin in self.plugin_list:
            (backend_options, mem_types) = self.agent.getPluginParams(plugin)
            self.plugin_b_options[plugin] = backend_options
            self.plugin_mem_types[plugin] = mem_types

        if instantiate_all:
            nixl_conf.backends = self.plugin_list

        for bknd in nixl_conf.backends:
            if bknd not in self.plugin_list:
                logger.warning(
                    "Skipping backend registration %s due to the missing plugin.",
                    bknd,
                )
            else:
                # TODO: improve population of init from nixl_conf
                init: dict[str, str] = {}
                if nixl_conf.num_threads > 0:
                    if bknd == "UCX" or bknd == "OBJ":
                        init["num_threads"] = str(nixl_conf.num_threads)
                    elif bknd == "GDS_MT":
                        init["thread_count"] = str(nixl_conf.num_threads)
                    elif bknd == "UCCL":
                        init["num_cpus"] = str(nixl_conf.num_threads)
                self.create_backend(bknd, init)

        self.nixl_mems = {
            "DRAM": nixlBind.DRAM_SEG,
            "VRAM": nixlBind.VRAM_SEG,
            "FILE": nixlBind.FILE_SEG,
            "BLOCK": nixlBind.BLK_SEG,
            "OBJ": nixlBind.OBJ_SEG,
            "cpu": nixlBind.DRAM_SEG,
            "cuda": nixlBind.VRAM_SEG,
        }
        self.nixl_ops = {
            "WRITE": nixlBind.NIXL_WRITE,
            "READ": nixlBind.NIXL_READ,
        }

        logger.info("Initialized NIXL agent: %s", agent_name)

    def __del__(self):
        # Best-effort cleanup of any leaked xfer handles belonging to this agent
        if getattr(self, "_leaked_xfer_handles", None):
            for h in list(self._leaked_xfer_handles):
                try:
                    self.releaseXferReq(h)
                except Exception as e:
                    try:
                        logger.error(
                            "Failed to finalize leaked nixl_xfer_handle 0x%x: %s", h, e
                        )
                    except Exception:
                        pass
            self._leaked_xfer_handles.clear()

    """
    @brief Get the list of available plugins.

    @return List of plugin names.
    """

    def get_plugin_list(self) -> list[str]:
        return self.plugin_list

    """
    @brief Get the memory types supported by a plugin.

    @param backend Name of the plugin.
    @return List of supported memory types.
    """

    def get_plugin_mem_types(self, backend: str) -> list[str]:
        if backend in self.plugin_mem_types:
            return self.plugin_mem_types[backend]
        else:
            logger.warning(
                "Plugin %s is not available to get its supported mem types.", backend
            )
            return []

    """
    @brief Get the initialization parameters of a plugin.
           This is a dictionary of strings (option name) to strings (default value for that option).

    @param backend Name of the plugin to get params for.
    @return Dictionary of plugin parameters, described above.
    """

    def get_plugin_params(self, backend: str) -> dict[str, str]:
        if backend in self.plugin_b_options:
            return self.plugin_b_options[backend]
        else:
            logger.warning("Plugin %s is not available to get its parameters.", backend)
            return {}

    """
    @brief  Get the memory types supported by a backend.
            Here, a backend means an initialized plugin.
            After a plugin is initialized, the supported memory types might have changed.
            This function is for getting a refreshed list of those memory types.

    @param backend Name of the backend.
    @return List of supported memory types.
    """

    def get_backend_mem_types(self, backend: str) -> list[str]:
        if backend in self.backend_mems:
            return self.backend_mems[backend]
        else:
            logger.warning(
                "Backend %s not instantiated to get its supported mem types.", backend
            )
            return []

    """
    @brief  Get the parameters of a backend.
            Here, a backend means an initialized plugin.
            Available initialization parameters (described above) might have changed after initialization.
            This function is for getting a refreshed list of those parameters.

    @param backend Name of the backend.
    @return Dictionary of backend parameters, described in get_plugin_params.
    """

    def get_backend_params(self, backend: str) -> dict[str, str]:
        if backend in self.backend_options:
            return self.backend_options[backend]
        else:
            logger.warning(
                "Backend %s not instantiated to get its parameters.", backend
            )
            return {}

    """
    @brief  Initialize a backend with the specified initialization parameters, described above.

    @param backend Name of the backend.
    @param initParams Dictionary of initialization parameters.
    """

    def create_backend(self, backend: str, initParams: dict[str, str] = {}):
        self.backends[backend] = self.agent.createBackend(backend, initParams)

        (backend_options, mem_types) = self.agent.getBackendParams(
            self.backends[backend]
        )
        self.backend_mems[backend] = mem_types
        self.backend_options[backend] = backend_options
        logger.info("Backend %s was instantiated", backend)

    """
    @brief Register memory regions, optionally with specified backends.

    @param reg_list List of either memory regions, tensors, or nixlRegDList to register.
    @param mem_type Optional memory type, necessary if specifying a list of memory regions.
    @param backends Optional list of backend names for registration, otherwise NIXL will try to
            register with all backends that support this memory type.
    @return nixlRegDList for the registered memory, can be used with deregister_memory.
    """

    def register_memory(
        self,
        reg_list,
        mem_type: Optional[str] = None,
        backends: list[str] = [],
    ) -> nixlBind.nixlRegDList:
        reg_descs = self.get_reg_descs(reg_list, mem_type)

        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.registerMem(reg_descs, handle_list)

        return reg_descs

    """
    @brief Deregister memory regions from the specified backends.

    @param dereg_list nixlRegDList of memory to deregister, received from register_memory or get_reg_descs.
    @param backends Optional list of backend names for deregistration, otherwise NIXL will deregister
            with all the backends that have these memory regions registered.
    """

    def deregister_memory(
        self, dereg_list: nixlBind.nixlRegDList, backends: list[str] = []
    ):
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.deregisterMem(dereg_list, handle_list)

    """
    @brief Query information about memory/storage for a specific backend.

    @param reg_list List of either memory regions, tensors, or nixlRegDList to query.
    @param backend Backend name for querying.
    @param mem_type Optional memory type, necessary if specifying a list of memory regions.
    @return List of query results where each item is either None if not found, or a dictionary with the info
    """

    def query_memory(
        self, reg_list, backend: str, mem_type: Optional[str] = None
    ) -> list[Optional[dict[str, str]]]:
        reg_descs = self.get_reg_descs(reg_list, mem_type)

        # Get the backend handle
        if backend not in self.backends:
            raise ValueError(
                f"Backend '{backend}' not found. Available backends: {list(self.backends.keys())}"
            )

        return self.agent.queryMem(reg_descs, self.backends[backend])

    """
    @brief  Proactively establish a connection with a remote agent,
            which will reduce the time spent in the first transfer between the two agents.
            NIXL will establish the connection for all the backends that talk to that remote
            agent, or limit to the set of backends passed through the backends argument.
            This function is optional.

    @param backends Optional list of backend names to limit the connections to specific backends
    @param remote_agent Name of the remote agent.
    """

    def make_connection(self, remote_agent: str, backends: list[str] = []):
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        self.agent.makeConnection(remote_agent, handle_list)

    """
    @brief  Prepare a transfer descriptor list for data transfer.
            Later, elements from this list can be used to create a transfer request by index.
            It should be done on the initiator agent, and for both sides of a transfer.
            Considering loopback, there are 3 modes for agent_name:
              - For local descriptors, it is set to NIXL_INIT_AGENT,
                indicating that this is a local preparation to be used as local_side handle.
              - For remote descriptors, it is set to the remote name, indicating
                that this is remote side preparation to be used for remote_side handle.
              - For loopback descriptors, it is set to local agent's name, indicating that
                this is for a loopback (local) transfer to be used for remote_side handle
            Preparation succeeds if there exists at least one backend that can handle all
            elements in the descriptor list.

    @param agent_name Name of the agent. It can be "NIXL_INIT_AGENT", local agent name, or remote agent name
    @param xfer_list List of transfer descriptors, can be list of memory region tuples, tensors,
                     Nx3 numpy array, or nixlXferDList. See get_xfer_descs for more details on the structure.
    @param mem_type Optional memory type necessary for list of memory regions.
    @param backends Optional list of backend names to limit which backends are used during preparation
    @return Opaque handle to the prepared transfer descriptor list.
    """

    def prep_xfer_dlist(
        self,
        agent_name: str,
        xfer_list,
        mem_type: Optional[str] = None,
        backends: list[str] = [],
    ) -> nixl_prepped_dlist_handle:
        descs = self.get_xfer_descs(xfer_list, mem_type)

        if agent_name == "NIXL_INIT_AGENT" or agent_name == "":
            agent_name = nixlBind.NIXL_INIT_AGENT

        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        handle = self.agent.prepXferDlist(agent_name, descs, handle_list)
        return nixl_prepped_dlist_handle(self.agent, handle)

    """
    @brief Estimate the cost of a transfer operation.
           Times are in microseconds and the method indicates how the estimation was performed.

    @param req_handle Handle to the transfer operation.
    @return Tuple of duration, error margin, method
    """

    def estimate_xfer_cost(self, req_handle: nixl_xfer_handle) -> tuple[int, int, int]:
        duration, err_margin, method = self.agent.estimateXferCost(req_handle._handle)
        if method == nixlBind.NIXL_COST_ANALYTICAL_BACKEND:
            method = "ANALYTICAL_BACKEND"
        else:
            method = "UNKNOWN"
        return duration, err_margin, method

    """
    @brief Prepare a transfer operation using prep_xfer_dlist handles.

    @param operation Type of operation ("WRITE" or "READ").
    @param local_xfer_side Handle to the local transfer descriptor list,
            received from prep_xfer_dlist.
    @param local_indices List or numpy array (dtype=int32) of indices for selecting local descriptors.
    @param remote_xfer_side Handle to the remote (or loopback) transfer descriptor list,
            received from prep_xfer_dlist.
    @param remote_indices List or numpy array (dtype=int32) of indices for selecting remote descriptors.
    @param notif_msg Optional notification message to send after transfer is done.
           notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
    @param backends Optional list of backend names to limit which backends NIXL can use.
    @param skip_desc_merge Whether to skip descriptor merging optimization.
    @return Opaque handle for posting/checking transfer.
            The handle can be released by calling release_xfer_handle from agent, or release() method on itself.
    """

    def make_prepped_xfer(
        self,
        operation: str,
        local_xfer_side: nixl_prepped_dlist_handle,
        local_indices: Union[list[int], np.ndarray],
        remote_xfer_side: nixl_prepped_dlist_handle,
        remote_indices: Union[list[int], np.ndarray],
        notif_msg: bytes = b"",
        backends: list[str] = [],
        skip_desc_merge: bool = False,
    ) -> nixl_xfer_handle:
        op = self.nixl_ops[operation]
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        handle = self.agent.makeXferReq(
            op,
            local_xfer_side._handle,
            local_indices,
            remote_xfer_side._handle,
            remote_indices,
            notif_msg,
            handle_list,
            skip_desc_merge,
        )

        return nixl_xfer_handle(self.agent, handle)

    """
    @brief  Initialize a transfer operation. This is a combined API, to create a transfer request
            from two descriptor lists, where NIXL prepares the descriptor lists and then the transfer.
            If there are common descriptors across different transfer requests, using
            this combined API will result in repeated computation, such as validity checks and
            pre-processing done in the preparation step.

    @param operation Type of operation ("WRITE" or "READ").
    @param local_descs List of local transfer descriptors, from get_xfer_descs.
    @param remote_descs List of remote (or loopback) transfer descriptors, from get_xfer_descs.
    @param remote_agent Name of the remote agent.
    @param notif_msg Optional notification message.
           notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
    @param backends Optional list of backend names to limit which backends NIXL can use.
    @return Opaque handle for posting/checking transfer.
            The handle can be released by calling release_xfer_handle from agent, or release() method on itself.
    """

    def initialize_xfer(
        self,
        operation: str,
        local_descs: nixlBind.nixlXferDList,
        remote_descs: nixlBind.nixlXferDList,
        remote_agent: str,
        notif_msg: bytes = b"",
        backends: list[str] = [],
    ) -> nixl_xfer_handle:
        op = self.nixl_ops[operation]
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        handle = self.agent.createXferReq(
            op, local_descs, remote_descs, remote_agent, notif_msg, handle_list
        )

        return nixl_xfer_handle(self.agent, handle)

    """
    @brief  Initiate a data transfer operation.
            After calling this, the transfer state can be checked asynchronously till completion.
            In case of small transfers that are completed as part of the call itself, return value
            will be "DONE", otherwise "PROC" or "ERR".

    @param handle Handle to the transfer operation, from make_prepped_xfer, or initialize_xfer.
    @param notif_msg Optional notification message can be specified or updated per transfer call.
           notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
    @return Status of the transfer operation ("DONE", "PROC", or "ERR").
    """

    def transfer(self, handle: nixl_xfer_handle, notif_msg: bytes = b"") -> str:
        status = self.agent.postXferReq(handle._handle, notif_msg)
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        elif status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        else:
            return "ERR"

    """
    @brief Check the state of a transfer operation.

    @param handle Handle to the transfer operation, from make_prepped_xfer, or initialize_xfer.
    @return Status of the transfer operation ("DONE", "PROC", or "ERR").
    """

    def check_xfer_state(self, handle: nixl_xfer_handle) -> str:
        status = self.agent.getXferStatus(handle._handle)
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        elif status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        else:
            return "ERR"

    """
    @brief Get telemetry information of a transfer request.
           The output object has three time values fields in microseconds
           (startTime, postDuration, xferDuration), as well as integer totalBytes transferred
           for the request, and integer descCount representing number of descriptors involved
           (for example if there was some merging of descriptors).

    @param handle Handle to the transfer operation, from make_prepped_xfer or initialize_xfer.
    @return nixlXferTelemetry object
    """

    def get_xfer_telemetry(
        self, handle: nixl_xfer_handle
    ) -> nixlBind.nixlXferTelemetry:
        return self.agent.getXferTelemetry(handle._handle)

    """
    @brief Query the backend that was chosen for a transfer operation.

    @param handle Handle to the transfer operation.
    @return Name of the backend decided for the transfer.
    """

    def query_xfer_backend(self, handle: nixl_xfer_handle) -> str:
        b_handle = self.agent.queryXferBackend(handle._handle)
        # this works because there should not be multiple matching handles in the Dict
        return next(
            backendS
            for backendS, backendH in self.backends.items()
            if backendH == b_handle
        )

    """
    @brief  Releases a transfer handle, which internally frees the memory used for the handle.
            If the transfer is active, NIXL will attempt to cancel it.
            If it cannot be canceled, an error will be returned and the handle will not be freed.

    @param handle Handle to the transfer operation from initialize_xfer or make_xfer.
    """

    def release_xfer_handle(self, handle: nixl_xfer_handle):
        handle.release()

    """
    @brief Release a descriptor list handle, which internally frees the memory used for the handle.

    @param handle Handle to the descriptor list from make_prepped_dlist.
    """

    def release_dlist_handle(self, handle: nixl_prepped_dlist_handle):
        handle.release()

    """
    @brief Get new notifications that have come to the agent.

    @param backends Optional list of backend names to limit which backends are checked for notifications.
    @return Dictionary of new notifications.
            Return Dict is a map of remote agent names to a list of notification messages from that agent.
    """

    def get_new_notifs(self, backends: list[str] = []) -> dict[str, list[bytes]]:
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        return self.agent.getNotifs({}, handle_list)

    """
    @brief Update notifications in a map
            Same as get_new_notifs, but returns all unhandled notifications in agent.

    @param backends Optional list of backend names to limit which backends are checked for notifications.
    @return Dictionary of updated notifications.
    """

    def update_notifs(self, backends: list[str] = []) -> dict[str, list[bytes]]:
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.notifs = self.agent.getNotifs(self.notifs, handle_list)  # Adds new notifs
        return self.notifs

    """
    @brief Check if a remote transfer is done with a specific notification.
           Will only remove the notification that is found.

    @param remote_agent_name Name of the remote agent.
    @param lookup_tag A tag to match against available messages in the notification map.
           The tag Can be the same as the entire expected message.
    @param backends Optional list of backend names to limit which backends are checked for notifications.
    @param tag_is_prefix Optionally specify that the tag you want to search with is just a prefix, or can be search as a substring of the full message.
    @return True if the notification is found, False otherwise.
    """

    def check_remote_xfer_done(
        self,
        remote_agent_name: str,
        lookup_tag: bytes,
        backends: list[str] = [],
        tag_is_prefix=True,
    ) -> bool:
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.notifs = self.agent.getNotifs(self.notifs, handle_list)  # Adds new notifs
        found = False
        message = None

        if remote_agent_name in self.notifs:
            for msg in self.notifs[remote_agent_name]:
                if (tag_is_prefix and msg.startswith(lookup_tag)) or (
                    not tag_is_prefix and lookup_tag in msg
                ):
                    message = msg
                    found = True
                    break
        if message:
            self.notifs[remote_agent_name].remove(message)
        return found

    """
    @brief Send a standalone notification to a remote agent, not bound to a transfer.

    @param remote_agent_name Name of the remote agent.
    @param notif_msg Message to send, it will be received as bytes.
           notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
    @param backends Optional a backend name to use to send the notifications.
    """

    def send_notif(
        self, remote_agent_name: str, notif_msg: bytes, backend: Optional[str] = None
    ):
        if backend is None:
            self.agent.genNotif(remote_agent_name, notif_msg)
        else:
            self.agent.genNotif(remote_agent_name, notif_msg, self.backends[backend])

    """
    @brief Get the full metadata of the local agent.

    @return Metadata of the local agent, in bytes.
    """

    def get_agent_metadata(self) -> bytes:
        return self.agent.getLocalMD()

    """
    @brief Get partial metadata of the local agent.

    @param descs         The list of descriptors to include metadata about.
                         List can be empty if only trying to send connection info.
    @param inc_conn_info Whether to include connection info in the metadata.
    @param backends      List of backends to consider when constructing partial metadata.

    @return Metadata of the local agent, in bytes.
    """

    def get_partial_agent_metadata(
        self,
        descs: nixlBind.nixlRegDList,
        inc_conn_info: bool = False,
        backends: list[str] = [],
    ) -> bytes:
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        return self.agent.getLocalPartialMD(descs, inc_conn_info, handle_list)

    """
    @brief Add a remote agent using its metadata. After this call, current agent can
            initiate transfers towards the remote agent.

    @param metadata Metadata of the remote agent, received out-of-band in bytes.
    @return Name of the added remote agent.
    """

    def add_remote_agent(self, metadata: bytes) -> str:
        agent_name = self.agent.loadRemoteMD(metadata)
        return agent_name

    """
    @brief Remove a remote agent. After this call, current agent cannot initiate
            transfers towards the remote agent specified in the call anymore.
            This call will also result in a disconnect between the two agents.

    @param agent Name of the remote agent.
    """

    def remove_remote_agent(self, agent: str):
        self.agent.invalidateRemoteMD(agent)

    """
    @brief Send all of your metadata to a peer or central metadata server.

    @param ip_addr If specified, will only send metadata to one peer by IP address.
                   Otherwise, metadata will be sent to central metadata server, if supported.
    @param port    If specified next to ip_addr, will try to send to this specific port of a peer.
                   Ignored when sending to a central metadata server.
    """

    def send_local_metadata(self, ip_addr: str = "", port: int = DEFAULT_COMM_PORT):
        self.agent.sendLocalMD(ip_addr, port)

    """
    @brief Send partial metadata of the local agent to a peer or central metadata server.

    @param descs         The list of descriptors to include metadata about.
                         List can be empty if only trying to send connection info.
    @param inc_conn_info Whether to include connection info in the metadata.
    @param backends      List of backends to consider when constructing partial metadata.
    @param ip_addr       If specified, will only send metadata to one peer by IP address.
                         Otherwise, metadata will be sent to central metadata server, if supported.
    @param port          If specified next to ip_addr, will try to send to this specific port of a peer.
                         Ignored when sending to a central metadata server.
    @param label         Label to use for the metadata when sending to central metadata server.
                         Ignored when sending to a peer.
    """

    def send_partial_agent_metadata(
        self,
        descs: nixlBind.nixlRegDList,
        inc_conn_info: bool = False,
        backends: list[str] = [],
        ip_addr: str = "",
        port: int = DEFAULT_COMM_PORT,
        label: str = "",
    ):
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.sendLocalPartialMD(
            descs, inc_conn_info, handle_list, ip_addr, port, label
        )

    """
    @brief Request metadata be retrieved from central metadata server or sent by peer.

    @param ip_addr If specified, will request metadata from one peer by IP address.
    @param port    If specified, will try to request on specific port.
    """

    def fetch_remote_metadata(
        self,
        remote_agent: str,
        ip_addr: str = "",
        port: int = DEFAULT_COMM_PORT,
        label: str = "",
    ):
        self.agent.fetchRemoteMD(remote_agent, ip_addr, port, label)

    """
    @brief Invalidate your own metadata in the central metadata server, or from a specific peer.

    @param ip_addr If specified, will only send invalidation to one peer by IP address.
    @param port    If specified, will try to send to specific port.
    """

    def invalidate_local_metadata(
        self, ip_addr: str = "", port: int = DEFAULT_COMM_PORT
    ):
        self.agent.invalidateLocalMD(ip_addr, port)

    """
    @brief Check if the remote metadata for a specific agent is available.
           When partial metadata methods are used, the descriptor list in question can be specified.

    @param agent Name of the remote agent.

    @return True if available, False otherwise
    """

    def check_remote_metadata(
        self, agent: str, descs: nixlBind.nixlXferDList = None
    ) -> bool:
        if descs is None:  # Just empty list, mem_type not important
            descs = nixlBind.nixlXferDList(nixlBind.DRAM_SEG)
        if self.agent.checkRemoteMD(agent, descs) == nixlBind.NIXL_SUCCESS:
            return True
        else:
            return False

    """
    @brief Get nixlXferDList from different input types:
            a) list of 3 element tuples (address, len, device ID) alongside a mandatory memory type
            b) a tensor
            c) a list of tensors
            d) a Nx3 2D numpy array, each row defines a single descriptor (address, len, device ID),
               alongside a mandatory memory type
            e) passes along if an xfer_dlist is given.

    @param descs List of any of the above types
    @param mem_type Optional memory type necessary for (a).
    @return Transfer descriptor list, nixlXferDList.
    """

    def get_xfer_descs(
        self,
        descs,
        mem_type: Optional[str] = None,
    ) -> nixlBind.nixlXferDList:
        # can add check for DLPack input

        if isinstance(descs, nixlBind.nixlXferDList):
            return descs
        elif isinstance(descs, nixlBind.nixlRegDList):
            logger.error("RegList type detected for transfer, please use XferList")
            new_descs = None
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 3:
                new_descs = nixlBind.nixlXferDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error("3-tuple list needed for transfer")
                new_descs = None
        elif isinstance(descs, np.ndarray):
            if mem_type is not None and descs.ndim == 2 and descs.shape[1] == 3:
                new_descs = nixlBind.nixlXferDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error(
                    "Nx3 shape required for transfer descriptor list from numpy array"
                )
                new_descs = None
        elif isinstance(descs, torch.Tensor):
            if descs.is_contiguous():
                mem_type = "cuda" if str(descs.device).startswith("cuda") else "cpu"
                base_addr = descs.data_ptr()
                region_len = descs.numel() * descs.element_size()
                gpu_id = descs.get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                new_descs = nixlBind.nixlXferDList(
                    self.nixl_mems[mem_type], [(base_addr, region_len, gpu_id)]
                )
            else:
                logger.error("Please use a list of contiguous Tensors")
                new_descs = None
        elif isinstance(descs[0], torch.Tensor):  # List[torch.Tensor]:
            tensor_type = descs[0].device
            dlist = np.zeros((len(descs), 3), dtype=np.uint64)

            for i in range(len(descs)):
                if descs[i].device != tensor_type:
                    return None
                if not descs[i].is_contiguous():
                    logger.error("Please use a list of contiguous Tensors")
                    return None
                base_addr = descs[i].data_ptr()
                region_len = descs[i].numel() * descs[i].element_size()
                gpu_id = descs[i].get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                dlist[i, :] = (base_addr, region_len, gpu_id)
            mem_type = "cuda" if str(tensor_type).startswith("cuda") else "cpu"
            new_descs = nixlBind.nixlXferDList(self.nixl_mems[mem_type], dlist)
        else:
            new_descs = None

        return new_descs

    """
    @brief Get nixlRegDList from different input types:
            a) list of 4 element tuples (address, len, device ID, meta info) alongside a mandatory memory type
            b) a tensor
            c) a list of tensors
            d) a Nx3 2D numpy array, each row defines a single descriptor (address, len, device ID),
               alongside a mandatory memory type. Empty meta info will be considered for each descriptor.
            e) passes along if a reg_dlist is given.

    @param descs List of any of the above types
    @param mem_type Optional memory type necessary for (a).
    @return Registration descriptor list, nixlRegDList.
    """

    def get_reg_descs(
        self,
        descs,
        mem_type: Optional[str] = None,
    ) -> nixlBind.nixlRegDList:
        # can add check for DLPack input

        if isinstance(descs, nixlBind.nixlRegDList):
            return descs
        elif isinstance(descs, nixlBind.nixlXferDList):
            logger.error("XferList type detected for registration, please use RegList")
            new_descs = None
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 4:
                new_descs = nixlBind.nixlRegDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error("4-tuple list needed for registration")
                new_descs = None
        elif isinstance(descs, np.ndarray):
            if mem_type is not None and descs.ndim == 2 and descs.shape[1] == 3:
                new_descs = nixlBind.nixlRegDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error(
                    "Nx3 shape required for transfer descriptor list from numpy array"
                )
                new_descs = None
        elif isinstance(descs, torch.Tensor):
            if descs.is_contiguous():
                mem_type = "cuda" if str(descs.device).startswith("cuda") else "cpu"
                base_addr = descs.data_ptr()
                region_len = descs.numel() * descs.element_size()
                gpu_id = descs.get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                new_descs = nixlBind.nixlRegDList(
                    self.nixl_mems[mem_type], [(base_addr, region_len, gpu_id, "")]
                )
            else:
                logger.error("Please use a list of contiguous Tensors")
                new_descs = None
        elif isinstance(descs[0], torch.Tensor):  # List[torch.Tensor]:
            tensor_type = descs[0].device
            dlist = np.zeros((len(descs), 3), dtype=np.uint64)

            for i in range(len(descs)):
                if descs[i].device != tensor_type:
                    return None
                if not descs[i].is_contiguous():
                    logger.error("Please use a list of contiguous Tensors")
                    return None
                base_addr = descs[i].data_ptr()
                region_len = descs[i].numel() * descs[i].element_size()
                gpu_id = descs[i].get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                dlist[i, :] = (base_addr, region_len, gpu_id)
            mem_type = "cuda" if str(tensor_type).startswith("cuda") else "cpu"
            new_descs = nixlBind.nixlRegDList(self.nixl_mems[mem_type], dlist)
        else:
            new_descs = None

        return new_descs

    """
    @brief Serialize NIXL descriptor list with pickle.

    @param descs NIXL list to serialize.
    @return Serialized descriptor list.
    """

    def get_serialized_descs(self, descs) -> bytes:
        return pickle.dumps(descs)

    """
    @brief Deserialize NIXL descriptor list.

    @param serialized_descs Serialized NIXL descriptor list.
    @return Deserialized NIXL descriptor list.
    """

    def deserialize_descs(self, serialized_descs: bytes):
        return pickle.loads(serialized_descs)
