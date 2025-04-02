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
from typing import Optional

import torch

import nixl._bindings as nixlBind

# Opaque nixl handle types
nixl_backend_handle = int
nixl_prepped_dlist_handle = int
nixl_xfer_handle = int


class nixl_agent_config:
    def __init__(self, enable_prog_thread=True, backends=["UCX"]):
        # TODO: add backend init parameters
        self.backends = backends
        self.enable_pthread = enable_prog_thread


class nixl_agent:
    def __init__(
        self,
        agent_name: str,
        nixl_conf: Optional[nixl_agent_config] = None,
        instantiate_all: bool = False,
    ):
        if nixl_conf and instantiate_all:
            instantiate_all = False
            print(
                "Ignoring instantiate_all based on the provided config in agent creation."
            )
        if not nixl_conf:
            nixl_conf = nixl_agent_config()  # Using defaults set in nixl_agent_config

        # Set agent config and instantiate an agent
        agent_config = nixlBind.nixlAgentConfig(nixl_conf.enable_pthread)
        self.agent = nixlBind.nixlAgent(agent_name, agent_config)

        self.name = agent_name
        self.notifs: dict[str, list[bytes]] = {}
        self.backends: dict[str, nixl_backend_handle] = {}
        self.backend_mems: dict[str, list[str]] = {}
        self.backend_options: dict[str, dict[str, str]] = {}

        self.plugin_list = self.agent.getAvailPlugins()
        if len(self.plugin_list) == 0:
            print("No plugins available, cannot start transfers!")
            raise RuntimeError("No plugins available for NIXL, cannot start transfers!")

        self.plugin_b_options: dict[str, dict[str, str]] = {}
        self.plugin_mem_types: dict[str, list[str]] = {}
        for plugin in self.plugin_list:
            (backend_options, mem_types) = self.agent.getPluginParams(plugin)
            self.plugin_b_options[plugin] = backend_options
            self.plugin_mem_types[plugin] = mem_types

        # TODO: populate init from default parameters, or define a set of params in python
        init: dict[str, str] = {}

        if instantiate_all:
            for plugin in self.plugin_list:
                self.backends[plugin] = self.agent.createBackend(plugin, init)
        else:
            for bknd in nixl_conf.backends:
                # TODO: populate init from nixl_conf when added
                if bknd not in self.plugin_list:
                    print(
                        "Skipping backend registration",
                        bknd,
                        "due to the missing plugin.",
                    )
                else:
                    self.backends[bknd] = self.agent.createBackend(bknd, init)

        for backend in self.backends:
            (backend_options, mem_types) = self.agent.getBackendParams(
                self.backends[backend]
            )
            self.backend_mems[backend] = mem_types
            self.backend_options[backend] = backend_options

        self.nixl_mems = {
            "DRAM": nixlBind.DRAM_SEG,
            "VRAM": nixlBind.VRAM_SEG,
            "cpu": nixlBind.DRAM_SEG,
            "cuda": nixlBind.VRAM_SEG,
        }
        self.nixl_ops = {
            "WRITE": nixlBind.NIXL_WRITE,
            "READ": nixlBind.NIXL_READ,
        }

        print("Initialized NIXL agent:", agent_name)

    def get_plugin_list(self) -> list[str]:
        return self.plugin_list

    def get_plugin_mem_types(self, backend: str) -> list[str]:
        if backend in self.plugin_mem_types:
            return self.plugin_mem_types[backend]
        else:
            print("Plugin", backend, "is not available to get its supported mem types.")
            return []

    def get_plugin_params(self, backend: str) -> dict[str, str]:
        if backend in self.plugin_b_options:
            return self.plugin_b_options[backend]
        else:
            print("Plugin", backend, "is not available to get its parameters.")
            return {}

    def get_backend_mem_types(self, backend: str) -> list[str]:
        if backend in self.backend_mems:
            return self.backend_mems[backend]
        else:
            print(
                "Backend", backend, "not instantiated to get its supported mem types."
            )
            return []

    def get_backend_params(self, backend: str) -> dict[str, str]:
        if backend in self.backend_options:
            return self.backend_options[backend]
        else:
            print("Backend", backend, "not instantiated to get its parameters.")
            return {}

    # initParams is a Dict of strings (input params) to strings (values)
    def create_backend(self, backend: str, initParams: dict[str, str] = {}):
        self.backends[backend] = self.agent.createBackend(backend, initParams)

        (backend_options, mem_types) = self.agent.getBackendParams(
            self.backends[backend]
        )
        self.backend_mems[backend] = mem_types
        self.backend_options[backend] = backend_options

    # For reg_list, it gets a) list of 4 element tuples, b) a tensor, c) a list
    # of tensors, or d) a reg_dlist from output of get_reg_desc. is given.
    # The next 3 optional parameters are dlist options, and finally an optional
    # backend engine can be specified for registration.
    # The returned descriptor object can be used for call to deregister
    def register_memory(
        self,
        reg_list,
        mem_type: Optional[str] = None,
        is_sorted: bool = False,
        backends: list[str] = [],
    ) -> nixlBind.nixlRegDList:
        reg_descs = self.get_reg_descs(reg_list, mem_type, is_sorted)

        # based on backend type and mem_type, figure what registrations are meaningful
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.registerMem(reg_descs, handle_list)

        return reg_descs

    # The output from get_reg_descs (which is later passed to register_memory for
    # registration) or direct output of register_memory is passed here
    def deregister_memory(
        self, dereg_list: nixlBind.nixlRegDList, backends: list[str] = []
    ):
        # based on backend type and mem_type, figure what deregistrations are needed
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.deregisterMem(dereg_list, handle_list)

    # Optional proactive make connection
    def make_connection(self, remote_agent: str):
        self.agent.makeConnection(remote_agent)

    # agent_name should be "NIXL_INIT_AGENT" for local descriptors on the initiator side. For a
    # remote agent, it would be that agent's name, or for loopback, local agent's name as target.
    # xfer_list can be any of the types supported by get_xfer_descs
    def prep_xfer_dlist(
        self,
        agent_name: str,
        xfer_list,
        mem_type: Optional[str] = None,
        is_sorted: bool = False,
        backends: list[str] = [],
    ) -> nixl_prepped_dlist_handle:
        descs = self.get_xfer_descs(xfer_list, mem_type, is_sorted)

        if agent_name == "NIXL_INIT_AGENT" or agent_name == "":
            agent_name = nixlBind.NIXL_INIT_AGENT

        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        handle = self.agent.prepXferDlist(agent_name, descs, handle_list)

        return handle

    # xfer_side parameters are opaque NIXL handles
    def make_prepped_xfer(
        self,
        operation: str,
        local_xfer_side: nixl_prepped_dlist_handle,
        local_indices: list[int],
        remote_xfer_side: nixl_prepped_dlist_handle,
        remote_indices: list[int],
        notif_msg: str = "",
        skip_desc_merge: bool = False,
    ) -> nixl_xfer_handle:
        op = self.nixl_ops[operation]
        if op:
            handle = self.agent.makeXferReq(
                op,
                local_xfer_side,
                local_indices,
                remote_xfer_side,
                remote_indices,
                notif_msg,
                skip_desc_merge,
            )

            return handle
        else:
            raise nixlBind.nixlInvalidParamError("Invalid op code")
            return nixlBind.nixlInvalidParamError

    def initialize_xfer(
        self,
        operation: str,
        local_descs: nixlBind.nixlXferDList,
        remote_descs: nixlBind.nixlXferDList,
        remote_agent: str,
        notif_msg: str = "",
        backends: list[str] = [],
    ) -> nixl_xfer_handle:
        op = self.nixl_ops[operation]
        if op:
            handle_list = []
            for backend_string in backends:
                handle_list.append(self.backends[backend_string])

            handle = self.agent.createXferReq(
                op, local_descs, remote_descs, remote_agent, notif_msg, handle_list
            )

            return handle
        else:
            raise nixlBind.nixlInvalidParamError("Invalid op code")
            return nixlBind.nixlInvalidParamError

    def transfer(self, handle: nixl_xfer_handle, notif_msg: str = "") -> str:
        status = self.agent.postXferReq(handle, notif_msg)
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        elif status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        else:
            return "ERR"

    def check_xfer_state(self, handle: nixl_xfer_handle) -> str:
        status = self.agent.getXferStatus(handle)
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        elif status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        else:
            return "ERR"

    def query_xfer_backend(self, handle: nixl_xfer_handle) -> str:
        b_handle = self.agent.queryXferBackend(handle)
        # this works because there should not be multiple matching handles in the Dict
        return next(
            backendS
            for backendS, backendH in self.backends.items()
            if backendH == b_handle
        )

    def release_xfer_handle(self, handle: nixl_xfer_handle):
        # frees the handle too
        self.agent.releaseXferReq(handle)

    def release_dlist_handle(self, handle: nixl_xfer_handle):
        # frees the handle too
        self.agent.releasedDlistH(handle)

    # Returns new notifs, without touching self.notifs
    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return self.agent.getNotifs({})

    # Adds new notifs to self.notifs and returns it
    def update_notifs(self) -> dict[str, list[bytes]]:
        self.notifs = self.agent.getNotifs(self.notifs)
        return self.notifs

    # Only removes the specific notification from self.notifs
    def check_remote_xfer_done(self, remote_agent_name: str, lookup_msg: bytes) -> bool:
        self.notifs = self.agent.getNotifs(self.notifs)  # Adds new notifs
        found = False
        message = None

        if remote_agent_name in self.notifs:
            for msg in self.notifs[remote_agent_name]:
                if lookup_msg in msg:
                    message = lookup_msg
                    found = True
                    break
        if message:
            self.notifs[remote_agent_name].remove(message)
        return found

    # Extra notification APIs
    def send_notif(self, remote_agent_name: str, notif_msg: str):
        # To be updated when automatic backend selection is supported
        self.agent.genNotif(remote_agent_name, notif_msg, self.backends["UCX"])

    def get_agent_metadata(self) -> bytes:
        return self.agent.getLocalMD()

    def add_remote_agent(self, metadata: bytes) -> str:
        agent_name = self.agent.loadRemoteMD(metadata)
        return agent_name

    def remove_remote_agent(self, agent: str):
        self.agent.invalidateRemoteMD(agent)

    # 4 methods to create and serialize/deserialize descriptors, provided through Agent

    # For descs, it gets a) list of 3 element tuples, b) a tensor, c) a list
    # of tensors, or d) passes along if an xfer_dlist is given. The other 3
    # optional parameters are dlist options.
    def get_xfer_descs(
        self,
        descs,
        mem_type: Optional[str] = None,
        is_sorted: bool = False,
    ) -> nixlBind.nixlXferDList:
        # can add check for DLPack input

        if isinstance(descs, nixlBind.nixlXferDList):
            return descs
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 3:
                new_descs = nixlBind.nixlXferDList(
                    self.nixl_mems[mem_type], descs, is_sorted
                )
            elif mem_type is None:
                print("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                print("3-tuple list needed for transfer")
                new_descs = None
        elif isinstance(descs, torch.Tensor):
            mem_type = "cuda" if str(descs.device).startswith("cuda") else "cpu"
            base_addr = descs.data_ptr()
            region_len = descs.numel() * descs.element_size()
            gpu_id = descs.get_device()
            if gpu_id == -1:  # DRAM
                gpu_id = 0
            new_descs = nixlBind.nixlRegDList(
                self.nixl_mems[mem_type],
                [(base_addr, region_len, gpu_id)],
                is_sorted,
            )
        elif isinstance(descs[0], torch.Tensor):  # List[torch.Tensor]:
            tensor_type = descs[0].device
            dlist = [(0, 0, 0)] * len(descs)

            for i in range(len(descs)):
                if descs[i].device != tensor_type:
                    return None
                base_addr = descs[i].data_ptr()
                region_len = descs[i].numel() * descs[i].element_size()
                gpu_id = descs[i].get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                dlist[i] = (base_addr, region_len, gpu_id)
            mem_type = "cuda" if str(tensor_type).startswith("cuda") else "cpu"
            new_descs = nixlBind.nixlXferDList(
                self.nixl_mems[mem_type], dlist, is_sorted
            )
        elif isinstance(descs, nixlBind.nixlRegDList):
            print("RegList type detected for transfer, please use XferList")
            new_descs = None
        else:
            new_descs = None

        return new_descs

    # For descs, it gets a) list of 4 element tuples, b) a tensor, c) a list
    # of tensors, or d) passes along if an xfer_dlist is given. The other 3
    # optional parameters are dlist options.
    def get_reg_descs(
        self,
        descs,
        mem_type: Optional[str] = None,
        is_sorted: bool = False,
    ) -> nixlBind.nixlRegDList:
        # can add check for DLPack input

        if isinstance(descs, nixlBind.nixlRegDList):
            return descs
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 4:
                new_descs = nixlBind.nixlRegDList(
                    self.nixl_mems[mem_type], descs, is_sorted
                )
            elif mem_type is None:
                print("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                print("4-tuple list needed for registration")
                new_descs = None
        elif isinstance(descs, torch.Tensor):
            mem_type = "cuda" if str(descs.device).startswith("cuda") else "cpu"
            base_addr = descs.data_ptr()
            region_len = descs.numel() * descs.element_size()
            gpu_id = descs.get_device()
            if gpu_id == -1:  # DRAM
                gpu_id = 0
            new_descs = nixlBind.nixlRegDList(
                self.nixl_mems[mem_type],
                [(base_addr, region_len, gpu_id, "")],
                is_sorted,
            )
        elif isinstance(descs[0], torch.Tensor):  # List[torch.Tensor]:
            tensor_type = descs[0].device
            dlist = [(0, 0, 0, "")] * len(descs)

            for i in range(len(descs)):
                if descs[i].device != tensor_type:
                    return None
                base_addr = descs[i].data_ptr()
                region_len = descs[i].numel() * descs[i].element_size()
                gpu_id = descs[i].get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                dlist[i] = (base_addr, region_len, gpu_id, "")
            mem_type = "cuda" if str(tensor_type).startswith("cuda") else "cpu"
            new_descs = nixlBind.nixlRegDList(
                self.nixl_mems[mem_type], dlist, is_sorted
            )
        elif isinstance(descs, nixlBind.nixlXferDList):
            print("XferList type detected for registration, please use RegList")
            new_descs = None
        else:
            new_descs = None

        return new_descs

    # nixl Descriptor Lists natively support pickling
    def get_serialized_descs(self, descs) -> bytes:
        return pickle.dumps(descs)

    def deserialize_descs(self, serialized_descs: bytes):
        return pickle.loads(serialized_descs)
