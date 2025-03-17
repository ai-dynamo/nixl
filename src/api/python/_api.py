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

import torch

import nixl._bindings as nixlBind


class nixl_agent_config:
    def __init__(self, enable_prog_thread=True, backends=["UCX"]):
        # TODO: add backend init parameters
        self.backends = backends
        self.enable_pthread = enable_prog_thread


class nixl_agent:
    def __init__(self, agent_name, nixl_conf=None, instantiate_all=False):
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
        self.notifs = {}
        self.backends = {}
        self.backend_mems = {}
        self.backend_options = {}

        self.plugin_list = self.agent.getAvailPlugins()
        if len(self.plugin_list) == 0:
            print("No plugins available, cannot start transfers!")
            raise RuntimeError("No plugins available for NIXL, cannot start transfers!")

        self.plugin_b_options = {}
        self.plugin_mem_types = {}
        for plugin in self.plugin_list:
            (backend_options, mem_types) = self.agent.getPluginParams(plugin)
            self.plugin_b_options[plugin] = backend_options
            self.plugin_mem_types[plugin] = mem_types

        init = {}

        if instantiate_all:
            # TODO: populate init from default parameters, or define a set of params in python
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

        print("Initializied NIXL agent:", agent_name)

    def get_plugin_list(self):
        return self.plugin_list

    def get_plugin_mem_types(self, backend):
        if backend in self.plugin_mem_types:
            return self.plugin_mem_types[backend]
        else:
            print("Plugin", backend, "is not available to get its supported mem types.")
            return None

    def get_plugin_params(self, backend):
        if backend in self.plugin_b_options:
            return self.plugin_b_options[backend]
        else:
            print("Plugin", backend, "is not available to get its parameters.")
            return None

    def get_backend_mem_types(self, backend):
        if backend in self.backend_mems:
            return self.backend_mems[backend]
        else:
            print(
                "Backend", backend, "not instantiated to get its supported mem types."
            )
            return None

    def get_backend_params(self, backend):
        if backend in self.backend_options:
            return self.backend_options[backend]
        else:
            print("Backend", backend, "not instantiated to get its parameters.")
            return None

    def create_backend(self, backend, initParams={}):
        self.backends[backend] = self.agent.createBackend(backend, initParams)

        (backend_options, mem_types) = self.agent.getBackendParams(
            self.backends[backend]
        )
        self.backend_mems[backend] = mem_types
        self.backend_options[backend] = backend_options

    # The returned descriptor object can be used for call to deregister
    def register_memory(
        self,
        reg_list,
        mem_type=None,
        is_unified_addr=True,
        is_sorted=False,
        backend=None,
    ):
        reg_descs = self.get_reg_descs(reg_list, mem_type, is_unified_addr, is_sorted)

        # based on backend type and mem_type, figure what registrations are meaningful
        if backend:
            ret = self.agent.registerMem(reg_descs, self.backends[backend])
        else:
            # TODO: rely on underlying capability to register with all when supported
            if (reg_descs.getType() == nixlBind.FILE_SEG) and ("GDS" in self.backends):
                ret = self.agent.registerMem(reg_descs, self.backends["GDS"])
            elif (reg_descs.getType() == nixlBind.DRAM_SEG) and (
                "UCX" in self.backends
            ):
                ret = self.agent.registerMem(reg_descs, self.backends["UCX"])
            elif (reg_descs.getType() == nixlBind.VRAM_SEG) and (
                "UCX" in self.backends
            ):
                ret = self.agent.registerMem(reg_descs, self.backends["UCX"])
            elif (reg_descs.getType() == nixlBind.VRAM_SEG) and (
                "GDS" in self.backends
            ):
                ret = self.agent.registerMem(reg_descs, self.backends["GDS"])
            else:
                return None

        if ret != 0:
            return None
        return reg_descs

    def deregister_memory(self, dereg_list, backend=None):
        # based on backend type and mem_type, figure what deregistrations are needed
        if backend:
            self.agent.deregisterMem(dereg_list, self.backends[backend])
        else:
            # TODO: rely on underlying capability to register with all when supported
            if (dereg_list.getType() == nixlBind.FILE_SEG) and ("GDS" in self.backends):
                ret = self.agent.deregisterMem(dereg_list, self.backends["GDS"])
            elif (dereg_list.getType() == nixlBind.DRAM_SEG) and (
                "UCX" in self.backends
            ):
                ret = self.agent.deregisterMem(dereg_list, self.backends["UCX"])
            elif (dereg_list.getType() == nixlBind.VRAM_SEG) and (
                "UCX" in self.backends
            ):
                ret = self.agent.deregisterMem(dereg_list, self.backends["UCX"])
            elif (dereg_list.getType() == nixlBind.VRAM_SEG) and (
                "GDS" in self.backends
            ):
                ret = self.agent.deregisterMem(dereg_list, self.backends["GDS"])
            else:
                return None

        if ret != 0:
            return None
        # is this the best ret value?
        return dereg_list

    # Optional proactive make connection
    def make_connection(self, remote_agent):
        self.agent.makeConnection(remote_agent)

    # "" remote agent means local. example xfer can be used to know the backend
    def prep_xfer_dlist(
        self,
        remote_agent,
        xfer_list,
        mem_type=None,
        is_unified_addr=True,
        is_sorted=False,
        xfer_backend=None,
    ):
        descs = self.get_xfer_descs(xfer_list, mem_type, is_unified_addr, is_sorted)
        if xfer_backend:
            handle = self.agent.prepXferDlist(remote_agent, descs, xfer_backend)
        else:
            # TODO: rely on underlying capability to register with all when supported
            if (descs.getType() == nixlBind.FILE_SEG) and ("GDS" in self.backends):
                handle = self.agent.prepXferDlist(
                    remote_agent, descs, self.backends["GDS"]
                )
            elif (descs.getType() == nixlBind.DRAM_SEG) and ("UCX" in self.backends):
                handle = self.agent.prepXferDlist(
                    remote_agent, descs, self.backends["UCX"]
                )
            elif (descs.getType() == nixlBind.VRAM_SEG) and ("UCX" in self.backends):
                handle = self.agent.prepXferDlist(
                    remote_agent, descs, self.backends["UCX"]
                )
            elif (descs.getType() == nixlBind.VRAM_SEG) and ("GDS" in self.backends):
                handle = self.agent.prepXferDlist(
                    remote_agent, descs, self.backends["GDS"]
                )
            else:
                return None

        if handle == 0:
            return None

        return handle

    def make_prepped_xfer(
        self,
        operation,
        local_xfer_side,
        local_indices,
        remote_xfer_side,
        remote_indices,
        notif_msg="",
        skip_desc_merge=False,
    ):
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
            if handle == 0:
                return None

            return handle
        else:
            return None

    def initialize_xfer(
        self,
        operation,
        local_descs,
        remote_descs,
        remote_agent,
        notif_msg="",
        xfer_backend=None,
    ):
        op = self.nixl_ops[operation]
        if op:
            if xfer_backend:
                handle = self.agent.createXferReq(
                    op,
                    local_descs,
                    remote_descs,
                    remote_agent,
                    notif_msg,
                    xfer_backend,
                )
            else:
                handle = self.agent.createXferReq(
                    op, local_descs, remote_descs, remote_agent, notif_msg
                )

            if handle == 0:
                return None
            return handle  # In case of error it will be None
        else:
            return None

    def transfer(self, handle, notif_msg=""):
        status = self.agent.postXferReq(handle, notif_msg)
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        elif status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        else:
            return "ERR"

    def check_xfer_state(self, handle):
        status = self.agent.getXferStatus(handle)
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        elif status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        else:
            return "ERR"

    def query_xfer_backend(self, handle):
        b_handle = self.agent.queryXferBackend(handle)
        # this works because there should not be multiple matching handles in the Dict
        return next(
            backendS for backendS, backendH in self.backends if backendH == b_handle
        )

    def release_xfer_handle(self, handle):
        # frees the handle too
        self.agent.releaseXferReq(handle)

    def release_dlist_handle(self, handle):
        # frees the handle too
        self.agent.releasedDlistH(handle)

    # Returns new notifs, without touching self.notifs
    def get_new_notifs(self):
        return self.agent.getNotifs({})

    # Adds new notifs to self.notifs and returns it
    def update_notifs(self):
        self.notifs = self.agent.getNotifs(self.notifs)
        return self.notifs

    # Only removes the specific notification from self.notifs
    def check_remote_xfer_done(self, remote_agent_name, lookup_msg):
        self.notifs = self.agent.getNotifs(self.notifs)  # Adds new notifs
        message = None
        if remote_agent_name in self.notifs:
            for msg in self.notifs[remote_agent_name]:
                if lookup_msg in msg:
                    message = msg
                    break
        if message:
            self.notifs[remote_agent_name].remove(message)
        return message

    # Extra notification APIs
    def send_notif(self, remote_agent_name, notif_msg):
        self.agent.genNotif(remote_agent_name, notif_msg)

    def get_agent_metadata(self):
        return self.agent.getLocalMD()

    def add_remote_agent(self, metadata):
        agent_name = self.agent.loadRemoteMD(metadata)
        return agent_name

    def remove_remote_agent(self, agent):
        self.agent.invalidateRemoteMD(agent)

    # 4 methods to create and serialize/deserialize descriptors, provided through Agent

    def get_xfer_descs(
        self, descs, mem_type=None, is_unified_addr=True, is_sorted=False
    ):
        # can add check for DLPack input

        if isinstance(descs, nixlBind.nixlXferDList):
            return descs
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 3:
                new_descs = nixlBind.nixlXferDList(
                    self.nixl_mems[mem_type], descs, is_unified_addr, is_sorted
                )
            elif mem_type is None:
                print("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                print("3-tuple list needed for transfer")
                new_descs = None
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
            new_descs = nixlBind.nixlXferDList(
                self.nixl_mems[str(tensor_type)], dlist, is_unified_addr, is_sorted
            )
        elif isinstance(descs, nixlBind.nixlRegDList):
            print("RegList type detected for transfer, please use XferList")
            new_descs = None
        else:
            new_descs = None

        return new_descs

    def get_reg_descs(
        self, descs, mem_type=None, is_unified_addr=True, is_sorted=False
    ):
        # can add check for DLPack input

        if isinstance(descs, nixlBind.nixlRegDList):
            return descs
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 4:
                new_descs = nixlBind.nixlRegDList(
                    self.nixl_mems[mem_type], descs, is_unified_addr, is_sorted
                )
            elif mem_type is None:
                print("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                print("4-tuple list needed for registration")
                new_descs = None
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
            new_descs = nixlBind.nixlRegDList(
                self.nixl_mems[str(tensor_type)], dlist, is_unified_addr, is_sorted
            )
        elif isinstance(descs, nixlBind.nixlXferDList):
            print("XferList type detected for registration, please use RegList")
            new_descs = None
        else:
            new_descs = None

        return new_descs

    def get_serialized_descs(self, descs):
        return pickle.dumps(descs)

    def deserialize_descs(self, serialized_descs):
        return pickle.loads(serialized_descs)
