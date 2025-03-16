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

class nixl_config:
    def __init(self, enable_prog_thread = True, backends=["UCX", "GDS"]):
        # TODO: add backend init parameters
        self.backends = backends
        self.enable_pthread = enable_prog_thread

class nixl_agent:
    def __init__(self, agent_name, nixl_config=None):
        # Set agent config and instantiate an agent
        if nixl_config:
            agent_config = nixlBind.nixlAgentConfig(nixl_config.enable_pthread)
        else:
            agent_config = nixlBind.nixlAgentConfig(True)
        self.agent = nixlBind.nixlAgent(agent_name, agent_config)

        self.name = agent_name
        self.notifs = {}
        self.backends = {}
        self.backend_mems = {}
        self.backend_options = {}

        self.plugin_list = self.getAvailPlugins()
        if len(self.plugin_list) == 0:
            print("No plugins available, cannot start transfers!")

        self.plugin_b_options = {}
        self.plugin_mem_types = {}
        for plugin in self.plugin_list:
            (backend_options, mem_types) = self.agent.getPluginParams(plugin)
            self.plugin_b_options[plugin] = backend_options
            self.plugin_mem_types[plugin] = mem_types

        init = {}
        if nixl_config:
            for x in nixl_config.backends:
                # TODO: populate init from nixl_config when added
                if x not in self.plugin_list:
                    print("Skiping backend registration", x, "due to the missing plugin.")
                else:
                    self.backends[x] = self.agent.createBackend(x, init)
        else:
            # TODO: populate init from default parameters, or define a set of params in python
            for plugin in self.plugin_list:
                self.backends[plugin] = self.agent.createBackend(plugin, init)

        for backend in self.backends:
            (backend_options, mem_types) = self.agent.getBackendParams(backend)
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
            print ("Plugin", backend, "is not available to get its supported mem types.")
            return None

    def get_plugin_params(self, backend):
        if backend in self.plugin_b_options:
            return self.plugin_b_options[backend]
        else:
            print ("Plugin", backend, "is not available to get its parameters.")
            return None

    def get_backend_mem_types(self, backend):
        if backend in self.backend_mems:
            return self.backend_mems[backend]
        else:
            print ("Backend", backend, "not instantiated to get its supported mem types.")
            return None

    def get_backend_params(self, backend):
        if backend in self.backend_options:
            return self.backend_options[backend]
        else:
            print ("Backend", backend, "not instantiated to get its parameters.")
            return None

    def create_backend(self, backend, initParams=None):
        self.backends[backend] = self.agent.createBackend(backend, initParams)

        (backend_options, mem_types) = self.agent.getBackendParams(
            self.backends[backend]
        )
        self.backend_option_map[backend] = backend_options
        self.mem_type_map[backend] = mem_types

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

    #TODO: these not necessarily agent specific, maybe separate somehow?
    def get_serialized_descs(self, descs):
        return pickle.dumps(descs)

    def deserialize_descs(self, serialized_descs):
        return pickle.loads(serialized_descs)

    # The returned descriptor object can be used for call to deregister
    def register_memory(
        self, reg_list, mem_type=None, is_unified_addr=True, is_sorted=False, backend=None
    ):
        reg_descs = self.get_reg_descs(reg_list, mem_type, is_unified_addr, is_sorted)

        # based on backend type and mem_type, figure what registrations are meaningful
        if backend:
            ret = self.agent.registerMem(reg_descs, self.backends[backend])
        else:
            # TODO: rely on underlying capability to register with all when supported
            if (reg_descs.getType() == nixl.FILE_SEG) and ("GDS" in self.backend):
                ret = self.agent.registerMem(reg_descs, self.backends["GDS"])
            else if (reg_descs.getType() == nixl.DRAM_SEG) and ("UCX" in self.backend):
                ret = self.agent.registerMem(reg_descs, self.backends["UCX"])
            else if (reg_descs.getType() == nixl.VRAM_SEG) and ("UCX" in self.backend):
                ret = self.agent.registerMem(reg_descs, self.backends["UCX"])
            else if (reg_descs.getType() == nixl.VRAM_SEG) and ("GDS" in self.backend):
                ret = self.agent.registerMem(reg_descs, self.backends["GDS"])
        if ret != 0:
            return None
        return reg_descs

    def deregister_memory(self, dereg_descs, backend=None):
        # based on backend type and mem_type, figure what deregistrations are needed
        if backend:
            self.agent.deregisterMem(dereg_descs, self.backends[backend])
        else:
            # TODO: rely on underlying capability to register with all when supported
            if (reg_descs.getType() == nixl.FILE_SEG) and ("GDS" in self.backend):
                ret = self.agent.deregisterMem(reg_descs, self.backends["GDS"])
            else if (reg_descs.getType() == nixl.DRAM_SEG) and ("UCX" in self.backend):
                ret = self.agent.deregisterMem(reg_descs, self.backends["UCX"])
            else if (reg_descs.getType() == nixl.VRAM_SEG) and ("UCX" in self.backend):
                ret = self.agent.deregisterMem(reg_descs, self.backends["UCX"])
            else if (reg_descs.getType() == nixl.VRAM_SEG) and ("GDS" in self.backend):
                ret = self.agent.deregisterMem(reg_descs, self.backends["GDS"])
        if ret != 0:
            return None
        return reg_descs

    # Optional proactive make connection
    def make_connection(self, remote_agent):
        self.agent.makeConnection(remote_agent)

    # "" remote agent means local. example xfer can be used to know the backend
    def prep_xfer_dlist(
        self,
        xfer_list,
        remote_agent,
        mem_type=None,
        is_unified_addr=True,
        is_sorted=False,
        xfer_backend=None
    ):
        descs = self.get_xfer_descs(xfer_list, mem_type, is_unified_addr, is_sorted)
        if xfer_backend:
            handle = self.agent.prepXferDlist(descs, remote_agent, xfer_backend)
        else:
            #TODO: need better way to select backend if not specified
            if (descs.getType() == nixl.FILE_SEG) and ("GDS" in self.backend):
                handle = self.agent.prepXferDlist(
                    descs, remote_agent, self.backends["GDS"]
                )
            else:
                handle = self.agent.prepXferDlist(
                    descs, remote_agent, self.backends["UCX"]
                )
        if handle == 0:
            return None

        return handle

    def make_resolved_xfer(
        self,
        operation,
        local_xfer_side,
        local_indices,
        remote_xfer_side,
        remote_indices,
        notif_msg = "",
        skip_desc_merge = False
    ):
        op = self.nixl_ops[operation]
        if op:
            handle = self.agent.makeXferReq(
                local_xfer_side,
                local_indices,
                remote_xfer_side,
                remote_indices,
                notif_msg,
                op,
                skip_desc_merge,
            )
            if handle == 0:
                return None

            return handle
        else:
            return None

    def create_xfer(
        self,
        operation,
        local_descs,
        remote_descs,
        remote_agent,
        notif_msg = "",
        xfer_backend = None
    ):
        op = self.nixl_ops[operation]
        if op:
            if xfer_backend:
                handle = self.agent.createXferReq(
                    local_descs,
                    remote_descs,
                    remote_agent,
                    notif_msg,
                    op,
                    xfer_backend,
                )
            else:
                handle = self.agent.createXferReq(
                    local_descs, remote_descs, remote_agent, notif_msg, op
                )
            return handle  # In case of error it will be None
        else:
            return None

    def transfer(self, handle, notif_msg = ""):
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

    def release_xfer_handle(self, handle):
        # frees the handle too
        self.agent.releaseXferReq(handle)

    def release_dlist_handle(self, handle):
        # frees the handle too
        self.agent.releasedDlistH(handle)

    # Extra notification APIs
    def send_notif(self, remote_agent_name, notif_msg):
        self.agent.genNotif(remote_agent_name, notif_msg)

    # Returns new notifs, without touching self.notifs
    def get_new_notifs(self):
        return self.agent.getNotifs({})

    # Adds new notifs to self.notifs and returns it
    def update_notifs(self):
        self.notifs = self.agent.getNotifs(self.notifs)
        return self.notifs

    def get_agent_metadata(self):
        return self.agent.getLocalMD()

    def add_remote_agent(self, metadata):
        agent_name = self.agent.loadRemoteMD(metadata)
        return agent_name

    def remove_remote_agent(self, agent):
        self.agent.invalidateRemoteMD(agent)
