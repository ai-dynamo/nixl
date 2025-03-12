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

import nixl_bindings as nixl
import torch


class nixl_config:
    def __init(self, backends=["UCX", "GDS"]):
        self.backends = backends


class nixl_wrapper:
    def __init__(self, agent_name, nixl_config=None):
        # Read available backends and device info from nixl_config
        # For now setting the multithreading to enabled.
        devices = nixl.nixlAgentConfig(True)
        init = {}

        self.name = agent_name
        self.notifs = {}
        self.backends = {}
        self.agent = nixl.nixlAgent(agent_name, devices)
        if nixl_config:
            for x in nixl_config.backends:
                self.backends[x] = self.agent.createBackend(x, init)
        else:  # Defaulting to UCX and GDS for now
            self.backends["UCX"] = self.agent.createBackend("UCX", init)
            self.backends["GDS"] = self.agent.createBackend("GDS", init)

        self.nixl_mems = {
            "DRAM": nixl.DRAM_SEG,
            "VRAM": nixl.VRAM_SEG,
            "FILE": nixl.FILE_SEG,
            "BLOCK": nixl.BLK_SEG,
            "cpu": nixl.DRAM_SEG,
            "cuda": nixl.VRAM_SEG,
        }

        self.nixl_ops = {
            "WRITE": nixl.NIXL_WRITE,
            "READ": nixl.NIXL_READ,
            "WRITE_NOTIF": nixl.NIXL_WR_NOTIF,
            "READ_NOTIF": nixl.NIXL_RD_NOTIF,
        }

        print("Initializied NIXL agent:", agent_name)

    def get_xfer_descs(
        self, descs, mem_type=None, is_unified_addr=True, is_sorted=False
    ):
        # can add check for DLPack input

        if isinstance(descs, nixl.nixlXferDList):
            return descs
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 3:
                new_descs = nixl.nixlXferDList(
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
            new_descs = nixl.nixlXferDList(
                self.nixl_mems[str(tensor_type)],
                dlist,
                is_unified_addr,
                is_sorted,
            )
        elif isinstance(descs, nixl.nixlRegDList):
            print("RegList type detected for transfer, please use XferList")
            new_descs = None
        else:
            new_descs = None

        return new_descs

    def get_reg_descs(
        self, descs, mem_type=None, is_unified_addr=True, is_sorted=False
    ):
        # can add check for DLPack input

        if isinstance(descs, nixl.nixlRegDList):
            return descs
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 4:
                new_descs = nixl.nixlRegDList(
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
            new_descs = nixl.nixlRegDList(
                self.nixl_mems[str(tensor_type)],
                dlist,
                is_unified_addr,
                is_sorted,
            )
        elif isinstance(descs, nixl.nixlXferDList):
            print("XferList type detected for registration, please use RegList")
            new_descs = None
        else:
            new_descs = None

        return new_descs

    def get_serialized_descs(self, descs):
        return pickle.dumps(descs)

    def deserialize_descs(self, serialized_descs):
        return pickle.loads(serialized_descs)

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
            if (reg_descs.getType() == nixl.FILE_SEG) and ("GDS" in self.backend):
                ret = self.agent.registerMem(reg_descs, self.backends["GDS"])
            else:
                ret = self.agent.registerMem(reg_descs, self.backends["UCX"])
        if ret != 0:
            return None
        return reg_descs

    def deregister_memory(self, dereg_descs, backend=None):
        # based on backend type and mem_type, figure what deregistrations are needed
        if backend:
            self.agent.deregisterMem(dereg_descs, self.backends[backend])
        else:
            if (dereg_descs.getType() == nixl.FILE_SEG) and ("GDS" in self.backend):
                self.agent.deregisterMem(dereg_descs, self.backends["GDS"])
            else:
                self.agent.deregisterMem(dereg_descs, self.backends["UCX"])
        # No return

    # Optional proactive make connection
    def make_connection(self, remote_agent):
        self.agent.makeConnection(remote_agent)

    def get_agent_metadata(self):
        return self.agent.getLocalMD()

    def add_remote_agent(self, metadata):
        agent_name = self.agent.loadRemoteMD(metadata)
        return agent_name

    def remove_remote_agent(self, agent):
        self.agent.invalidateRemoteMD(agent)

    def initialize_xfer(
        self,
        local_descs,
        remote_descs,
        remote_agent,
        notif_msg,
        operation,
        xfer_backend=None,
    ):
        op = self.nixl_ops[operation + "_NOTIF" if len(notif_msg) != 0 else operation]
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

    # "" remote agent means local. example xfer can be used to know the backend
    def prep_xfer_side(
        self,
        remote_agent,
        xfer_list,
        mem_type=None,
        is_unified_addr=True,
        is_sorted=False,
        xfer_backend=None,
        example_xfer=None,
    ):
        descs = self.get_xfer_descs(xfer_list, mem_type, is_unified_addr, is_sorted)
        if xfer_backend:
            handle = self.agent.prepXferSide(descs, remote_agent, xfer_backend)
        elif example_xfer:
            backend = self.agent.getXferBackend(example_xfer)
            handle = self.agent.prepXferSide(descs, remote_agent, backend)
        else:
            if (descs.getType() == nixl.FILE_SEG) and ("GDS" in self.backend):
                handle = self.agent.prepXferSide(
                    descs, remote_agent, self.backends["GDS"]
                )
            else:
                handle = self.agent.prepXferSide(
                    descs, remote_agent, self.backends["UCX"]
                )
        return handle  # In case of error it will be None

    def make_prepped_xfer(
        self,
        local_xfer_side,
        local_indices,
        remote_xfer_side,
        remote_indices,
        notif_msg,
        operation,
    ):
        op = self.nixl_ops[operation + "_NOTIF" if len(notif_msg) != 0 else operation]
        if op:
            handle = self.agent.makeXferReq(
                local_xfer_side,
                local_indices,
                remote_xfer_side,
                remote_indices,
                notif_msg,
                op,
            )
            return handle  # In case of error it will be None
        else:
            return None

    def delete_xfer_side(self, handle):
        # frees the handle too
        self.agent.invalidateXferSide(handle)

    def transfer(self, handle):
        status = self.agent.postXferReq(handle)
        if status == nixl.NIXL_XFER_ERR:
            return "ERR"
        elif status != nixl.NIXL_XFER_DONE:
            return "PROC"
        else:
            return "DONE"

    def check_xfer_state(self, handle):
        status = self.agent.getXferStatus(handle)
        if status == nixl.NIXL_XFER_ERR:
            return "ERR"
        elif status != nixl.NIXL_XFER_DONE:
            return "PROC"
        else:
            return "DONE"

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

    def abort_xfer(self, handle):
        # frees the handle too
        self.agent.invalidateXferReq(handle)

    # Extra notification APIs
    def send_notif(self, remote_agent_name, notif_msg):
        self.agent.genNotif(remote_agent_name, notif_msg)

    # Adds new notifs to self.notifs and returns it
    def update_notifs(self):
        self.notifs = self.agent.getNotifs(self.notifs)
        return self.notifs

    # Returns new notifs, without touching self.notifs
    def get_new_notifs(self):
        return self.agent.getNotifs({})
