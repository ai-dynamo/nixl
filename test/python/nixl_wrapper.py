import nixl_bindings as nixl
import torch
import pickle

class nixl_wrapper:

    def __init__(self, agent_name, nixl_config):
        # Read available backends and device info from nixl_config
        # For now setting the multithreading to enabled.
        devices = nixl.nixlAgentConfig(False)
        init = nixl.nixlUcxInitParams()

        self.name = agent_name
        self.notifs = {}
        self.backends = {}
        self.agent = nixl.nixlAgent(agent_name, devices)
        self.backends["UCX"] = self.agent.createBackend(init)

        self.nixl_mems = {"DRAM":       nixl.DRAM_SEG,
                          "VRAM":       nixl.VRAM_SEG,
                          "cpu":        nixl.DRAM_SEG,
                          "cuda":       nixl.VRAM_SEG}
        self.nixl_ops = {"WRITE":       nixl.NIXL_WRITE,
                         "READ":        nixl.NIXL_READ,
                         "WRITE_NOTIF": nixl.NIXL_WR_NOTIF,
                         "READ_NOTIF":  nixl.NIXL_RD_NOTIF}

        print("Initializied NIXL agent:", agent_name)


    # TODO: Maybe we should add unifiedAddr flag, and remove from register/prep.
    def get_descs(self, arg, is_sorted=False):
        # can add check for DLPack input
        if isinstance(arg, list): # List[torch.Tensor]:
            tensor_type = arg[0].device
            dlist = [ (0,0,0) ] * len(arg)

            for i in range(len(arg)):
                if tensors[i].device != tensor_type:
                    return None
                base_addr = tensors[i].data_ptr()
                region_len = tensor[i].numel() * tensors[i].element_size()
                gpu_id = tensors[i].get_device()
                if (gpu_id==-1): # DRAM
                    gpu_id = 0
                dlist[i] = (base_addr, region_len, gpu_id)
            descs = nixl.nixlDescList(self.nixl_mems[tensor_type], dlist, True, is_sorted)


        elif isinstance(arg, tuple): # (str, List[(int,int,int)]):
            descs = nixl.nixlDescList(self.nixl_mems[arg[0]], arg[1], True, is_sorted)

        elif isinstance(arg, nixl.nixlDescList):
            return arg

        return descs


    def get_serialized_descs(self, descs):
        return pickle.dumps(descs)


    def deserialize_descs(self, serialized_descs):
        return pickle.loads(serialized_descs)


    # The returned descriptor object can be used for call to deregister
    def register_memory(self, arg, is_sorted=False):
        # based on backend type and mem_type, figure what registrations are meaningful
        reg_descs = self.get_descs(arg, is_sorted)
        ret = self.agent.registerMem(reg_descs, self.backends["UCX"])
        if (ret != 0):
            return None
        return reg_descs


    def deregister_memory(self, dereg_descs):
        # based on backend type and mem_type, figure what deregistrations are needed
        self.agent.deregisterMem(dereg_descs, self.backends["UCX"])
        # No return


    # Optional proactive make connection
    def make_connection(self, remote_agent):
        self.agent.makeConnection(remote_agent)


    def get_agent_metadata(self):
        return self.agent.getLocalMD()


    def add_remote_agent(self, metdata):
        agent_name = self.agent.loadRemoteMD(metdata)
        return agent_name


    def remove_remote_agent(self, agent):
        self.agent.invalidateRemoteMD(agent)


    def initialize_xfer(self, local_descs, remote_descs, remote_agent, notif_msg, operation):
        op = self.nixl_ops[operation+"_NOTIF" if len(notif_msg)!=0 else operation]
        if (op):
            handle = self.agent.createXferReq(local_descs, remote_descs, remote_agent, notif_msg, op)
            return handle # In case of error it will be None
        else:
            return None


    # "" remote agent means local. example xfer can be used to know the backend
    def prep_xfer_side(self, args, remote_agent="", is_sorted=False, xfer_backend=None, example_xfer=None):
        descs = self.get_descs(args, is_sorted)
        if (xfer_backend):
            handle = self.agent.prepXferSide(descs, remote_agent, xfer_backend);
        elif (example_xfer):
            backend = self.agent.getXferBackend(example_xfer);
            handle = self.agent.prepXferSide(descs, remote_agent, backend);
        else:
            # Or use same logic that we used in register_memory
            handle = self.agent.prepXferSide(descs, remote_agent, self.backends["UCX"]);
        return handle # In case of error it will be None


    def make_prepped_xfer(self, local_xfer_side, local_indices, remote_xfer_side, \
                        remote_indices, notif_msg, operation):
        op = self.nixl_ops[operation+"_NOTIF" if len(notif_msg)!=0 else operation]
        if (op):
            handle = self.agent.makeXferReq(local_xfer_side, local_indices, remote_xfer_side, \
                                            remote_indices, notif_msg, op)
            return handle # In case of error it will be None
        else:
            return None


    def delete_xfer_side (self, handle):
        # frees the handle too
        self.agent.invalidateXferSide(handle)


    def transfer(self, handle):
        status = self.agent.postXferReq(handle)
        if status==nixl.NIXL_XFER_ERR:
            return "ERR"
        elif (status!=nixl.NIXL_XFER_DONE):
            return "PROC"
        else:
            return "DONE"


    def check_xfer_state (self, handle):
        status = self.agent.getXferStatus(handle)
        if status==nixl.NIXL_XFER_ERR:
            return "ERR"
        elif (status!=nixl.NIXL_XFER_DONE):
            return "PROC"
        else:
            return "DONE"


    # Only removes the specific notification from self.notifs
    def check_remote_xfer_done (self, remote_agent_name, lookup_msg):
        self.notifs = self.agent.getNotifs(self.notifs) # Adds new notifs
        message = None
        if remote_agent_name in self.notifs:
            for msg in self.notifs[remote_agent_name]:
                if lookup_msg in msg:
                    message = msg
                    break
        if message:
            self.notifs[remote_agent_name].remove(message)
        return message


    def abort_xfer (self, handle):
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
