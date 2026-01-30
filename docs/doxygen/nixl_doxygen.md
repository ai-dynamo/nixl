<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# NVIDIA Inference Xfer Library (NIXL)
NIXL is targeted for accelerating point to point communications in AI inference frameworks such as [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo), while providing an abstraction over various types of memory (e.g., CPU and GPU) and storage (e.g., file, block and object store) through a modular plug-in architecture.

# Background
Distributed inference workloads present complex challenges for systems, including but not limited to networking and communication issues. These challenges encompass high-performance requirements, heterogeneous data paths that span both memory and storage, and the need for dynamic scaling up and down.

NIXL is designed to support inference frameworks by addressing their challenges while delivering high-bandwidth, low-latency point-to-point data transfers. It offers a unified abstraction across various memory types, including HBM, DRAM, local or remote SSDs, and distributed storage systems, through a versatile API. This API can support multiple backend plugins like UCX, GDS, S3, and other protocols or clients. Furthermore, NIXL abstracts away additional backend specifics, such as connection management, addressing schemes, and memory characteristics, to streamline integration with inference frameworks.

# Overview

The following figure illustrates NIXL's relationship to inference server stacks. NIXL functions as a standalone library, providing the necessary abstraction for various network and storage backends to support the data plane operations of distributed inference platforms, such as NVIDIA Dynamo. The currently supported backends in NIXL are [UCX](https://github.com/openucx/ucx) and [NVIDIA Magnum IO GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html). Support for other file systems (including DFS), block and object storage are in development. NIXL offers generic interfaces capable of supporting data transfers in the form of tensors, bytes, or objects.

\image html doxygen/nixl.png "Figure of NIXL high level architecture" width=500px
\image latex doxygen/nixl.png "Figure of NIXL high level architecture" width=500px

NIXL operates under the assumption that it is managed by a conductor process responsible for orchestrating the inference process. This includes tasks such as memory allocations, handling user requests, and providing the means to exchange the necessary serialized metadata through a side-channel or a central metadata service such as "etcd" or Redis.


# Design
NIXL's Transfer Agent abstracts away three key entities:

1. [Memory Section](#memory-sections): Unifies various types of memory and storage, allowing the agent to accept a buffer list primitive for transactions regardless of the memory type.
2. [Transfer Backend Interface](#transfer-backend-interface): Abstracts different transfer backends from the agent, facilitating efficient transfers by selecting the most optimal backend.
3. [Metadata Handler](#metadata-handler): Encapsulates the information required to perform data transfers between NIXL agents running on different workers (e.g., a single GPU or a DGX server). It also caches required remote agent information for the local agent.

Using these components, the NIXL Transfer Agent provides the desired interface by receiving a buffer list primitive from the distributed inference conductor and returning an asynchronous handle to check the status. The agent supports multiple backends, allowing seamless data movement via [RoCE](https://docs.nvidia.com/networking/display/ofedv502180/rdma+over+converged+ethernet+(roce)), [InfiniBand](https://www.nvidia.com/en-us/networking/products/infiniband/), [GPUDirect RDMA](https://developer.nvidia.com/gpudirect), NVMe-oF, TCP, [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/), and potentially file system emulation. NIXL determines the optimal transfer backend based on the memory types used in a transfer, as well as common backends available in both agents. For example, if the source is DRAM and the target is VRAM, UCX might be used. If the transfer involves VRAM as the source and PFS as the backend, GPUDirect Storage APIs could be employed. In cases where multiple backends that can perform a transfer request, NIXL internally decides which one to use.

A NIXL agent is instantiated within each inference conductor process that may be managing one or more GPU devices. For instance, on a DGX server, a single agent in the main conductor process can access several GPUs and other memory or storage nodes. Each agent is identified by a unique and global ID/name assigned by the inference platform.

### Memory Sections
A Memory Section is a mixture of address ranges (segments) registered with the agent. NIXL supports multiple segment types, including DRAM, VRAM, NVMe-oF, Object storage, and File. The Memory Section comprises the local information required by the transfer backend engines as well the required identifiers for remote agents to access the corresponding segments.

### Transfer Backend Interface
Each transfer backend must be initialized for the corresponding transfer agent. This process enables the transfer agent to keep track of the available transfer engines.
Using this information, the most suitable backend for a transfer can be determined based on segment descriptors. For example, it is possible for the same memory location to be registered with multiple backends. In this case, NIXL agent will select the optimal one based on the source and destination memory types, as well as the available backends on the remote node.

### Metadata Handler
The Metadata Handler manages the data necessary for establishing data communication between the NIXL agents. This metadata can be exchanged via a secure side channel or through a centralized metadata server like "etcd" or Redis. The metadata includes connection information for each backend, as well as remote segment identifiers. The metadata shared with remote NIXL agents is minimal and excludes any local information that is not necessary on the remote side.
When loading metadata, the backend type that generated the remote identifiers is included within the serialized metada. This ensures that the received metadata is routed to the appropriate backend on the receiving agent. If that backend is not available, the relevant metadata portion is ignored.

For example, a memory section tagged with UCX indicates it was generated by a UCX backend engine and should be delivered to a UCX backend engine on the receiving agent, if present.

In the optimal mode of operation, the metadata exchange occurs on the control path (not the data path). It is advised to register the memory segments that are being used within transfers with the NIXL agent during application initialization time so the metadata is exchanged only once.

To support the dynamic nature of inference workloads, metadata caching is leveraged. This approach allows to:

* Avoid fetching the same metadata on each transfer
* Dynamically add new agents by providing its metadata to the caching subsystem of the agents that communicate with it.
* Dynamically exclude an agent by invalidating the caches of the agent that previously interacted with it

Adding a remote agent metadata does not cause a connection to be initiated, as this might be just a prefetch optimization. If desired, there is an optional connection API for this usage. Conversely, removing a remote agent metadata will result in a disconnect, if a connection was already established.

# Example procedure
The following image shows a basic example of how to use NIXL between 2 nodes. The API usage is further reviewed in three parts: initialization, transfer, and dynamicity.

\image html doxygen/nixl_two_nodes.png "Example of NIXL with two nodes" width=500px
\image latex doxygen/nixl_two_nodes.png "Example of NIXL with two nodes" width=500px

## Initialization
For each node in the system, the runtime creates an agent, and if necessary gives the list of devices to it (not shown in the above example). Then for each desired transfer backend that is supported for the system, it calls the create_transfer_backend API. Next, the NIXL segments are created by each relevant backend. Optionally, user can specify a list of backends or no backend, and NIXL agent will register the segment with those backends, or all backends that support that memory/storage type respectively. For instance, in the above example, we register all GPU HBM memory with UCX.


```
# list of devices (GPUs, DRAM, NICs, NVMe, etc)
create_agent(name, optional_devices)

# User allocates memory from their preferred devices -> alloced_mem
foreach transfer_backend:
    create_transfer_backend(corresponding_backend_init)
foreach alloced_mem:
    register_memory (desc_list) # or list of desired backend can be specified
```

## Metadata Exchange
Once backends and memory regions are registered with NIXL, the runtime queries the metadata from each agent, either directly or by sending it to a central metadata server. This metadata is necessary for initiator agents to connect to target agents and facilitate data transfers between them. In the example provided, metadata is exchanged directly without a metadata server. However, agent A's metadata can also be sent to agent B if B needs to initiate a transfer to A.
Following the metadata exchange, which includes connection information for each registered backend that can talk to remote agents, the runtime can proactively call the make_connection API using the target agent's name if the agents involved in the transfer are known in advance. This will make a connection between all common backends between the two agents. Otherwise, the connection is established during the first transfer. This functionality is optional and up to the backend to specify what can happen during this stage.

### Side channel
```
# In each agent:
get_local_metadata()

# Exchange the metadata to the desired agents

# In each agent:
    for each received metadata:
        remote_agent_name = load_remote_metadata()
        make_connection(remote_agent_name) # optional

```

### Central metadata
```
# In each agent:
send_local_metadata()

for each target_agent:
    fetch_remote_metadata(remote_agent_name)
    make_connection(remote_agent_name) # optional
```

## Transfer
To initiate a transfer, the initiator must provide a list of local buffer descriptions and a list of remote buffer descriptors. The remote buffers can be communicated out of band. Both the local and remote buffers should be within the registered memories of their corresponding NIXL agent. The initiator agent checks the remote addresses based on the information available in the exchanged metadata. Using these descriptor lists, along with the target agent's name and the transfer operation (read or write), a transfer handle can be created. Optionally, a notification message can be specified for the operation at this time.

The create_xfer_req function performs the necessary checks for the transfer and determines which backend will be used, unless a backend is specifically requested. If successful, the runtime can subsequently post a transfer request described by the received handle one or more times (per transfer handle, there can be only 1 active transfer at any given point in time to avoid data corruption). The notification message can be given during post instead of handle creation too, or changed per repost. The status of the transfer can be checked using the get_xfer_status function. In the example provided, the process blocks on the transfer checks since it is the only transfer in the system and there is no computation to overlap it with.

```
# On initiator agent
hdl = create_xfer_req(operation (RD/WR),
                      local_descs, target_descs,
                      target_agent_name, notif_msg)

# Example of reposting the same transfer request
for n iterations:
    post_transfer_request(hdl)
    while (get_xfer_status(hdl) != complete):
        # do other tasks, non-blocking

```

## Adding/removing agents (dynamic scaling)
Adding a new agent to a service involves creating the agent and exchanging its metadata with the existing agents in the service. To remove an agent or handle a failure, you can use one of the metadata invalidate APIs. This triggers disconnections for backends connected to the agent and purges the cached metadata values.

### Side channel

```
# In each agent (the connected ones or all, both options are fine)
invalidate_remote_agent(remote_agent_name)

# In case of failure, the heartbeat system sends this message
# to connected or all agents through the side channel control
```

### Central metadata

```
# In the agent that is going to be removed
Invalidate local agent metadata()

# In case of failure, the heartbeat system sends this message
# to the central metadata server
```

## Teardown
Similar to removing an agent, invalidating an agent's metadata is necessary through one of the two APIs. When an agent is destroyed (via its destructor), it deregisters all the remaining registered memory regions (not already deregistered by the user), calls destructors of each backend which invalidate their internal transfer states, and releases other internal resources in the agent.

### Side channel

```
# In each agent (the connected ones or all, both options are fine)
invalidate_remote_agent(remote_agent_name)

# In the agent that is going to be deleted
deregister memory regions # optional
delete agent
```

### Central metadata

```
# In the agent that is going to be removed
Invalidate local agent metadata()
deregister memory regions # optional
delete agent
```
